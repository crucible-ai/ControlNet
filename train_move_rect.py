import json
import os
import random
from glob import glob
from typing import Optional, Tuple

import numpy
import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image, ImageDraw
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from cldm.ddim_hacked import DDIMSampler
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from share import *


class MoveRectDataset(Dataset):
    def __init__(
            self,
            image_size: int = 512,
            image_path_glob: str = "training/move_rect_dataset/*",
            rotation_max_degrees: float = 0.0,
            scale_min: float = 0.5,
            scale_max: float = 0.8,
    ):
        self.image_size = image_size
        self.image_filenames = glob(image_path_glob)
        self.rotation_max_degrees = rotation_max_degrees
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]

        stamp_image = Image.open(filename).convert("RGBA")  # Convert to RGBA because we have to do some compositing.
        # Resize the image so that the max edge is smaller than our base size.
        scale_factor = self.image_size/max(stamp_image.size)
        stamp_image = stamp_image.resize((1+int(stamp_image.size[0]*scale_factor), 1+int(stamp_image.size[1]*scale_factor)))

        # Make our source image somewhere.  We don't care about the parameters of the translation.
        source_image, _, _, _ = self.make_composite_image(stamp_image, None, None, None, None)

        # Make our target image, keeping track of the different transforms.  We'll need them to write our rect onto the source.
        target_image, target_scale, target_rotation, target_translation = self.make_composite_image(stamp_image, None, None, None, None)

        # Now make a rectangle that will be our second stamp.
        rectangle_stamp = Image.new("RGBA", stamp_image.size)
        canvas = ImageDraw.Draw(rectangle_stamp)
        canvas.rectangle(
            xy=(0, 0, rectangle_stamp.size[0]-1, rectangle_stamp.size[1]-1),
            fill=None,
            outline=(255, 255, 255, 255),
            width=2,
        )

        source_image, _, _, _ = self.make_composite_image(rectangle_stamp, source_image, target_scale, target_rotation, target_translation)

        source = numpy.asarray(source_image.convert("RGB"))
        target = numpy.asarray(target_image.convert("RGB"))

        # Normalize source images [0, 1].
        source = source.astype(numpy.float32) / 255.0
        target = (target.astype(numpy.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=random.choice(["an image", "a picture", ".", " "]), hint=source)

    def make_composite_image(
            self,
            stamp_image: Image.Image,
            out_image: Optional[Image.Image],
            scale: Optional[float],
            rotation: Optional[float],
            translation: Optional[Tuple[int, int]]
    ) -> Tuple[Image.Image, float, float, Tuple[int, int]]:
        """Given a stamp, a scale amount, a rotation amount, and an offset, stamp the given image onto the output.
        If scale, rotation, and translation are not given, compute them based on the internal parameters."""
        if stamp_image.mode != "RGBA":
            stamp_image = stamp_image.convert("RGBA")  # We need this because we use alpha during rotation.

        if out_image is None:
            out_image = Image.new("RGBA", (self.image_size, self.image_size))
        elif out_image.mode != "RGBA":
            out_image = out_image.convert("RGBA")

        if scale is None:
            scale = (random.random() * (self.scale_max - self.scale_min)) + self.scale_min
        scaled_stamp_image = stamp_image.resize(
            (int(stamp_image.size[0] * scale), int(stamp_image.size[1] * scale))
        )

        if rotation is None:
            rotation = ((random.random() - 0.5) * 2.0) * self.rotation_max_degrees
        scaled_rotated_stamp_image = scaled_stamp_image.rotate(rotation, expand=True, fillcolor=(0, 0, 0, 0))  # ALPHA!

        if translation is None:
            dx_space = out_image.size[0] - scaled_rotated_stamp_image.size[0]  # TODO: What do we do if dx_space < 0?
            dy_space = out_image.size[1] - scaled_rotated_stamp_image.size[1]
            translation = (
                random.randrange(0, int(dx_space)),
                random.randrange(0, int(dy_space)),
            )

        out_image.paste(scaled_rotated_stamp_image, translation, mask=scaled_rotated_stamp_image)  # Mask will auto-use alpha.
        return out_image, scale, rotation, translation


# Configs
resume_path = './models/controlnet_sd15_moverect.ckpt'
batch_size = 8
logger_freq = 2500
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Save every so often:
ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath="./checkpoints/",
        filename="controlnet_sd15_moverect_face_{epoch}_{step}_{loss_simple_step}",
        monitor='train/loss_simple_step',
        save_top_k=5,
        every_n_train_steps=5000,
        save_last=True,
)


# Make a logging callback to store a sample image to the tensorboard.
class TensorboardDiffusionImageSampler(pl.callbacks.Callback):
    def __init__(self, ds: MoveRectDataset) -> None:
        super().__init__()
        self.ds = ds

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        #dim = (self.num_samples, pl_module.hparams.latent_dim)
        #z = torch.normal(mean=0.0, std=1.0, size=dim, device=pl_module.device)

        example_dict = self.ds[random.randint(0, len(self.ds)-1)]
        with torch.no_grad():
            pl_module.eval()
            single_image = TensorboardDiffusionImageSampler.run_inference(
                model=pl_module,
                image_tensor=example_dict['hint'].unsqueeze(0)  # Batch of '1'.
            )
            pl_module.train()
        source_prediction_expected = torch.stack([example_dict['hint'], single_image.squeeze(0), example_dict['jpg']], dim=0)

        grid = torchvision.utils.make_grid(
            tensor=source_prediction_expected,
            nrow=3,
            padding=2,
            normalize=True,
            scale_each=True,  # Scale all images to 0,1 individually, not together.
        )
        str_title = f"{pl_module.__class__.__name__}_images"
        trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)

    @staticmethod
    def run_inference(model, image_tensor, prompt, n_prompt: str = "", ddim_steps: int = 20, guidance_scale: float = 9.0):
        sampler = DDIMSampler(model)
        with torch.no_grad():
            B, C, H, W = image_tensor.shape
            cond = {"c_concat": [image_tensor], "c_crossattn": [model.get_learned_conditioning([prompt] * B)]}
            un_cond = {"c_concat": image_tensor, "c_crossattn": [model.get_learned_conditioning([n_prompt] * B)]}
            shape = (4, H // 8, W // 8)
            model.control_scales = ([1.0] * 13)
            samples, intermediates = sampler.sample(ddim_steps, B, shape, cond, verbose=False, eta=0,
                                                    unconditional_guidance_scale=guidance_scale,
                                                    unconditional_conditioning=un_cond)
            result = model.decode_first_stage(samples)
            return result  # [B, C, H, W], same format image_grid requires.


# Misc
dataset = MoveRectDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
image_logger_callback = ImageLogger(batch_frequency=logger_freq)
tensorboard_image_callback = TensorboardDiffusionImageSampler(ds=dataset)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[image_logger_callback, ckpt_callback, tensorboard_image_callback])

# Train!
trainer.fit(model, dataloader)
