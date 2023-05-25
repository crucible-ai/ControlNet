import json
import os
import random
from glob import glob
from typing import Optional, Tuple

import numpy
import pytorch_lightning as pl
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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
        target = target.astype(numpy.float32) / 255.0

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
        filename="controlnet_sd15_moverect_face_{epoch}_{step}_{loss_simple_step}.ckpt",
        monitor='train/loss_simple_step',
        save_top_k=5,
        every_n_train_steps=5000,
        save_last=True,
)

# Misc
dataset = MoveRectDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, ckpt_callback])

# Train!
trainer.fit(model, dataloader)