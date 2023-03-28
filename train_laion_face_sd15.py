from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from laion_face_dataset import LaionDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
resume_path = './models/controlnet_sd15_laion_face.ckpt'
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
        filename="controlnet_sd15_laion_face_{epoch}_{step}_{loss}.ckpt",
        monitor='train/loss_simple_step',
        save_top_k=5,
        every_n_train_steps=5000,
        save_last=True,
)

# Misc
dataset = LaionDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, ckpt_callback])

# Train!
trainer.fit(model, dataloader)