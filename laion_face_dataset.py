import json
import numpy
import os
from PIL import Image
from torch.utils.data import Dataset


class LaionDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/laion-face-processed/prompt.jsonl', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = os.path.split(item['source'])[-1]
        target_filename = os.path.split(item['target'])[-1]
        prompt = item['prompt']

        source_image = Image.open('./training/laion-face-processed/source/' + source_filename).convert("RGB")
        target_image = Image.open('./training/laion-face-processed/target/' + source_filename).convert("RGB")
        # Resize the image so that the minimum edge is bigger than 512x512, then crop center.
        img_size = source_image.size
        scale_factor = 512/min(img_size)
        source_image = source_image.resize((1+int(img_size[0]*scale_factor), 1+int(img_size[1]*scale_factor)))
        target_image = target_image.resize((1+int(img_size[0]*scale_factor), 1+int(img_size[1]*scale_factor)))
        img_size = source_image.size
        left_padding = (img_size[0] - 512)//2
        top_padding = (img_size[1] - 512)//2
        source_image = source_image.crop((left_padding, top_padding, left_padding+512, top_padding+512))
        target_image = target_image.crop((left_padding, top_padding, left_padding+512, top_padding+512))

        source = numpy.asarray(source_image)
        target = numpy.asarray(target_image)

        # Normalize source images to [0, 1].
        source = source.astype(numpy.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(numpy.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

