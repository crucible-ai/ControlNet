import json
import numpy
import os
from abc import ABC, abstractmethod
from typing import Generator, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from annotator.facenet import FaceNet


BASE_TRAINING_PATH = os.environ.get("BASE_TRAINING_PATH", "./training/laion-face-processed/")
# For LAION ./training/laion-face-processed/.  Should have metadata.json, prompt.json, source, and target.
# For MSCOCO2017, ./training/mscoco/.  Should have annotations (with a bunch of JSON) and train2017 (with *.jpg).


class _BaseFacenetDataset(Dataset, ABC):

    @classmethod
    @abstractmethod
    def iterate_candidate_targets_and_captions(cls):
        # Generator -- tuples of filename, prompt.
        # Targets should not include BASE_TRAINING_PATH, but should include subdir/image_filename.png
        yield None, None

    def is_valid_example(self, target_filename, caption) -> bool:
        # Return true if this is usable in the output.
        try:
            if not caption:
                return False
            target_filename = os.path.join(BASE_TRAINING_PATH, target_filename)
            target_image = self.crop_512(target_filename)
            source_image = self.model.create_embedding_image(target_image, empty_image_on_failure=False)
            #img = Image.open(target_filename).convert("RGB")
            #faces, probabilities, bboxes, _ = self.model.detector(img)
            if source_image is None:
                return False
            return True
        except InterruptedError:
            raise
        except IOError:
            return False

    def __init__(self, model_path: os.PathLike, cache_filename: str):
        self.model = FaceNet(model_path=model_path)
        self.image_caption_pairs = list()  # image filename, caption
        # Not all images in X have people or faces.  We run through all images and discard the files that don't.
        # Also, cache this in the temp directory.  If we're restarting, just reload the old file.
        cached_file_list = os.path.join(BASE_TRAINING_PATH, cache_filename)
        if os.path.exists(cached_file_list) and os.path.isfile(cached_file_list):
            with open(cached_file_list, 'rt') as fin:
                cached_data = json.load(fin)
                self.image_caption_pairs = cached_data['image_caption_pairs']
                # After loading, JSON tends to cast int values to strings.
                self.image_caption_pairs = [(x[0], x[1]) for x in self.image_caption_pairs]
        else:
            for target_filename, caption in self.iterate_candidate_targets_and_captions():
                if self.is_valid_example(target_filename, caption):
                    self.image_caption_pairs.append((target_filename, caption))
            with open(cached_file_list, 'wt') as fout:
                json.dump({'image_caption_pairs': self.image_caption_pairs}, fout)

    @classmethod
    def crop_512(cls, image_filename: str) -> Image.Image:
        target_image = Image.open(image_filename).convert("RGB")
        # Resize the image so that the minimum edge is bigger than 512x512, then crop center.
        # This may cut off some parts of the face image, but in general they're smaller than 512x512 and we still want
        # to cover the literal edge cases.
        img_size = target_image.size
        scale_factor = 512 / min(img_size)
        target_image = target_image.resize((1 + int(img_size[0] * scale_factor), 1 + int(img_size[1] * scale_factor)))
        img_size = target_image.size
        left_padding = (img_size[0] - 512) // 2
        top_padding = (img_size[1] - 512) // 2
        target_image = target_image.crop((left_padding, top_padding, left_padding + 512, top_padding + 512))
        return target_image

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, idx):
        image_filename, prompt = self.image_caption_pairs[idx]

        image_filename = os.path.join(BASE_TRAINING_PATH, image_filename)
        target_image = self.crop_512(image_filename)
        source_image = self.model.create_embedding_image(target_image)

        # Convert both to tensors.
        source = numpy.asarray(source_image)
        target = numpy.asarray(target_image)

        # Normalize source images to [0, 1].
        source = source.astype(numpy.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(numpy.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


class FaceNetLaionFaceDataset(_BaseFacenetDataset):
    def __init__(self, model_path: os.PathLike):
        super().__init__(model_path=model_path, cache_filename="laion_cached_images_with_people.json")

    @classmethod
    def iterate_candidate_targets_and_captions(cls):
        with open(os.path.join(BASE_TRAINING_PATH, 'prompt.jsonl'), 'rt') as f:
            for line in f:
                example = json.loads(line)
                prompt = example["prompt"] or "an image"
                original_path = example["target"]
                filename = os.path.basename(original_path)
                new_path = os.path.join("target", filename)
                yield new_path, prompt


class FaceNetMSCOCODataset(_BaseFacenetDataset):
    def __init__(self, model_path: os.PathLike):
        super().__init__(model_path=model_path, cache_filename="mscoco_cached_images_with_people.json")

    @classmethod
    def iterate_candidate_targets_and_captions(cls):
        with open(os.path.join(BASE_TRAINING_PATH, 'annotations', 'captions_train2017.json'), 'rt') as f:
            raw_data = json.load(f)
        # First: load and remap data.
        image_id_to_filename = {int(data["id"]): data["file_name"] for data in raw_data["images"]}
        for sample in raw_data["annotations"]:
            image_id = int(sample["image_id"])  # NOTE: Image ID != Caption ID
            caption = sample["caption"]
            if not caption:
                caption = "an image"
            if image_id not in image_id_to_filename:
                print(f"Got caption for image {image_id}, but image not found in image list.")
                continue
            filename = os.path.join("train2017", image_id_to_filename[image_id])
            yield filename, caption

