import json
import numpy
import os
import tempfile

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from annotator.facenet import FaceNet


BASE_TRAINING_PATH = os.environ.get("BASE_TRAINING_PATH", "./training/laion-face-processed/")
# For LAION ./training/laion-face-processed/.  Should have metadata.json, prompt.json, source, and target.
# For MSCOCO2017, ./training/mscoco/.  Should have annotations (with a bunch of JSON) and train2017 (with *.jpg).


class FaceNetLaionFaceDataset(Dataset):
    def __init__(self, model_path: os.PathLike):
        self.model = FaceNet(model_path=model_path)
        self.data = []
        with open(os.path.join(BASE_TRAINING_PATH, 'prompt.jsonl'), 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = os.path.split(item['source'])[-1]
        target_filename = os.path.split(item['target'])[-1]
        prompt = item['prompt']

        # If prompt is "" or null, make it something simple.
        if not prompt:
            print(f"Image with index {idx} / {source_filename} has no text.")
            prompt = "an image"

        target_image = Image.open(os.path.join(BASE_TRAINING_PATH, 'target', target_filename)).convert("RGB")
        # Resize the image so that the minimum edge is bigger than 512x512, then crop center.
        # This may cut off some parts of the face image, but in general they're smaller than 512x512 and we still want
        # to cover the literal edge cases.
        img_size = target_image.size
        scale_factor = 512/min(img_size)
        target_image = target_image.resize((1+int(img_size[0]*scale_factor), 1+int(img_size[1]*scale_factor)))
        img_size = target_image.size
        left_padding = (img_size[0] - 512)//2
        top_padding = (img_size[1] - 512)//2
        target_image = target_image.crop((left_padding, top_padding, left_padding+512, top_padding+512))
        source_image = self.model.create_embedding_image(target_image)

        # Convert both to tensors.
        source = numpy.asarray(source_image)
        target = numpy.asarray(target_image)

        # Normalize source images to [0, 1].
        source = source.astype(numpy.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(numpy.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


class FaceNetMSCOCODataset(Dataset):
    def __init__(self, model_path: os.PathLike):
        self.model = FaceNet(model_path=model_path)
        self.image_caption_pairs = list()
        self.image_id_to_filename = dict()
        # Not all images in MSCOCO have people or faces.  We run through all images and discard the files that don't.
        # Also, cache this in the temp directory.  If we're restarting, just reload the old file.
        tempdir = tempfile.gettempdir()
        cached_file_list = os.path.join(tempdir, "mscoco_cached_images_with_people.json")
        if os.path.exists(cached_file_list) and os.path.isfile(cached_file_list):
            with open(cached_file_list, 'rt') as fin:
                cached_data = json.load(fin)
                self.image_id_to_filename = cached_data['image_to_filename']
                self.image_caption_pairs = cached_data['image_caption_pairs']
                # After loading, JSON tends to cast int values to strings.
                self.image_id_to_filename = {int(k): v for k, v in self.image_id_to_filename.items()}
                self.image_caption_pairs = [(int(x[0]), x[1]) for x in self.image_caption_pairs]
        else:
            print("Preprocessing images in captions_train2017 and removing non-face images.")
            with open(os.path.join(BASE_TRAINING_PATH, 'annotations', 'captions_train2017.json'), 'rt') as f:
                raw_data = json.load(f)
            # First: load and remap data.
            self.image_id_to_filename = {int(data["id"]): data["file_name"] for data in raw_data["images"]}
            for sample in raw_data["annotations"]:
                image_id = int(sample["image_id"])  # NOTE: Image ID != Caption ID
                caption = sample["caption"]
                if image_id not in self.image_id_to_filename:
                    print(f"Got caption for image {image_id}, but image not found in image list.")
                    continue
                # Ordering is important here.  If these lines get swapped, it's len(self.captions)-1.
                self.image_caption_pairs.append((image_id, caption))

            # Second: Filter images without faces.
            image_ids_to_remove = set()
            for image_id, filename in self.image_id_to_filename.items():
                image = Image.open(os.path.join(BASE_TRAINING_PATH, "train2017", filename))
                image = image.convert("RGB")
                faces, probabilities, bboxes, _ = self.model.detector(image)
                if faces is None:
                    image_ids_to_remove.add(image_id)
            self.image_caption_pairs = [pair for pair in self.image_caption_pairs if pair[0] not in image_ids_to_remove]
            for image_id_to_remove in image_ids_to_remove:
                del self.image_id_to_filename[image_id_to_remove]

            # Third: cache the remainder.
            with open(cached_file_list, 'wt') as fout:
                json.dump({
                    "image_to_filename": self.image_id_to_filename,
                    "image_caption_pairs": self.image_caption_pairs,
                }, fout)

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, idx):
        image_id, prompt = self.image_caption_pairs[idx]
        image_filename = self.image_id_to_filename[image_id]

        # If prompt is "" or null, make it something simple.
        if not prompt:
            print(f"Image with index {idx} / {image_filename} has no text.")
            prompt = "an image"

        full_filename = os.path.join(BASE_TRAINING_PATH, "train2017", image_filename)
        target_image = Image.open(full_filename).convert("RGB")
        # Resize the image so that the minimum edge is bigger than 512x512, then crop center.
        # This may cut off some parts of the face image, but in general they're smaller than 512x512 and we still want
        # to cover the literal edge cases.
        img_size = target_image.size
        scale_factor = 512/min(img_size)
        target_image = target_image.resize((1+int(img_size[0]*scale_factor), 1+int(img_size[1]*scale_factor)))
        img_size = target_image.size
        left_padding = (img_size[0] - 512)//2
        top_padding = (img_size[1] - 512)//2
        target_image = target_image.crop((left_padding, top_padding, left_padding+512, top_padding+512))

        source_image = self.model.create_embedding_image(target_image)

        # Convert both to tensors.
        source = numpy.asarray(source_image)
        target = numpy.asarray(target_image)

        # Normalize source images to [0, 1].
        source = source.astype(numpy.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(numpy.float32) / 127.5) - 1.0

        # Expand the dimensions since both of these are luma.
        source = numpy.expand_dims(source, axis=0)
        target = numpy.expand_dims(target, axis=0)

        return dict(jpg=target, txt=prompt, hint=source)

