import math
import os
from typing import Mapping, Tuple

import numpy
import torch
from PIL import Image, ImageDraw

from .models.mtcnn import MTCNN
from .models.inception_resnet_v1 import InceptionResnetV1


EMBEDDING_SIZE = 512
PIXEL_FORMAT = 'RGB'  # ControlNet expected RGB data.


class FaceNet:
    def __init__(self, model_path=None):
        device = torch.device("cpu")
        # Note: image-size is the desired OUTPUT image size after a crop is performed.
        # It gives the square width of x_aligned, probs = mtcnn(img...)
        self.detector = MTCNN(
            output_image_size=160,
            keep_all=True,
            device=device,
            model_path=model_path or os.path.join(os.path.dirname(__file__), "data")
        )
        # TODO: We should/could reuse mediapipe face here, but I'm not sure about the alignment.
        self.encoder = InceptionResnetV1(
            pretrained='vggface2',
            classify=False,
            device=device,
            model_path=model_path or os.path.join(os.path.dirname(__file__), "data")
        ).eval()  # .to(device)

    def __call__(self, img, bbox_to_identity: Mapping):
        raise NotImplemented()
        #return cv2.Canny(img, low_threshold, high_threshold)

    @staticmethod
    def draw_embedding_image(
            embedding: numpy.ndarray,
            destination: Tuple[int, int],
            size: int,
            image: Image.Image,
    ):
        """Given an embedding and a destination, draw an appropriately sized encoding for the embedding.
        :param numpy.ndarray embedding:
        An embedding of shape (128,) or something that can be losslessly viewed as a 128x1 vector.
        :param (int, int) destination:
        A tuple of x, y indicating the center of the face in pixels.
        :param int size:
        The width (in pixels) that the face embedding should occupy.  Generally, this should be the max(width,height) of
         the bounding box that encloses the face.  Must be at least 12 pixels to fit the embedding!
        :param image: A PIL image. which will be MODIFIED IN PLACE!  Image must be larger than 12x12.
        """
        canvas = ImageDraw.Draw(image)
        x, y = destination
        halfwidth = size // 2
        x_image_offset = x - halfwidth
        y_image_offset = y - halfwidth
        embedding_square_width_in_blocks = int(math.ceil(math.sqrt(EMBEDDING_SIZE)))
        embedding_square_block_size = size // embedding_square_width_in_blocks
        assert size > embedding_square_block_size
        embedding = embedding.reshape(1, -1).squeeze()
        assert embedding.shape[0] == EMBEDDING_SIZE

        # Blank out the area where we put the embedding.
        if image.mode == 'L':
            block_fill = 0
        elif image.mode == 'RGB':
            block_fill = (0, 0, 0)
        else:
            raise ValueError(f"image must be mode 'L' or 'RGB', got {image.mode}")
        canvas.rectangle((x-halfwidth, y-halfwidth, x+halfwidth, y+halfwidth), fill=block_fill)

        # Draw embedding.
        for i in range(0, EMBEDDING_SIZE):
            block_luma = int(127.0 + 128.0*max(-1.0, min(1.0, embedding[i])))
            if image.mode == 'L':
                block_fill = block_luma
            elif image.mode == 'RGB':
                block_fill = (block_luma, block_luma, block_luma)
            else:
                raise ValueError(f"image must be mode 'L' or 'RGB', got {image.mode}")
            block_x = i % embedding_square_width_in_blocks
            block_y = i // embedding_square_width_in_blocks
            canvas.rectangle(
                (
                        x_image_offset + (block_x * embedding_square_block_size),
                        y_image_offset + (block_y * embedding_square_block_size),
                        x_image_offset + ((block_x + 1) * embedding_square_block_size),
                        y_image_offset + ((block_y + 1) * embedding_square_block_size),
                ),
                fill=block_fill
            )

    def create_embedding_image(self, image: Image.Image, empty_image_on_failure: bool = True) -> Image.Image:
        image = Image.new(PIXEL_FORMAT, image.size)
        # Detect all faces:
        faces, probabilities, bboxes, _ = self.detector(image)
        if faces is None:
            if empty_image_on_failure:
                return image
            else:
                return None
        embeddings = self.encoder(faces)
        for idx in range(0, faces.shape[0]):
            width = bboxes[idx, 2] - bboxes[idx, 0]
            height = bboxes[idx, 3] - bboxes[idx, 1]
            center_x = bboxes[idx, 0] + (width//2)
            center_y = bboxes[idx, 1] + (height//2)
            size = max(width, height)
            FaceNet.draw_embedding_image(
                embeddings[idx, :],
                (center_x, center_y),
                size,
                image,
            )
        return image


def get_torch_home():
    torch_home = os.path.expanduser(
        os.getenv("MODEL_PATH",
            os.getenv(
                'TORCH_HOME',
                os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')
            )
        )
    )
    return torch_home