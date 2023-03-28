import json
import os
import sys
from dataclasses import dataclass, field
from glob import glob
from typing import Mapping

from PIL import Image
from tqdm import tqdm

from laion_face_common import generate_annotation


@dataclass
class RunProgress:
    pending: list = field(default_factory=list)
    success: list = field(default_factory=list)
    skipped_size: list = field(default_factory=list)
    skipped_nsfw: list = field(default_factory=list)
    skipped_noface: list = field(default_factory=list)
    skipped_smallface: list = field(default_factory=list)


def main(
        status_filename: str,
        prompt_filename: str,
        input_glob: str,
        output_directory: str,
        annotated_output_directory: str = "",
        min_image_size: int = 384,
        max_image_size: int = 32766,
        min_face_size_pixels: int = 64,
        prompt_mapping: dict = None,  # If present, maps a filename to a text prompt.
):
    status = RunProgress()

    if os.path.exists(status_filename):
        print("Continuing from checkpoint.")
        # Restore a saved state:
        status_temp = json.load(open(status_filename, 'rt'))
        for k in status.__dict__.keys():
            status.__setattr__(k, status_temp[k])
        # Output label file:
        pout = open(prompt_filename, 'at')
    else:
        print("Starting run.")
        status = RunProgress()
        status.pending = list(glob(input_glob))
        # Output label file:
        pout = open(prompt_filename, 'wt')
        with open(status_filename, 'wt') as fout:
            json.dump(status.__dict__, fout)

    print(f"{len(status.pending)} images remaining")

    # If we don't have a preexisting set of labels (like for ImageNet/MSCOCO), just null-fill the mapping.
    # We will try on a per-image basis to see if there's a metadata .json.
    if prompt_mapping is None:
        prompt_mapping = dict()

    step = 0
    with tqdm(total=len(status.pending)) as pbar:
        while len(status.pending) > 0:
            full_filename = status.pending.pop()
            pbar.update(1)
            step += 1

            if step % 100 == 0:
                # Checkpoint save:
                with open(status_filename, 'wt') as fout:
                    json.dump(status.__dict__, fout)

            _fpath, fname = os.path.split(full_filename)

            # Make our output filenames.
            # We used to do this here so we could check if a file existed before writing, then skip it, but since we
            # have a 'status' that we cache and update, we no longer have to do this check.
            annotation_filename = ""
            if annotated_output_directory:
                annotation_filename = os.path.join(annotated_output_directory, fname)
            output_filename = os.path.join(output_directory, fname)

            # The LAION dataset has accompanying .json files with each image.
            partial_filename, extension = os.path.splitext(full_filename)
            candidate_json_fullpath = partial_filename + ".json"
            image_metadata = {}
            if os.path.exists(candidate_json_fullpath):
                try:
                    image_metadata = json.load(open(candidate_json_fullpath, 'rt'))
                except Exception as e:
                    print(e)
            if "NSFW" in image_metadata:
                nsfw_marker = image_metadata.get("NSFW")  # This can be "", None, or other weird things.
                if nsfw_marker is not None and nsfw_marker.lower() != "unlikely":
                    # Skip NSFW images.
                    status.skipped_nsfw.append(full_filename)
                    continue

            # Try to get a prompt/caption from the metadata or the prompt mapping.
            image_prompt = image_metadata.get("caption", prompt_mapping.get(fname, ""))

            # Load image:
            img = Image.open(full_filename).convert("RGB")
            img_width = img.size[0]
            img_height = img.size[1]
            img_size = min(img.size[0], img.size[1])
            if img_size < min_image_size or max(img_width, img_height) > max_image_size:
                status.skipped_size.append(full_filename)
                continue

            # We re-initialize the detector every time because it has a habit of triggering weird race conditions.
            empty, annotated, faces_before_filtering, faces_after_filtering = generate_annotation(
                img,
                max_faces=5,
                min_face_size_pixels=min_face_size_pixels,
                return_annotation_data=True
            )
            if faces_before_filtering == 0:
                # Skip images with no faces.
                status.skipped_noface.append(full_filename)
                continue
            if faces_after_filtering == 0:
                # Skip images with no faces large enough
                status.skipped_smallface.append(full_filename)
                continue

            Image.fromarray(empty).save(output_filename)
            if annotation_filename:
                Image.fromarray(annotated).save(annotation_filename)

            # See https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md for the training file format.
            # prompt.json
            # a JSONL file with {"source": "source/0.jpg", "target": "target/0.jpg", "prompt": "..."}.
            # a source/xxxxx.jpg or source/xxxx.png file for each of the inputs.
            # a target/xxxxx.jpg for each of the outputs.
            pout.write(json.dumps({
                "source": os.path.join(output_directory, fname),
                "target": full_filename,
                "prompt": image_prompt,
            }) + "\n")
            pout.flush()
            status.success.append(full_filename)

    # We do save every 100 iterations, but it's good to save on completion, too.
    with open(status_filename, 'wt') as fout:
        json.dump(status.__dict__, fout)

    pout.close()
    print("Done!")
    print(f"{len(status.success)} images added to dataset.")
    print(f"{len(status.skipped_size)} images rejected for size.")
    print(f"{len(status.skipped_smallface)} images rejected for having faces too small.")
    print(f"{len(status.skipped_noface)} images rejected for not having faces.")
    print(f"{len(status.skipped_nsfw)} images rejected for NSFW.")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and "-h" not in sys.argv:
        prompt_jsonl = sys.argv[1]
        in_glob = sys.argv[2]  # Should probably be in a directory called "target/*.jpg".
        output_dir = sys.argv[3]  # Should probably be a directory called "source".
        annotation_dir = ""
        if len(sys.argv) > 4:
            annotation_dir = sys.argv[4]
        main("generate_face_poses_checkpoint.json", prompt_jsonl, in_glob, output_dir, annotation_dir)
    else:
        print(f"""Usage:
        python {sys.argv[0]} prompt.jsonl target/*.jpg source/ [annotated/]
        source and target are slightly confusing in this context.  We are writing the image names to prompt.jsonl, so 
        the naming system has to be consistent with what ControlNet expects.  In ControlNet, the source is the input and
        target is the output.  We are generating source images from targets in this application, so the second argument 
        should be a folder full of images.  The third argument should be 'source', where the images should be places.
        Optionally, an 'annotated' directory can be provided.  Augmented images will be placed here.

        A checkpoint file named 'generate_face_poses_checkpoint.json' will be created in the place where the script is 
        run.  If a run is cancelled, it can be resumed from this checkpoint.

        If invoking the script from bash, do not forget to enclose globs with quotes.  Example usage:
        `python ./tool_generate_face_poses.py ./face_prompt.jsonl "/home/josephcatrambone/training_data/data-mscoco/images/train2017/*" /home/josephcatrambone/training_data/data-mscoco/images/source_2017/`
        """)
