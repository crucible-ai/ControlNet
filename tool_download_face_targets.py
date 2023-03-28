#!/usr/bin/python3
"""
tool_download_face_targets.py

Reads in the metadata from the LAION images and begins downloading all images.
"""

import json
import os
import sys
import time
import urllib
import urllib.request
try:
    from tqdm import tqdm
except ImportError:
    # Wrap this method into the identity.
    print("TQDM not found.  Progress will be quiet without 'verbose'.")
    def tqdm(x):
        return x


def main(logfile_path: str, verbose: bool = False, pause_between_fetches: float = 0.0):
    """Open the metadata.json file from the training directory and fetch all target images."""
    # Toggle a function pointer so we don't have to check verbosity everywhere.
    def out(x):
        pass
    if verbose:
        out = print

    log = open(logfile_path, 'at')
    skipped_image_count = 0
    errored_image_count = 0
    successful_image_count = 0
    if not os.path.exists("training"):
        print("ERROR: training directory does not exist in the current directory.")
        print("Has the archive been unzipped?")
        print("Are you running from the project root?")
        return 2  # BASH: No such directory.
    if not os.path.exists("training/laion-face-processed/metadata.json"):
        print("ERROR: metadata.json was not found in training/laion-face-processed.")
        return 2
    with open("training/laion-face-processed/metadata.json", 'rt') as md_in:
        metadata = json.load(md_in)
    # Create the directory for targets if it does not exist.
    if not os.path.exists("training/laion-face-processed/target"):
        os.mkdir("training/laion-face-processed/target")
    for image_id, image_data in tqdm(metadata.items()):
        filename = f"training/laion-face-processed/target/{image_id}.jpg"
        if os.path.exists(filename):
            out(f"Skipping {image_id}: file exists.")
            skipped_image_count += 1
            continue
        if not download_file(image_data['url'], filename, verbose):
            error_message = f"Problem downloading {image_id}"
            out(error_message)
            log.write(error_message + "\n")
            log.flush()  # Flush often in case we crash.
            errored_image_count += 1
        if pause_between_fetches > 0.0:
            time.sleep(pause_between_fetches)
        successful_image_count += 1
    log.close()
    print("Run success.")
    print(f"{skipped_image_count} images skipped")
    print(f"{errored_image_count} images failed to download")
    print(f"{successful_image_count} images downloaded")


def download_file(url: str, output_path: str, verbose: bool = False) -> bool:
    """Download the file with the given URL and save it to the specified path.  Return true on success."""
    try:
        r = urllib.request.urlopen(url)
        if not r.status == 200:
            return False
        with open(output_path, 'wb') as fout:
            fout.write(r.read())
        return True
    except Exception as e:
        if verbose:
            print(e)
        return False


if __name__ == "__main__":
    main("downloads.log", verbose="-v" in sys.argv)
