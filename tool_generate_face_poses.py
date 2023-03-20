"""
tool_generate_face_poses.py

This tool accepts a directory and iterates over it, picking out the images with faces that are more than 10% of the
image.  Images are saved with their input filenames, but in a different directory.  JPEG artifacts may results.
"""

import json
import os
import sys
from dataclasses import dataclass, field
from glob import glob
from typing import Mapping

import mediapipe as mp
import numpy
from PIL import Image
from tqdm import tqdm


# from mediapipe.solutions.drawing_styles import DrawingSpec, PoseLandmark  # Why can't we do this?
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_detection  # Only for counting faces.
mp_face_mesh = mp.solutions.face_mesh
mp_face_connections = mp.solutions.face_mesh_connections.FACEMESH_TESSELATION
mp_hand_connections = mp.solutions.hands_connections.HAND_CONNECTIONS
mp_body_connections = mp.solutions.pose_connections.POSE_CONNECTIONS

DrawingSpec = mp.solutions.drawing_styles.DrawingSpec
PoseLandmark = mp.solutions.drawing_styles.PoseLandmark

# The bridges between the parts of the body:
# Higher = more green channel
# Left = more red channel
# Right = mode blue channel
d_thick = 5
c_rad = 1
default_draw = DrawingSpec(color=(250, 250, 250), thickness=d_thick, circle_radius=c_rad)

face_right_draw = DrawingSpec(color=(10, 200, 200), thickness=d_thick, circle_radius=c_rad)
face_left_draw = DrawingSpec(color=(200, 200, 10), thickness=d_thick, circle_radius=c_rad)

shoulder_draw = DrawingSpec(color=(10, 120, 10), thickness=d_thick, circle_radius=c_rad)
left_torso_draw = DrawingSpec(color=(200, 100, 10), thickness=d_thick, circle_radius=c_rad)
right_torso_draw = DrawingSpec(color=(10, 100, 200), thickness=d_thick, circle_radius=c_rad)
pelvis_torso_draw = DrawingSpec(color=(10, 90, 10), thickness=d_thick, circle_radius=c_rad)

left_upper_leg_draw = DrawingSpec(color=(200, 70, 10), thickness=d_thick, circle_radius=c_rad)
left_lower_leg_draw = DrawingSpec(color=(200, 50, 10), thickness=d_thick, circle_radius=c_rad)
right_upper_leg_draw = DrawingSpec(color=(10, 70, 200), thickness=d_thick, circle_radius=c_rad)
right_lower_leg_draw = DrawingSpec(color=(10, 50, 200), thickness=d_thick, circle_radius=c_rad)

left_foot_draw = DrawingSpec(color=(250, 40, 10), thickness=d_thick, circle_radius=c_rad)
right_foot_draw = DrawingSpec(color=(10, 40, 250), thickness=d_thick, circle_radius=c_rad)

left_upper_arm_draw = DrawingSpec(color=(220, 120, 10), thickness=d_thick, circle_radius=c_rad)
left_lower_arm_draw = DrawingSpec(color=(240, 100, 10), thickness=d_thick, circle_radius=c_rad)
right_upper_arm_draw = DrawingSpec(color=(10, 120, 220), thickness=d_thick, circle_radius=c_rad)
right_lower_arm_draw = DrawingSpec(color=(10, 100, 240), thickness=d_thick, circle_radius=c_rad)

left_hand_draw = DrawingSpec(color=(250, 80, 10), thickness=d_thick, circle_radius=c_rad)
right_hand_draw = DrawingSpec(color=(10, 80, 250), thickness=d_thick, circle_radius=c_rad)

pose_connection_spec = {
    # We are omitting the head because we're using the high-res head capture.
    # Head:
    # (PoseLandmark.RIGHT_EYE_OUTER, PoseLandmark.RIGHT_EAR): face_right_draw,
    # (PoseLandmark.LEFT_EYE_OUTER, PoseLandmark.LEFT_EAR): face_left_draw,
    # (PoseLandmark.NOSE, PoseLandmark.LEFT_EYE_INNER): face_left_draw,
    # (PoseLandmark.MOUTH_LEFT, PoseLandmark.MOUTH_RIGHT): face_right_draw,
    # (PoseLandmark.LEFT_EYE_INNER, PoseLandmark.LEFT_EYE): face_left_draw,
    # (PoseLandmark.NOSE, PoseLandmark.RIGHT_EYE_INNER): face_right_draw,
    # (PoseLandmark.LEFT_EYE, PoseLandmark.LEFT_EYE_OUTER): face_left_draw,
    # (PoseLandmark.RIGHT_EYE_INNER, PoseLandmark.RIGHT_EYE): face_right_draw,
    # (PoseLandmark.RIGHT_EYE, PoseLandmark.RIGHT_EYE_OUTER): face_right_draw,

    # Torso:
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER): shoulder_draw,
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP): left_torso_draw,
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP): right_torso_draw,
    (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP): pelvis_torso_draw,

    # Legs:
    (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE): left_upper_leg_draw,
    (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE): right_upper_leg_draw,
    (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE): left_lower_leg_draw,
    (PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE): right_lower_leg_draw,

    # Arms:
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW): right_upper_arm_draw,
    (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST): right_lower_arm_draw,
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW): left_upper_arm_draw,
    (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST): left_lower_arm_draw,

    # Hands:
    (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_THUMB): left_hand_draw,
    (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_PINKY): left_hand_draw,
    (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_INDEX): left_hand_draw,
    (PoseLandmark.LEFT_PINKY, PoseLandmark.LEFT_INDEX): left_hand_draw,

    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_INDEX): right_hand_draw,
    (PoseLandmark.RIGHT_PINKY, PoseLandmark.RIGHT_INDEX): right_hand_draw,
    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_THUMB): right_hand_draw,
    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_PINKY): right_hand_draw,

    # Feet:
    (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_HEEL): right_foot_draw,
    (PoseLandmark.RIGHT_HEEL, PoseLandmark.RIGHT_FOOT_INDEX): right_foot_draw,
    (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_FOOT_INDEX): right_foot_draw,
    (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_FOOT_INDEX): left_foot_draw,
    (PoseLandmark.LEFT_HEEL, PoseLandmark.LEFT_FOOT_INDEX): left_foot_draw,
    (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_HEEL): left_foot_draw,
}

right_iris_draw = DrawingSpec(color=(10, 200, 250), thickness=4, circle_radius=1)
right_eye_draw = DrawingSpec(color=(10, 200, 180), thickness=4, circle_radius=1)
right_eyebrow_draw = DrawingSpec(color=(10, 220, 180), thickness=4, circle_radius=1)
left_iris_draw = DrawingSpec(color=(250, 200, 10), thickness=4, circle_radius=1)
left_eye_draw = DrawingSpec(color=(180, 200, 10), thickness=4, circle_radius=1)
left_eyebrow_draw = DrawingSpec(color=(180, 220, 10), thickness=4, circle_radius=1)
mouth_draw = DrawingSpec(color=(10, 180, 10), thickness=4, circle_radius=1)
head_draw = DrawingSpec(color=(10, 200, 10), thickness=4, circle_radius=1)

# mp_face_mesh.FACEMESH_CONTOURS has all the items we care about.
face_connection_spec = {}
for edge in mp_face_mesh.FACEMESH_FACE_OVAL:
    face_connection_spec[edge] = head_draw
for edge in mp_face_mesh.FACEMESH_LEFT_EYE:
    face_connection_spec[edge] = left_eye_draw
for edge in mp_face_mesh.FACEMESH_LEFT_EYEBROW:
    face_connection_spec[edge] = left_eyebrow_draw
#for edge in mp_face_mesh.FACEMESH_LEFT_IRIS:
#    face_connection_spec[edge] = left_iris_draw
for edge in mp_face_mesh.FACEMESH_RIGHT_EYE:
    face_connection_spec[edge] = right_eye_draw
for edge in mp_face_mesh.FACEMESH_RIGHT_EYEBROW:
    face_connection_spec[edge] = right_eyebrow_draw
#for edge in mp_face_mesh.FACEMESH_RIGHT_IRIS:
#    face_connection_spec[edge] = right_iris_draw
for edge in mp_face_mesh.FACEMESH_LIPS:
    face_connection_spec[edge] = mouth_draw

iris_landmark_spec = {}
#for i in range(478):
#    iris_landmark_spec[i] = DrawingSpec(color=(0, 0, 0), thickness=1, circle_radius=1)
iris_landmark_spec[468] = right_iris_draw
iris_landmark_spec[473] = left_iris_draw


def draw_pupils(image, landmark_list, drawing_spec, halfwidth: int = 2):
    """We have a custom function to draw the pupils because the mp.draw_landmarks method requires a parameter for all
    landmarks.  Until our PR is merged into mediapipe, we need this separate method."""
    if len(image.shape) != 3:
        raise ValueError("Input image must be H,W,C.")
    image_rows, image_cols, image_channels = image.shape
    if image_channels != 3:  # BGR channels
        raise ValueError('Input image must contain three channel bgr data.')
    for idx, landmark in enumerate(landmark_list.landmark):
        if (
                (landmark.HasField('visibility') and landmark.visibility < 0.9) or
                (landmark.HasField('presence') and landmark.presence < 0.5)
        ):
            continue
        if landmark.x >= 1.0 or landmark.x < 0 or landmark.y >= 1.0 or landmark.y < 0:
            continue
        image_x = int(image_cols*landmark.x)
        image_y = int(image_rows*landmark.y)
        draw_color = None
        if isinstance(drawing_spec, Mapping):
            if drawing_spec.get(idx) is None:
                continue
            else:
                draw_color = drawing_spec[idx].color
        elif isinstance(drawing_spec, DrawingSpec):
            draw_color = drawing_spec.color
        image[image_y-halfwidth:image_y+halfwidth, image_x-halfwidth:image_x+halfwidth, :] = draw_color


def reverse_channels(image):
    """Given a numpy array in RGB form, convert to BGR.  Will also convert from BGR to RGB."""
    # im[:,:,::-1] is a neat hack to convert BGR to RGB by reversing the indexing order.
    # im[:,:,::[2,1,0]] would also work but makes a copy of the data.
    return image[:, :, ::-1]


def compute_face_size_in_image(landmarks) -> float:
    face_rect = [
        landmarks[0].x,
        landmarks[0].y,
        landmarks[0].x,
        landmarks[0].y,
    ]  # Left, up, right, down.
    for i in range(len(landmarks)):
        face_rect[0] = min(face_rect[0], landmarks[i].x)
        face_rect[1] = min(face_rect[1], landmarks[i].y)
        face_rect[2] = max(face_rect[2], landmarks[i].x)
        face_rect[3] = max(face_rect[3], landmarks[i].y)
    face_width = abs(face_rect[2] - face_rect[0])
    face_height = abs(face_rect[3] - face_rect[1])
    face_percentage = face_width * face_height  # Already normalized, so we don't have to divide by image size.
    return face_percentage


def count_faces(image, min_confidence: float = 0.5) -> int:
    """Given a BGR image, count the number of faces detected."""
    CLOSE_RANGE_MODEL = 0
    MID_RANGE_MODEL = 0
    faces_detected = 0
    for mod in [CLOSE_RANGE_MODEL, MID_RANGE_MODEL]:
        with mp_face_detection.FaceDetection(
            model_selection=mod,
            min_detection_confidence=min_confidence
        ) as face_detection:
            results = face_detection.process(image)
            if not results.detections:
                continue
            else:
                faces_detected = max(len(results.detections), faces_detected)
    return faces_detected


@dataclass
class RunProgress:
    pending: list = field(default_factory=list)
    success: list = field(default_factory=list)
    skipped_small: list = field(default_factory=list)
    skipped_nsfw: list = field(default_factory=list)
    skipped_noface: list = field(default_factory=list)
    skipped_smallface: list = field(default_factory=list)


def main(
        status_filename: str,
        prompt_filename: str,
        input_glob: str,
        output_directory: str,
        annotated_output_directory: str = "",
        min_image_size: int = 512,
        min_face_size: float = 0.1,
        min_face_detection_confidence: float = 0.5,
        prompt_mapping: dict = None,  # If present, maps a filename to a text prompt.
):
    status = RunProgress()

    # Maybe resume from a previous run:
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
    print(f"{len(status.pending)} images remain")

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
            img = reverse_channels(numpy.asarray(Image.open(full_filename).convert("RGB")))
            if min(img.shape[0], img.shape[1]) < min_image_size:
                status.skipped_small.append(full_filename)
                continue

            # We re-initialize the detector every time because it has a habit of triggering weird race conditions.
            with mp.solutions.holistic.Holistic(
                    static_image_mode=True,
                    model_complexity=2,
                    enable_segmentation=True,
                    refine_face_landmarks=True,
                    min_detection_confidence=min_face_detection_confidence,
            ) as holistic:
                results = holistic.process(img.copy())

                # How much of the image does the face use?
                if results.face_landmarks is None:
                    # Skip images with no faces.
                    status.skipped_noface.append(full_filename)
                    continue
                else:
                    # Find a bounding box for the face.
                    face_percentage = compute_face_size_in_image(results.face_landmarks.landmark)
                    if face_percentage < min_face_size:
                        status.skipped_smallface.append(full_filename)
                        continue

                annotated = numpy.asarray(img.copy())
                empty = numpy.zeros_like(annotated)

                for out in [annotated, empty]:
                    # Draw the body:
                    #mp_drawing.draw_landmarks(
                    #    out,
                    #    results.pose_landmarks,
                    #    connections=pose_connection_spec.keys(),
                    #    landmark_drawing_spec=None,
                    #    connection_drawing_spec=pose_connection_spec
                    #)
                    # Draw the face:
                    mp_drawing.draw_landmarks(
                        out,
                        results.face_landmarks,
                        connections=face_connection_spec.keys(),
                        landmark_drawing_spec=None,
                        connection_drawing_spec=face_connection_spec
                    )
                    draw_pupils(out, results.face_landmarks, iris_landmark_spec, 2)
                    # Draw hands
                    #mp_drawing.draw_landmarks(
                    #    out,
                    #    results.left_hand_landmarks,
                    #    connections=mp_hand_connections,
                    #    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
                    #)
                    #mp_drawing.draw_landmarks(
                    #    out,
                    #    results.right_hand_landmarks,
                    #    connections=mp_hand_connections,
                    #    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
                    #)

            # Both annotated and empty are in BGR, not RGB.
            annotated = reverse_channels(annotated)
            empty = reverse_channels(empty)
            if annotation_filename:
                Image.fromarray(annotated).save(annotation_filename)
            Image.fromarray(empty).save(output_filename)

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
    print(f"{len(status.skipped_small)} images rejected for size.")
    print(f"{len(status.skipped_smallface)} images rejected for having faces too small.")
    print(f"{len(status.skipped_noface)} images rejected for not having faces.")


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
