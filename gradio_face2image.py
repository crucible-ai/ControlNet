import os
from typing import Mapping

import gradio as gr
import numpy
import torch
import random
from PIL import Image
from tqdm import tqdm

import config
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from share import *

# All of this is dedicated to producing an image that is compatible with the control network.

import mediapipe as mp
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

f_thick = 2
f_rad = 1
right_iris_draw = DrawingSpec(color=(10, 200, 250), thickness=f_thick, circle_radius=f_rad)
right_eye_draw = DrawingSpec(color=(10, 200, 180), thickness=f_thick, circle_radius=f_rad)
right_eyebrow_draw = DrawingSpec(color=(10, 220, 180), thickness=f_thick, circle_radius=f_rad)
left_iris_draw = DrawingSpec(color=(250, 200, 10), thickness=f_thick, circle_radius=f_rad)
left_eye_draw = DrawingSpec(color=(180, 200, 10), thickness=f_thick, circle_radius=f_rad)
left_eyebrow_draw = DrawingSpec(color=(180, 220, 10), thickness=f_thick, circle_radius=f_rad)
mouth_draw = DrawingSpec(color=(10, 180, 10), thickness=f_thick, circle_radius=f_rad)
head_draw = DrawingSpec(color=(10, 200, 10), thickness=f_thick, circle_radius=f_rad)

# mp_face_mesh.FACEMESH_CONTOURS has all the items we care about.
face_connection_spec = {}
for edge in mp_face_mesh.FACEMESH_FACE_OVAL:
    face_connection_spec[edge] = head_draw
for edge in mp_face_mesh.FACEMESH_LEFT_EYE:
    face_connection_spec[edge] = left_eye_draw
for edge in mp_face_mesh.FACEMESH_LEFT_EYEBROW:
    face_connection_spec[edge] = left_eyebrow_draw
# for edge in mp_face_mesh.FACEMESH_LEFT_IRIS:
#    face_connection_spec[edge] = left_iris_draw
for edge in mp_face_mesh.FACEMESH_RIGHT_EYE:
    face_connection_spec[edge] = right_eye_draw
for edge in mp_face_mesh.FACEMESH_RIGHT_EYEBROW:
    face_connection_spec[edge] = right_eyebrow_draw
# for edge in mp_face_mesh.FACEMESH_RIGHT_IRIS:
#    face_connection_spec[edge] = right_iris_draw
for edge in mp_face_mesh.FACEMESH_LIPS:
    face_connection_spec[edge] = mouth_draw
iris_landmark_spec = {468: right_iris_draw, 473: left_iris_draw}

model = create_model('./models/cldm_v21.yaml').cpu()
#model.load_state_dict(load_state_dict('./models/controlnet_face_condition_epoch_0.ckpt', location='cuda'))
model.load_state_dict(load_state_dict('./models/controlnet_face_condition_epoch_4_0percent.ckpt', location='cuda'))
#model.load_state_dict(load_state_dict('./models/controlnet_sd21_face.ckpt', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)  # ControlNet _only_ works with DDIM.


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


def process(input_image: Image.Image, prompt, a_prompt, n_prompt, max_faces, num_samples, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=max_faces,
                refine_landmarks=True,
                min_detection_confidence=0.01,
        ) as facemesh:
            img_rgb = numpy.asarray(input_image)
            results = facemesh.process(img_rgb).multi_face_landmarks

            # Filter faces that are too small
            filtered_landmarks = []
            for lm in results:
                landmarks = lm.landmark
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
                filtered_landmarks.append(lm)

            # Annotations are drawn in BGR for some stupid reason.
            empty = numpy.zeros_like(img_rgb)

            # Draw detected faces:
            for face_landmarks in filtered_landmarks:
                mp_drawing.draw_landmarks(
                    empty,
                    face_landmarks,
                    connections=face_connection_spec.keys(),
                    landmark_drawing_spec=None,
                    connection_drawing_spec=face_connection_spec
                )
                draw_pupils(empty, face_landmarks, iris_landmark_spec, 2)

        # Both annotated and empty are in BGR, not RGB.
        empty = reverse_channels(empty)
        visualization = Image.fromarray(empty)  # Save to help debug.

        empty = numpy.moveaxis(empty, 2, 0)  # h, w, c -> c, h, w
        control = torch.from_numpy(empty.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        # control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        # Sanity check the dimensions.
        B, C, H, W = control.shape
        assert C == 3
        assert B == num_samples

        if seed != -1:
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            numpy.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        # x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(numpy.uint8)
        x_samples = numpy.moveaxis((x_samples * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(numpy.uint8), 1, -1)  # b, c, h, w -> b, h, w, c
        results = [visualization] + [x_samples[i] for i in range(num_samples)]

    return results


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with a Facial Pose")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                max_faces = gr.Slider(label="Max Faces", minimum=1, maximum=5, value=1, step=1)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
    ips = [input_image, prompt, a_prompt, n_prompt, max_faces, num_samples, ddim_steps, guess_mode, strength, scale, seed, eta]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(server_name='0.0.0.0')
