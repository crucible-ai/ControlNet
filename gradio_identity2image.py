import os
from typing import Mapping

import gradio as gr
import numpy
import torch
import random
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from annotator.facenet import FaceNet
from share import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
facenet_model = FaceNet("./annotator/facenet/models/data", device=device)
controlnet_model = create_model('./models/cldm_v21.yaml').cpu()
controlnet_model.load_state_dict(load_state_dict('./models/controlnet_sd15_laion_face_epoch=2_step=84999_loss=0.ckpt', location=device.type))
controlnet_model = controlnet_model.to(device)
ddim_sampler = DDIMSampler(controlnet_model)  # ControlNet _only_ works with DDIM.


def process(
        identity_image: Image.Image,
        target_position: Image.Image,
        prompt: str,
        a_prompt: str,
        n_prompt: str,
        num_samples: int,
        ddim_steps: int,
        guess_mode: bool,
        strength: float,
        scale,
        seed,
        eta
):
    with torch.no_grad():
        # Extract the embedding from the face.
        face_embedding = facenet_model.create_embedding_image(identity_image, empty_image_on_failure=False)
        assert face_embedding is not None
        visualization = face_embedding.copy()  # Save to help debug.

        #empty = numpy.moveaxis(empty, 2, 0)  # h, w, c -> c, h, w
        control = pil_to_tensor(face_embedding).to(device).to(torch.float) / 255.0
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
            controlnet_model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [controlnet_model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [controlnet_model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            controlnet_model.low_vram_shift(is_diffusing=True)

        controlnet_model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
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
            controlnet_model.low_vram_shift(is_diffusing=False)

        x_samples = controlnet_model.decode_first_stage(samples)
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
            identity_image = gr.Image(source='upload', type="pil")
            target_position = gr.Image(source='canvas', type="pil", tool="sketch")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                num_samples = gr.Slider(label="Num Samples", minimum=1, maximum=16, value=1, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='')
                n_prompt = gr.Textbox(label="Negative Prompt", value='')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
    ips = [identity_image, target_position, prompt, a_prompt, n_prompt, num_samples, ddim_steps, guess_mode, strength, scale, seed, eta]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(server_name='0.0.0.0')
