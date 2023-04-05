import os

import torch
from safetensors.torch import load_file, save_file

from cldm.model import create_model, load_state_dict


def extract(base_model_yaml="./models/cldm_v15.yaml", model_checkpoint="./models/controlnet_sd15_laion_face.ckpt"):
    model = create_model(base_model_yaml).cpu()
    model.load_state_dict(load_state_dict(model_checkpoint))
    state_dict = model.state_dict()
    state_dict = {k.replace("control_model.", ""): v for k, v in state_dict.items() if k.startswith("control_model.")}
    root_name = os.path.splitext(model_checkpoint)[0]
    save_file(state_dict, root_name + ".safetensors")
    torch.save({"state_dict": state_dict}, root_name + ".pt")


if __name__=="__main__":
    extract(sys.argv[1], sys.argv[2])
