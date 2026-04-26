import os
import random

import folder_paths
from PIL import Image
import numpy as np
import torch
import hashlib
from comfy.cli_args import args
from pathlib import Path

# set the models directory

if args.base_directory:
    base_path = os.path.join(Path(os.path.abspath(args.base_directory)).parent.parent, "models")
else:
    base_path = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))).parent.parent, "models")

os.makedirs(os.path.join(base_path, "conditionings"), exist_ok=True)

folder_paths.folder_names_and_paths["conditionings"] = ([os.path.join(base_path, "conditionings")], [".bin"])

class SaveConditioning:
    def __init__(self):
        self.output_dir = os.path.join(base_path, "conditionings")

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"conditionings": ("CONDITIONING", ),},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_conditioning"
    OUTPUT_NODE = True
    CATEGORY = "endman100"

    def save_conditioning(self, conditionings): # conditionings : [[text, {"pooled_output"}]...]
        results = list()
        for (batch_number, conditioning) in enumerate(conditionings):
            save_path = os.path.join(self.output_dir, f"{batch_number:05}_conditionings.bin")
            print(conditioning)
            print(f"conditioning[0].shape:{conditioning[0].shape}, save_path:{save_path}")
            for key, value in conditioning[1].items():
                if(type(value) == torch.Tensor):
                    print(f"key:{key}, type:{type(value)}, shape:{value.shape}")
                else:
                    print(f"key:{key}, type:{type(conditioning[1][key])}")

            if(hasattr(conditioning[0], "addit_embeds")):
               for key, value in conditioning[0].addit_embeds.items():
                    if(type(value) == torch.Tensor):
                        print(f"key:{key}, type:{type(value)}, shape:{value.shape}")
                    else:
                        print(f"key:{key}, type:{type(value)}")

            torch.save(conditioning, save_path)

        return { "ui": { "conditionings": results } }

class LoadContditioning():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "conditioning": (folder_paths.get_filename_list("conditionings"), )}}

    CATEGORY = "endman100"
    RETURN_TYPES = ("CONDITIONING", )
    FUNCTION = "load_conditioning"

    def load_conditioning(self, conditioning):
        conditioning_path = folder_paths.get_full_path("conditionings", conditioning)
        conditioning_list = torch.load(conditioning_path)
        conditioning_list[0] = conditioning_list[0].cpu()
        for key, value in conditioning_list[1].items():
            if(type(value) == torch.Tensor):
                conditioning_list[1][key] = value.cpu()
        if(hasattr(conditioning_list[0], "addit_embeds")):
            for key, value in conditioning_list[0].addit_embeds.items():
                if(type(value) == torch.Tensor):
                    conditioning_list[0].addit_embeds[key] = value.cpu()
        return ([conditioning_list], )

    @classmethod
    def IS_CHANGED(s, conditioning):
        conditioning_path = folder_paths.get_full_path("conditionings", conditioning)
        m = hashlib.sha256()
        with open(conditioning_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, conditioning):
        conditioning_path = folder_paths.get_full_path("conditionings", conditioning)
        if not os.path.exists(conditioning_path):
            return "Invalid conditioning file: {}".format(conditioning_path)
        return True
    

NODE_CLASS_MAPPINGS = {
    "SaveConditioning": SaveConditioning,
    "LoadContditioning": LoadContditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Save Conditioning": "SaveConditioning",
    "Load Contditioning": "LoadContditioning"
}
