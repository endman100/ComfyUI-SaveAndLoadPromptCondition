import os
import hashlib
import torch
import folder_paths

# 1) Basis-Verzeichnis & Suffix einrichten
if "conditionings" not in folder_paths.folder_names_and_paths:
    current_paths = [os.path.join(folder_paths.models_dir, "conditionings")]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["conditionings"]
folder_paths.folder_names_and_paths["conditionings"] = (current_paths, ".bin")

COND_BASE_DIR = folder_paths.folder_names_and_paths["conditionings"][0][0]
COND_SUFFIX   = folder_paths.folder_names_and_paths["conditionings"][1]

# 2) SaveConditioning Node
class SaveConditioning:
    def __init__(self):
        self.base_dir = COND_BASE_DIR

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pos":      ("CONDITIONING",),
                "neg":      ("CONDITIONING",),
                "filename": ("STRING", {"default": "my_conditioning"})
            }
        }

    RETURN_TYPES = ()
    FUNCTION     = "save_conditioning"
    OUTPUT_NODE  = True
    CATEGORY     = "endman100"

    def save_conditioning(self, pos, neg, filename):
        # jeweils erstes Element holen
        pos_cond = pos[0]
        neg_cond = neg[0]
        # Pfad + Unterordner erlauben
        save_path = os.path.join(self.base_dir, f"{filename}{COND_SUFFIX}")
        save_dir  = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # Optional: Debug-Ausgaben
        print(">> Saving POS and NEG conditioning:")
        print(f" - pos.shape: {pos_cond[0].shape}, neg.shape: {neg_cond[0].shape}")
        print(f" - save_path: {save_path}")

        # In Datei packen
        packed = {
            "pos": pos_cond,
            "neg": neg_cond
        }
        torch.save(packed, save_path)

        return {"ui": {"conditionings": []}}

# 3) LoadConditioning Node
class LoadConditioning:
    @classmethod
    def INPUT_TYPES(s):
        # alle .bin-Dateien rekursiv sammeln
        files = []
        for root, dirs, fns in os.walk(COND_BASE_DIR):
            for fn in fns:
                if fn.endswith(COND_SUFFIX):
                    full = os.path.join(root, fn)
                    rel  = os.path.relpath(full, COND_BASE_DIR)
                    files.append(rel)
        return {"required": {"conditioning": (tuple(files),)}}

    CATEGORY     = "endman100"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    FUNCTION     = "load_conditioning"

    def load_conditioning(self, conditioning):
        path = os.path.join(COND_BASE_DIR, conditioning)
        data = torch.load(path)

        # Tensoren auf CPU
        pos, neg = data["pos"], data["neg"]
        pos[0] = pos[0].cpu()
        neg[0] = neg[0].cpu()
        for key, v in pos[1].items():
            if isinstance(v, torch.Tensor):
                pos[1][key] = v.cpu()
        for key, v in neg[1].items():
            if isinstance(v, torch.Tensor):
                neg[1][key] = v.cpu()
        # ggf. addit_embeds behandeln
        for cond in (pos, neg):
            if hasattr(cond[0], "addit_embeds"):
                for k, v in cond[0].addit_embeds.items():
                    if isinstance(v, torch.Tensor):
                        cond[0].addit_embeds[k] = v.cpu()

        # zwei Outputs: pos und neg
        return ([pos], [neg])

    @classmethod
    def IS_CHANGED(s, conditioning):
        path = os.path.join(COND_BASE_DIR, conditioning)
        m = hashlib.sha256()
        with open(path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, conditioning):
        path = os.path.join(COND_BASE_DIR, conditioning)
        if not os.path.exists(path):
            return f"Invalid conditioning file: {path}"
        return True

# 4) Registrierung
NODE_CLASS_MAPPINGS = {
    "SaveConditioning": SaveConditioning,
    "LoadConditioning": LoadConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Save Conditioning": "SaveConditioning",
    "Load Conditioning": "LoadConditioning"
}
