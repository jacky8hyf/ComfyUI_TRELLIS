# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import gc
import os
import pathlib
import torch
import uuid
from .app import image_to_3d
from .trellis.pipelines import TrellisImageTo3DPipeline
from .utils import glb2obj_,obj2fbx_,tensor2imglist,pre_img
import folder_paths

MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
current_path = os.path.dirname(os.path.abspath(__file__))

weigths_dinov2_current_path = os.path.join(folder_paths.models_dir, "dinov2")
if not os.path.exists(weigths_dinov2_current_path):
    os.makedirs(weigths_dinov2_current_path)

try:
    folder_paths.add_model_folder_path("dinov2", weigths_dinov2_current_path, False)
except:
    folder_paths.add_model_folder_path("dinov2", weigths_dinov2_current_path)

class Trellis_LoadModel:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo": ("STRING", {"default": "JeffreyXiang/TRELLIS-image-large"}),
                "dino": (["none"] + folder_paths.get_filename_list("dinov2"),),
                "attn_backend":(["xformers","flash-attn"],),
                "spconv_algo":(["auto","flash-native"],),
            }
        }

    RETURN_TYPES = ("MODEL_TRELLIS", )
    RETURN_NAMES = ("model",)
    FUNCTION = "main_loader"
    CATEGORY = "Trellis"

    def main_loader(self, repo,dino,attn_backend,spconv_algo):
        if attn_backend=="xformers":
            os.environ['ATTN_BACKEND'] = 'xformers'
        else:
            os.environ['ATTN_BACKEND'] = 'flash-attn'
        if spconv_algo=="auto":
            os.environ['SPCONV_ALGO'] = 'auto'
        else:
            os.environ['SPCONV_ALGO'] = 'native'

        if dino=="none":
            raise "need choice dinov2 checkpoint"

        TrellisImageTo3DPipeline.dino=folder_paths.get_full_path("dinov2", dino)
        TrellisImageTo3DPipeline.dino_moudel=os.path.join(current_path,"facebookresearch/dinov2")
        if repo:
            model=TrellisImageTo3DPipeline.from_pretrained(repo)
        else:
            raise "need fill repo"
        return (model,)


class Trellis_Sampler:
    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "image": ("IMAGE",),  # [B,H,W,C], C=3
                "model": ("MODEL_TRELLIS",),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "cfg": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "steps": ("INT", {"default": 12, "min": 1, "max": 50}),
                "slat_cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "slat_steps": ("INT", {"default": 12, "min": 1, "max": 50}),
                "preprocess_image": ("BOOLEAN", {"default": False},),
                "texture_size": ("INT", {"default": 512, "min": 512, "max": 2048, "step": 512, "display": "number"}),
                "mesh_simplify": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 0.98, "step": 0.01}),
                "mode":(["fast","opt"],),
                "multi_image": ("BOOLEAN", {"default": False},),
                "multiimage_algo":(["multidiffusion", "stochastic"],),
                "gaussians2ply": ("BOOLEAN", {"default": False},),
                "covert2video": ("BOOLEAN", {"default": False},),
                "glb2obj": ("BOOLEAN", {"default": False},),
                "glb2fbx": ("BOOLEAN", {"default": False},),
                "custom_path": ("STRING", {"default": ""}),
                "filename_prefix": ("STRING", {"default": ""}),
                "timestamp_fixed": ("STRING", {"default": ""}),
            }
        }

    OUTPUT_IS_LIST = (True, True, True,)
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("glb_paths", "obj_paths", "fbx_paths")
    FUNCTION = "sampler_main"
    CATEGORY = "Trellis"

    def sampler_main(self, image, model,  seed, cfg, steps,slat_cfg, slat_steps,preprocess_image,texture_size,mesh_simplify,mode,multi_image,multiimage_algo,gaussians2ply,covert2video,glb2obj,glb2fbx,custom_path,filename_prefix,timestamp_fixed):

        image_list,image_batch=tensor2imglist(image) #pil_list,batch

        if multi_image and image_batch % 3 == 0:
            print("********infer multi image,like Three views ******")
            image_list=[image_list[i:i + 3] for i in range(0, len(image_list), 3)] #三等分列表
            is_multiimage=True
        else:
            is_multiimage = False

        if filename_prefix or timestamp_fixed:
            if not filename_prefix:
                stem = timestamp_fixed
            elif not timestamp_fixed:
                stem = filename_prefix
            else:
                stem = f"{filename_prefix}_{timestamp_fixed}"
        else:
            stem = str(uuid.uuid4())

        if custom_path:
            path_rel = pathlib.Path(custom_path) / stem
        else:
            path_rel = pathlib.Path(stem)

        path_base = pathlib.Path(folder_paths.get_output_directory()) / path_rel
        path_base.parent.mkdir(parents=True, exist_ok=True)

        glb_paths = []
        for i,img in enumerate(image_list):
            model.cuda()
            glb=image_to_3d(model,img,preprocess_image,covert2video,path_rel,seed,cfg,steps,slat_cfg,slat_steps,mesh_simplify,texture_size,mode,is_multiimage,gaussians2ply,multiimage_algo)
            glb_path = f"{path_base}_{i}.glb"
            glb.export(glb_path)
            glb_paths.append(glb_path)
            model.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            print(f"glb save in {glb_path} ")

        obj_paths = []
        if glb2obj or glb2fbx:
            for path in glb_paths:
                obj_path = path.with_suffix(".obj")
                glb2obj_(path, obj_path)
                obj_paths.append(obj_path)

        fbx_paths = []
        if glb2fbx:
            for path in obj_paths:
                fbx_path = path.with_suffix(".fbx")
                obj2fbx_(path, fbx_path)
                fbx_paths.append(fbx_path)

        return ([str(path) for path in glb_paths], [str(path) for path in obj_paths], [str(path) for path in fbx_paths])

class Trellis_multiimage_loader:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image_a": ("IMAGE",),
                             },
                "optional": {"image_b": ("IMAGE",),
                             "image_c": ("IMAGE",),}
                }

    RETURN_TYPES = ("IMAGE",)
    ETURN_NAMES = ("image",)
    FUNCTION = "main_batch"
    CATEGORY = "Trellis"

    def main_batch(self, image_a, **kwargs):
        image_b = kwargs.get("image_b")
        image_c = kwargs.get("image_c")
        _,height_a,_,_ = image_a.shape
        if isinstance(image_b, torch.Tensor) and isinstance(image_c, torch.Tensor):
            _, height_b, _, _ = image_b.shape
            _, height_c, _, _ = image_c.shape
            height = max(height_a, height_b, height_c)
            img_list=[pre_img(image_a, height),pre_img(image_b, height),pre_img(image_c, height)]
            image = torch.cat(img_list, dim=0)
        elif isinstance(image_b, torch.Tensor) and not isinstance(image_c, torch.Tensor):
            _, height_b, _, _ = image_b.shape
            height = max(height_a, height_b,)
            img_list = [pre_img(image_a, height), pre_img(image_b, height)]
            image = torch.cat(img_list, dim=0)
        elif not isinstance(image_b, torch.Tensor) and isinstance(image_c, torch.Tensor):
            _, height_c, _, _ = image_c.shape
            height = max(height_a, height_c, )
            img_list = [pre_img(image_a, height), pre_img(image_b, height)]
            image = torch.cat(img_list, dim=0)
        else:
            image=image_a

        return (image,)


NODE_CLASS_MAPPINGS = {
    "Trellis_LoadModel": Trellis_LoadModel,
    "Trellis_Sampler": Trellis_Sampler,
    "Trellis_multiimage_loader":Trellis_multiimage_loader

}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Trellis_LoadModel": "Trellis_LoadModel",
    "Trellis_Sampler": "Trellis_Sampler",
    "Trellis_multiimage_loader":"Trellis_multiimage_loader"
}
