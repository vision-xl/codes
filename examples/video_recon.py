import argparse
from pathlib import Path

from munch import munchify
from torchvision.utils import save_image

from latent_diffusion import get_solver
from latent_sdxl import get_solver as get_solver_sdxl
from utils.callback_util import ComposeCallback
from utils.log_util import create_workdir, set_seed

from PIL import Image
import torchvision.transforms as transforms
import torch
import os
from einops import rearrange
import numpy as np
import glob

SD_XL_BASE_RATIOS = {
    "0.5": (704, 1408),
    "0.52": (704, 1344),
    "0.57": (768, 1344),
    "0.6": (768, 1280),
    "0.68": (832, 1216),
    "0.72": (832, 1152),
    "0.78": (896, 1152),
    "0.82": (896, 1088),
    "0.88": (960, 1088),
    "0.94": (960, 1024),
    "1.0": (1024, 1024),
    "1.07": (1024, 960),
    "1.13": (1088, 960),
    "1.21": (1088, 896),
    "1.29": (1152, 896),
    "1.38": (1152, 832),
    "1.46": (1216, 832),
    "1.67": (1280, 768),
    "1.75": (1344, 768),
    "1.91": (1344, 704),
    "2.0": (1408, 704),
    "2.09": (1472, 704),
    "2.4": (1536, 640),
    "2.5": (1600, 640),
    "2.89": (1664, 576),
    "3.0": (1728, 576),
}

def load_image_batch(folder_path, height, width):
    image_tensors = []
    # Define a transform to resize the image and normalize it
    transform = transforms.Compose([
        transforms.Resize((height, width)),   # Resize to HxW
        transforms.ToTensor(),                # Convert to tensor
        transforms.Lambda(lambda x: x*2 - 1)  # Scale pixel values to [-1, 1]
    ])
    print(folder_path)

    # Iterate through all files in the folder
    for filename in sorted(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, filename)
        if os.path.isfile(image_path):  # Check if it is a valid file
            try:
                # Open and transform the image
                image = Image.open(image_path).convert('RGB')
                image_tensor = transform(image)
                image_tensors.append(image_tensor.unsqueeze(0))  # Add batch dimension (1, C, H, W)
            except Exception as e:
                print(f"Error loading image {filename}: {e}")

    # Stack all image tensors into a single batch (N, C, H, W)
    if len(image_tensors) > 0:
        image_batch = torch.cat(image_tensors, dim=0).to("cuda").to(torch.float16)

    return image_batch

def save_video_to_gif(video, save_path, duration = 125):
    vid = (
        (rearrange(video, "t c h w -> t h w c") * 255)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    ## To gif
    images = [Image.fromarray(vid[i]) for i in range(vid.shape[0])]                                    
    images[0].save(save_path, save_all=True, append_images=images[1:], duration=duration, loop=0)

def main():
    parser = argparse.ArgumentParser(description="Latent Diffusion")
    parser.add_argument("--workdir", type=Path, default="examples/workdir/result")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--null_prompt", type=str, default="")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--cfg_guidance", type=float, default=0.6)
    parser.add_argument("--degradation", type=str, default="deblur")
    parser.add_argument("--inversion_time", type=int, default=300)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--DDS_step", type=int, default=10)
    parser.add_argument("--lpf_scale", type=int, default=2)
    parser.add_argument("--method", type=str, default='vision-xl')
    parser.add_argument("--model", type=str, default='sdxl', choices=["sd15", "sd20", "sdxl", "sdxl_lightning"])
    parser.add_argument("--NFE", type=int, default=25)
    parser.add_argument("--folder_path", type=str, default='examples/assets/pexels_sample/landscape')
    parser.add_argument("--ratio", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    #set_seed(args.seed)

    solver_config = munchify({'num_sampling': args.NFE })
    callback = ComposeCallback(workdir=args.workdir,
                               frequency=1,
                               callbacks=[])
    # callback = None
    H, W = SD_XL_BASE_RATIOS[f'{args.ratio}']
    
    # Data load
    all_folders = sorted([f.path for f in os.scandir(args.folder_path) if f.is_dir()])
    if len(all_folders) == 0:
        raise ValueError("Folder does not contain any folders.")
    print(f"Succesfully load {len(all_folders)} videos for reconstruction.")

    for index in range(len(all_folders)):
        folder_path = all_folders[index]
        # Load sample image batch
        video = load_image_batch(folder_path, H, W)

        if args.model == "sdxl" or args.model == "sdxl_lightning":
            solver = get_solver_sdxl(args.method,
                                    solver_config=solver_config,
                                    device=args.device)
            
            result, measurement = solver.sample(video=video,
                                    prompt1=[args.null_prompt, args.prompt],
                                    prompt2=[args.null_prompt, args.prompt],
                                    cfg_guidance=args.cfg_guidance,
                                    degradation=args.degradation, 
                                    inversion_time=args.inversion_time, 
                                    DDS_step=args.DDS_step,
                                    lpf_scale=args.lpf_scale,
                                    eta=args.eta,
                                    target_size=(H, W),
                                    callback_fn=callback)

        else:
            solver = get_solver(args.method,
                                solver_config=solver_config,
                                device=args.device)
            result = solver.sample(prompt=[args.null_prompt, args.prompt],
                                cfg_guidance=args.cfg_guidance,
                                callback_fn=callback)

        os.makedirs(os.path.join(args.workdir, 'recon'), exist_ok=True)
        os.makedirs(os.path.join(args.workdir, 'measurement'), exist_ok=True)
        
        folder_name = folder_path.split('/')[-1]
        save_video_to_gif(result, os.path.join(args.workdir, f'recon/{folder_name}.gif'))
        save_video_to_gif(measurement, os.path.join(args.workdir, f'measurement/{folder_name}.gif'))

if __name__ == "__main__":
    main()