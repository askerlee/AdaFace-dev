from adaface.adaface_wrapper import AdaFaceWrapper
import torch
#import torch.nn.functional as F
from PIL import Image
import numpy as np
import os, argparse, glob, re

def save_images(images, num_images_per_row, subject_name, prompt, noise_level, save_dir = "samples-ada"):
    if num_images_per_row > len(images):
        num_images_per_row = len(images)
        
    os.makedirs(save_dir, exist_ok=True)
        
    num_columns = int(np.ceil(len(images) / num_images_per_row))
    # Save 4 images as a grid image in save_dir
    grid_image = Image.new('RGB', (512 * num_images_per_row, 512 * num_columns))
    for i, image in enumerate(images):
        image = image.resize((512, 512))
        grid_image.paste(image, (512 * (i % num_images_per_row), 512 * (i // num_images_per_row)))

    prompt_sig = prompt.replace(" ", "_").replace(",", "_")
    grid_filepath = os.path.join(save_dir, f"{subject_name}-{prompt_sig}-noise{noise_level:.02f}.png")
    if os.path.exists(grid_filepath):
        grid_count = 2
        grid_filepath = os.path.join(save_dir, f'{subject_name}-{prompt_sig}-noise{noise_level:.02f}-{grid_count}.jpg')
        while os.path.exists(grid_filepath):
            grid_count += 1
            grid_filepath = os.path.join(save_dir, f'{subject_name}-{prompt_sig}-noise{noise_level:.02f}-{grid_count}.jpg')

    grid_image.save(grid_filepath)
    print(f"Saved to {grid_filepath}")

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PL_GLOBAL_SEED"] = str(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default='runwayml/stable-diffusion-v1-5', 
                        help="Type of checkpoints to use (default: SD 1.5)")
    parser.add_argument("--embman_ckpt", type=str, required=True,
                        help="Path to the checkpoint of the embedding manager")
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--example_image_count", type=int, default=-1, help="Number of example images to use")
    parser.add_argument("--out_image_count",     type=int, default=4,  help="Number of images to generate")
    parser.add_argument("--prompt", type=str, default="a woman z in superman costume")
    parser.add_argument("--noise", dest='noise_level', type=float, default=0)
    parser.add_argument("--randface", action="store_true")
    parser.add_argument("--scale", dest='guidance_scale', type=float, default=4, 
                        help="Guidance scale for the diffusion model")
    parser.add_argument("--id_cfg_scale", type=float, default=1, 
                        help="CFG scale when generating the identity embeddings")
    
    parser.add_argument("--subject_string", 
                        type=str, default="z",
                        help="Subject placeholder string used in prompts to denote the concept.")
    parser.add_argument("--num_vectors", type=int, default=16,
                        help="Number of vectors used to represent the subject.")
    parser.add_argument("--num_images_per_row", type=int, default=4,
                        help="Number of images to display in a row in the output grid image.")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of DDIM inference steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--seed", type=int, default=42, 
                        help="the seed (for reproducible sampling). Set to -1 to disable.")
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.seed != -1:
        seed_everything(args.seed)

    if re.match(r"^\d+$", args.device):
        args.device = f"cuda:{args.device}"
    print(f"Using device {args.device}")

    adaface = AdaFaceWrapper("text2img", args.base_model_path, args.embman_ckpt, args.device, 
                             args.subject_string, args.num_vectors, args.num_inference_steps)

    if not args.randface:
        image_folder = args.subject
        if image_folder.endswith("/"):
            image_folder = image_folder[:-1]

        if os.path.isfile(image_folder):
            # Get the second to the last part of the path
            subject_name = os.path.basename(os.path.dirname(image_folder))
            image_paths = [image_folder]

        else:
            subject_name = os.path.basename(image_folder)
            image_types = ["*.jpg", "*.png", "*.jpeg"]
            alltype_image_paths = []
            for image_type in image_types:
                # glob returns the full path.
                image_paths = glob.glob(os.path.join(image_folder, image_type))
                if len(image_paths) > 0:
                    alltype_image_paths.extend(image_paths)

            # Filter out images of "*_mask.png"
            alltype_image_paths = [image_path for image_path in alltype_image_paths if "_mask.png" not in image_path]

            # image_paths contain at most args.example_image_count full image paths.
            if args.example_image_count > 0:
                image_paths = alltype_image_paths[:args.example_image_count]
            else:
                image_paths = alltype_image_paths
    else:
        subject_name = None
        image_paths = None
        image_folder = None

    subject_name = "randface-" + str(torch.seed()) if args.randface else subject_name
    rand_face_embs = torch.randn(1, 512)

    pre_face_embs = rand_face_embs if args.randface else None
    noise = torch.randn(args.out_image_count, 4, 64, 64).cuda()
    # args.noise_level: the *relative* std of the noise added to the face embeddings.
    # A noise level of 0.08 could change gender, but 0.06 is usually safe.
    # adaface_subj_embs is not used. It is generated for the purpose of updating the text encoder (within this function call).
    adaface_subj_embs = adaface.generate_adaface_embeddings(image_paths, image_folder, pre_face_embs, args.randface, 
                                                            out_id_embs_scale=args.id_cfg_scale, noise_level=args.noise_level, 
                                                            update_text_encoder=True)    
    images = adaface(noise, args.prompt, args.guidance_scale, args.out_image_count, verbose=True)
    save_images(images, args.num_images_per_row, subject_name, f"guide{args.guidance_scale}", args.noise_level)
