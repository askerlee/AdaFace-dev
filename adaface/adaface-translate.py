from adaface.adaface_wrapper import AdaFaceWrapper
import torch
#import torch.nn.functional as F
from PIL import Image
import numpy as np
import os, argparse, glob, re, shutil

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PL_GLOBAL_SEED"] = str(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default='models/realisticvision/realisticVisionV40_v40VAE.safetensors', 
                        help="Path to the UNet checkpoint (default: RealisticVision 4.0)")
    parser.add_argument("--embman_ckpt", type=str, required=True,
                        help="Path to the checkpoint of the embedding manager")
    parser.add_argument("--in_folder",  type=str, required=True, help="Path to the folder containing input images")
    # If True, the input folder contains images of mixed subjects.
    # If False, the input folder contains multiple subfolders, each of which contains images of the same subject.
    parser.add_argument("--is_mix_subj_folder", type=str2bool, const=True, default=False, nargs="?", 
                        help="Whether the input folder contains images of mixed subjects")
    parser.add_argument("--max_images_per_subject", type=int, default=5, help="Number of example images used per subject")
    parser.add_argument("--trans_subject_count", type=int, default=-1, help="Number of example images to be translated")
    parser.add_argument("--out_folder", type=str, required=True, help="Path to the folder saving output images")
    parser.add_argument("--out_count_per_input_image", type=int, default=1,  help="Number of output images to generate per input image")
    parser.add_argument("--copy_masks", action="store_true", help="Copy the mask images to the output folder")
    parser.add_argument("--noise", dest='noise_level', type=float, default=0)
    parser.add_argument("--scale", dest='guidance_scale', type=float, default=4, 
                        help="Guidance scale for the diffusion model")
    parser.add_argument("--ref_img_strength", type=float, default=0.8,
                        help="Strength of the reference image in the output image.")
    parser.add_argument("--subject_string", 
                        type=str, default="z",
                        help="Subject placeholder string used in prompts to denote the concept.")
    parser.add_argument("--num_vectors", type=int, default=16,
                        help="Number of vectors used to represent the subject.")
    parser.add_argument("--prompt", type=str, default="a person z")
    parser.add_argument("--num_images_per_row", type=int, default=4,
                        help="Number of images to display in a row in the output grid image.")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of DDIM inference steps")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use. If num_gpus > 1, use accelerate for distributed execution.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--seed", type=int, default=42, 
                        help="the seed (for reproducible sampling). Set to -1 to disable.")
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.seed != -1:
        seed_everything(args.seed)
 
# screen -dm -L -Logfile trans_rv4-2.txt accelerate launch --multi_gpu --num_processes=2 scripts/adaface-translate.py 
# --embman_ckpt logs/subjects-celebrity2024-05-16T17-22-46_zero3-ada/checkpoints/embeddings_gs-30000.pt 
# --base_model_path models/realisticvision/realisticVisionV40_v40VAE.safetensors --in_folder /data/shaohua/VGGface2_HQ_masks/ 
# --is_mix_subj_folder 0 --out_folder /data/shaohua/VGGface2_HQ_masks_rv4a --copy_masks --num_gpus 2
    if args.num_gpus > 1:
        from accelerate import PartialState
        distributed_state = PartialState()
        args.device = distributed_state.device
        process_index = distributed_state.process_index
    elif re.match(r"^\d+$", args.device):
        args.device = f"cuda:{args.device}"
        distributed_state = None
        process_index = 0

    adaface = AdaFaceWrapper("img2img", args.base_model_path, args.embman_ckpt, args.device, 
                             args.subject_string, args.num_vectors, args.num_inference_steps)

    in_folder = args.in_folder
    if os.path.isfile(in_folder):
        subject_folders = [ os.path.dirname(in_folder) ]
        images_by_subject = [[in_folder]]
    else:
        if not args.is_mix_subj_folder:
            in_folders = [in_folder]
        else:
            in_folders = [ os.path.join(in_folder, subfolder) for subfolder in sorted(os.listdir(in_folder)) ]
        
        images_by_subject = []
        subject_folders   = []
        for in_folder in in_folders:                
            image_types = ["*.jpg", "*.png", "*.jpeg"]
            alltype_image_paths = []
            for image_type in image_types:
                # glob returns the full path.
                image_paths = glob.glob(os.path.join(in_folder, image_type))
                if len(image_paths) > 0:
                    alltype_image_paths.extend(image_paths)

            # Filter out images of "*_mask.png"
            alltype_image_paths = [image_path for image_path in alltype_image_paths if "_mask.png" not in image_path]
            alltype_image_paths = sorted(alltype_image_paths)

            if not args.is_mix_subj_folder:
                # image_paths contain at most args.max_images_per_subject full image paths.
                if args.max_images_per_subject > 0:
                    image_paths = alltype_image_paths[:args.max_images_per_subject]
                else:
                    image_paths = alltype_image_paths

                images_by_subject.append(image_paths)
                subject_folders.append(in_folder)
            else:
                # Each image in the folder is treated as an individual subject.
                images_by_subject.extend([[image_path] for image_path in alltype_image_paths])
                subject_folders.extend([in_folder] * len(alltype_image_paths))

            if args.trans_subject_count > 0 and len(subject_folders) >= args.trans_subject_count:
                break

    if args.trans_subject_count > 0:
        images_by_subject = images_by_subject[:args.trans_subject_count]
        subject_folders   = subject_folders[:args.trans_subject_count]

    out_image_count = 0
    out_mask_count  = 0
    if not args.out_folder.endswith("/"):
        args.out_folder += "/"

    if args.num_gpus > 1:
        # Split the subjects across the GPUs.
        subject_folders = subject_folders[process_index::args.num_gpus]
        images_by_subject = images_by_subject[process_index::args.num_gpus]
        #subject_folders, images_by_subject = distributed_state.split_between_processes(zip(subject_folders, images_by_subject))

    for (subject_folder, image_paths) in zip(subject_folders, images_by_subject):
        # If is_mix_subj_folder, then image_paths only contains 1 image, and we use the file name as the signature of the image.
        # Otherwise, we use the folder name as the signature of the images.
        images_sig = subject_folder if not args.is_mix_subj_folder else os.path.basename(image_paths[0])

        print(f"Translating {images_sig}...")
        with torch.no_grad():
            adaface_subj_embs = adaface.generate_adaface_embeddings(image_paths, subject_folder, None, False, 
                                                                    out_id_embs_scale=1, noise_level=args.noise_level, 
                                                                    update_text_encoder=True)

        # Replace the first occurrence of "in_folder" with "out_folder" in the path of the subject_folder.
        subject_out_folder = subject_folder.replace(args.in_folder, args.out_folder, 1)
        if not os.path.exists(subject_out_folder):
            os.makedirs(subject_out_folder)
        print(f"Output images will be saved to {subject_out_folder}")

        in_images = []
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB").resize((512, 512))
            # [512, 512, 3] -> [3, 512, 512].
            image = np.array(image).transpose(2, 0, 1)
            # Convert the image to a tensor of shape (1, 3, 512, 512) and move it to the GPU.
            image = torch.tensor(image).unsqueeze(0).float().cuda()
            in_images.append(image)

        # Put all input images of the subject into a batch. This assumes max_images_per_subject is small.
        # NOTE: For simplicity, we do not check overly large batch sizes.
        in_images = torch.cat(in_images, dim=0) 
        # in_images: [5, 3, 512, 512].
        # Normalize the pixel values to [0, 1].
        in_images = in_images / 255.0
        num_out_images = len(in_images) * args.out_count_per_input_image

        with torch.no_grad():
            # args.noise_level: the *relative* std of the noise added to the face embeddings.
            # A noise level of 0.08 could change gender, but 0.06 is usually safe.
            # The returned adaface_subj_embs are already incorporated in the text encoder, and not used explicitly.
            # NOTE: We assume out_count_per_input_image == 1, so that the output images are of the same number as the input images.
            out_images = adaface(in_images, args.prompt, args.guidance_scale, num_out_images, ref_img_strength=args.ref_img_strength)

            for img_i, img in enumerate(out_images):
                # out_images: subj_1, subj_2, ..., subj_n, subj_1, subj_2, ..., subj_n, ...
                subj_i = img_i %  len(in_images) 
                copy_i = img_i // len(in_images)
                image_filename_stem, image_fileext = os.path.splitext(os.path.basename(image_paths[subj_i]))
                if copy_i == 0:
                    img.save(os.path.join(subject_out_folder, f"{image_filename_stem}{image_fileext}"))
                else:
                    img.save(os.path.join(subject_out_folder, f"{image_filename_stem}_{copy_i}{image_fileext}"))

                if args.copy_masks:
                    mask_path = image_paths[subj_i].replace(image_fileext, "_mask.png")
                    if os.path.exists(mask_path):
                        if copy_i == 0:
                            shutil.copy(mask_path, subject_out_folder)
                        else:
                            mask_filename_stem = image_filename_stem
                            shutil.copy(mask_path, os.path.join(subject_out_folder, f"{mask_filename_stem}_{copy_i}_mask.png"))

                        out_mask_count += 1

            out_image_count += len(out_images)

    print(f"{out_image_count} output images and {out_mask_count} masks saved to {args.out_folder}")
