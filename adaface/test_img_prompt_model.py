import torch
from PIL import Image
import os, argparse, glob
from .face_id_to_ada_prompt import create_id2img_prompt_encoder
from .teacher_pipelines import create_arc2face_pipeline, create_consistentid_pipeline
from transformers import CLIPTextModel
import numpy as np

def save_images(images, subject_name, id2img_prompt_encoder_type,
                prompt, noise_level, save_dir = "samples-ada"):
    os.makedirs(save_dir, exist_ok=True)
    # Save 4 images as a grid image in save_dir
    grid_image = Image.new('RGB', (512 * 2, 512 * 2))
    for i, image in enumerate(images):
        image = image.resize((512, 512))
        grid_image.paste(image, (512 * (i % 2), 512 * (i // 2)))

    prompt_sig = prompt.replace(" ", "_").replace(",", "_")
    grid_filepath = os.path.join(save_dir, 
                "-".join([subject_name, id2img_prompt_encoder_type, 
                          prompt_sig, f"noise{noise_level:.02f}.png"]))
    
    if os.path.exists(grid_filepath):
        grid_count = 2
        grid_filepath = os.path.join(save_dir, 
                        "-".join([ subject_name, id2img_prompt_encoder_type, 
                                   prompt_sig, f"noise{noise_level:.02f}", str(grid_count) ]) + ".png")
        while os.path.exists(grid_filepath):
            grid_count += 1
            grid_filepath = os.path.join(save_dir, 
                        "-".join([ subject_name, id2img_prompt_encoder_type, 
                                   prompt_sig, f"noise{noise_level:.02f}", str(grid_count) ]) + ".png")

    grid_image.save(grid_filepath)
    print(f"Saved to {grid_filepath}")

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PL_GLOBAL_SEED"] = str(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # --base_model_path models/Realistic_Vision_V4.0_noVAE
    parser.add_argument("--base_model_path", type=str, default="models/sar/sar.safetensors")    
    parser.add_argument("--id2img_prompt_encoder_type", type=str, default="arc2face",
                        choices=["arc2face", "consistentID"], help="Type of the ID2Img prompt encoder")    
    parser.add_argument("--subject", type=str, default="subjects-celebrity/taylorswift")
    parser.add_argument("--example_image_count", type=int, default=5, help="Number of example images to use")
    parser.add_argument("--out_image_count",     type=int, default=4, help="Number of images to generate")
    parser.add_argument("--prompt", type=str, default="portrait photo of a person in superman costume")
    parser.add_argument("--use_teacher_neg", action="store_true")
    parser.add_argument("--use_core_only", action="store_true")
    parser.add_argument("--noise", type=float, default=0)
    parser.add_argument("--randface", action="store_true")
    parser.add_argument("--seed", type=int, default=-1)

    args = parser.parse_args()
    if args.seed > 0:
        seed_everything(args.seed)

    if args.id2img_prompt_encoder_type == "arc2face":
        pipeline = create_arc2face_pipeline(args.base_model_path)
    elif args.id2img_prompt_encoder_type == "consistentID":
        pipeline = create_consistentid_pipeline(args.base_model_path)

    pipeline = pipeline.to('cuda', torch.float16)

    id2img_prompt_encoder = create_id2img_prompt_encoder(args.id2img_prompt_encoder_type)
    id2img_prompt_encoder.to('cuda')

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
            # image_paths contain at most args.example_image_count full image paths.
            image_paths = alltype_image_paths[:args.example_image_count]
    else:
        subject_name = None
        image_paths = None
        image_folder = None

    subject_name = "randface-" + str(torch.seed()) if args.randface else subject_name
    id_batch_size = args.out_image_count

    input_max_length = 22
    text_encoder = pipeline.text_encoder
    orig_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to("cuda") 

    # Noise level is the *relative* std of the noise added to the face embeddings.
    # A noise level of 0.08 could change gender, but 0.06 is usually safe.
    for noise_level in (0,): # 0.03):
        if args.randface:
            init_id_embs = torch.randn(1, 512, device='cuda', dtype=torch.float16)
            if args.id2img_prompt_encoder_type == "arc2face":
                pre_clip_features = None
            elif args.id2img_prompt_encoder_type == "consistentID":
                # For ConsistentID, random clip features are much better than zero clip features.
                rand_clip_fgbg_features = torch.randn(1, 514, 1280, device='cuda', dtype=torch.float16)
                rand_clip_neg_features  = torch.randn(1, 257, 1280, device='cuda', dtype=torch.float16)
                pre_clip_features = (rand_clip_fgbg_features, rand_clip_neg_features)
            else:
                breakpoint()
        else:
            init_id_embs = None
            pre_clip_features = None

        # id_prompt_emb is in the image prompt space.
        # neg_id_prompt_emb is used in ConsistentID only.
        face_image_count, faceid_embeds, id_prompt_emb, neg_id_prompt_emb \
            = id2img_prompt_encoder.get_img_prompt_embs( \
                init_id_embs=init_id_embs,
                pre_clip_features=pre_clip_features,
                image_paths=image_paths,
                image_objs=None,
                id_batch_size=id_batch_size,
                noise_level=noise_level,
                return_core_id_embs_only=False,
                avg_at_stage='id_emb',
                verbose=True)

        pipeline.text_encoder = orig_text_encoder

        comp_prompt     = args.prompt 
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        # prompt_embeds_, negative_prompt_embeds_: [4, 77, 768]
        prompt_embeds_, negative_prompt_embeds_ = \
            pipeline.encode_prompt(comp_prompt, device='cuda', num_images_per_prompt=args.out_image_count,
                                   do_classifier_free_guidance=True, negative_prompt=negative_prompt)
        #pipeline.text_encoder = text_encoder
        # Postpend the id prompt embeddings to the prompt embeddings.
        # For arc2face, id_prompt_emb can be either pre- or post-pended.
        # But for ConsistentID, id_prompt_emb has to be **post-pended**. Otherwise, the result images are blank.
        prompt_embeds_ = torch.cat([prompt_embeds_, id_prompt_emb], dim=1)
        M = id_prompt_emb.shape[1]

        if (not args.use_teacher_neg) or neg_id_prompt_emb is None:
            # For arc2face, neg_id_prompt_emb is None. So we concatenate the last M negative prompt embeddings,
            # to make the negative prompt embeddings have the same length as the prompt embeddings.
            negative_prompt_embeds_ = torch.cat([negative_prompt_embeds_, negative_prompt_embeds_[:, -M:]], dim=1)
        else:
            # NOTE: For ConsistentID, neg_id_prompt_emb has to be present in the negative prompt embeddings.
            # Otherwise, the result images are cartoonish.
            negative_prompt_embeds_ = torch.cat([negative_prompt_embeds_, neg_id_prompt_emb], dim=1)

        if args.use_core_only:
            prompt_embeds_ = id_prompt_emb
            if (not args.use_teacher_neg) or neg_id_prompt_emb is None:
                negative_prompt_embeds_ = negative_prompt_embeds_[:, :M]
            else:
                negative_prompt_embeds_ = neg_id_prompt_emb

        noise = torch.randn(args.out_image_count, 4, 64, 64, device='cuda', dtype=torch.float16)
        
        for guidance_scale in [4]:
            images = pipeline(latents=noise,
                              prompt_embeds=prompt_embeds_, 
                              negative_prompt_embeds=negative_prompt_embeds_, 
                              num_inference_steps=40, 
                              guidance_scale=guidance_scale, 
                              num_images_per_prompt=1).images

            save_images(images, subject_name, args.id2img_prompt_encoder_type, 
                        f"guide{guidance_scale}", noise_level)
