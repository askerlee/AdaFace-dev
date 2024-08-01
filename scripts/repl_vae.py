import argparse
from scripts.repl_lib import save_ckpt, load_two_models

parser = argparse.ArgumentParser()
parser.add_argument("--base_ckpt", type=str, required=True, help="Path to the base checkpoint")
parser.add_argument("--vae_ckpt", type=str, required=True, help="Path to the checkpoint providing vae")
parser.add_argument("--out_ckpt", type=str, required=True, help="Path to the output checkpoint")
args = parser.parse_args()

base_ckpt, base_state_dict, vae_state_dict = load_two_models(args.base_ckpt, args.vae_ckpt)
# base_state_dict = sd_ckpt["state_dict"]

repl_count = 0

for k in base_state_dict:
    if k.startswith("first_stage_model."):
        # We replace the vae of a SD 1.5 ckpt (as the base) with a standalone VAE (as the donor). 
        # Therefore we remove the prefix "first_stage_model." from the donor ckpt.
        k2 = k.replace("first_stage_model.", "")
        if k2 not in vae_state_dict:
            print(f"!!!! '{k2}' not in VAE checkpoint")
            continue
        if base_state_dict[k].shape != vae_state_dict[k2].shape:
            print(f"!!!! '{k}' shape mismatch: {base_state_dict[k].shape} vs {vae_state_dict[k2].shape} !!!!")
            continue
        print(k)
        base_state_dict[k] = vae_state_dict[k2]
        repl_count += 1

if repl_count > 0:
    print(f"{repl_count} parameters replaced")
    save_ckpt(base_ckpt, base_state_dict, args.out_ckpt)
else:
    print("ERROR: No parameter replaced")
