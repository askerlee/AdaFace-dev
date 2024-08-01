import argparse
from scripts.repl_lib import save_ckpt, load_two_models

parser = argparse.ArgumentParser()
parser.add_argument("--base_ckpt", type=str, required=True, help="Path to the base checkpoint")
parser.add_argument("--te_ckpt", type=str, required=True, help="Path to the checkpoint providing text encoder")
parser.add_argument("--out_ckpt", type=str, required=True, help="Path to the output checkpoint")
args = parser.parse_args()

base_ckpt, base_state_dict, te_state_dict = load_two_models(args.base_ckpt, args.te_ckpt)
# base_state_dict = sd_ckpt["state_dict"]

repl_count = 0

for k in base_state_dict:
    if k.startswith("cond_stage_model."):
        if k not in te_state_dict:
            print(f"!!!! '{k}' not in TE checkpoint")
            continue
        if base_state_dict[k].shape != te_state_dict[k].shape:
            print(f"!!!! '{k}' shape mismatch: {base_state_dict[k].shape} vs {te_state_dict[k].shape} !!!!")
            continue
        print(k)
        base_state_dict[k] = te_state_dict[k]
        repl_count += 1

if repl_count > 0:
    print(f"{repl_count} parameters replaced")
    save_ckpt(base_ckpt, base_state_dict, args.out_ckpt)
else:
    print("ERROR: No parameter replaced")
