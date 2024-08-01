import argparse
from scripts.repl_lib import save_ckpt, load_two_models, str2bool
import re

parser = argparse.ArgumentParser()
parser.add_argument("--base_ckpt", type=str, required=True, help="Path to the base checkpoint")
parser.add_argument("--donor_ckpt", type=str, required=True, help="Path to the checkpoint providing parts")
parser.add_argument("--out_ckpt", type=str, required=True, help="Path to the output checkpoint")
parser.add_argument("--pattern", type=str, required=True, help="Pattern in the parts to be implanted from the donor checkpoint")
parser.add_argument("--skip_ema", type=str2bool, const=True, default=True, nargs="?", help="Skip EMA weights")
args = parser.parse_args()

base_ckpt, base_state_dict, donor_state_dict = load_two_models(args.base_ckpt, args.donor_ckpt)
# base_state_dict = sd_ckpt["state_dict"]

repl_count = 0

for k in base_state_dict:
    if args.skip_ema and k.startswith("model_ema."):
        continue
    if re.search(args.pattern, k):
        if k not in donor_state_dict:
            print(f"!!!! '{k}' not in the donor checkpoint")
            continue
        if base_state_dict[k].shape != donor_state_dict[k].shape:
            print(f"!!!! '{k}' shape mismatch: {base_state_dict[k].shape} vs {donor_state_dict[k].shape} !!!!")
            continue
        print(k)
        base_state_dict[k] = donor_state_dict[k]
        repl_count += 1

if repl_count > 0:
    print(f"{repl_count} parameters replaced")
    save_ckpt(base_ckpt, base_state_dict, args.out_ckpt)
else:
    print("ERROR: No parameter replaced")
