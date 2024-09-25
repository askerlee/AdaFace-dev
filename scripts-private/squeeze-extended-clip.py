import torch
import sys
import argparse
argparse = argparse.ArgumentParser()
argparse.add_argument('--ckpt', type=str, required=True)
argparse.add_argument('--generator_indices', type=int, nargs='+', default=[])
argparse.add_argument('--divisor', type=int, default=2)
args = argparse.parse_args()

# "logs/VGGface2_HQ_masks2024-09-11T00-11-41_zero3-ada/checkpoints/embeddings_gs-21000.pt"
ckpt = torch.load(args.ckpt)
subj_basis_generator = ckpt['string_to_subj_basis_generator_dict']['z']
if isinstance(subj_basis_generator, torch.nn.ModuleList):
    subj_basis_generators = subj_basis_generator
else:
    subj_basis_generators = [subj_basis_generator]

total_num_squeezed_layers = 0
for i in args.generator_indices:
    subj_basis_generator = subj_basis_generators[i]
    num_squeezed_layers = subj_basis_generator.squeeze_prompt2token_proj_attention(divisor=args.divisor)
    total_num_squeezed_layers += num_squeezed_layers

if total_num_squeezed_layers == 0:
    print("No squeeze was performed")
    sys.exit(0)
else:
    print(f"Squeezed {total_num_squeezed_layers} layers by a factor of {args.divisor}")
    ckpt_new_path = args.ckpt.replace('.pt', 'a.pt')
    torch.save(ckpt, ckpt_new_path)
    print(f"Saved squeezed checkpoint to {ckpt_new_path}")
