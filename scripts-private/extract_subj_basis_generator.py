import torch
import sys
adaface_ckpt_path = sys.argv[1]

ckpt = torch.load(adaface_ckpt_path, map_location='cpu')
string_to_subj_basis_generator_dict = ckpt["string_to_subj_basis_generator_dict"]
new_adaface_ckpt_path = adaface_ckpt_path.replace(".pt", "-2.pt")
torch.save({'string_to_subj_basis_generator_dict': string_to_subj_basis_generator_dict}, new_adaface_ckpt_path)
print(f"Saved to {new_adaface_ckpt_path}")
