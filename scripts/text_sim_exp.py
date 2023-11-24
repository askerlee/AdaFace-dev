import torch
from evaluation.clip_eval import CLIPEvaluator

clip = CLIPEvaluator(clip_model='ViT-L/14', device='cuda')
triplets = [ ['camel', 'giraffe', 'cashmere'], 
             ['snail', 'ladybug', 'winding'], 
             ['dietitian', 'pharmacist', 'nutritious'], 
             ['snake', 'twisted', 'gecko'],
             ['reflections of earth', 'sphere', 'civilization'],
             ['fear', 'scream', 'wolf'],
             ['snail', 'table', 'cake'],
             ["camel", "giraffe", "door"]
           ]  

for triplet in triplets:
    wa, wb, wc = triplet
    print(f"Triplet: '{wa}' vs '{wb}'\t'{wc}'")
    for get_token_emb in (False, True):
      # fa, fb, fc: [1, 768].
      fa, fb, fc = [ clip.get_text_features(w, get_token_emb=get_token_emb) for w in triplet ]
      fb_fc = torch.cat([fb, fc], dim=0)
      if fa.shape[0] > 1:
        fa = fa.mean(dim=0, keepdim=True)
        
      sim_ab, sim_ac = torch.cosine_similarity(fa, fb_fc, dim=-1)
      print(f"'{wa}', '{wb}': {sim_ab:.3f}, '{wa}', '{wc}': {sim_ac:.3f}")
      fbc = fb + fc
      fbc = fbc / fbc.norm(dim=-1, keepdim=True)
      sim_a_bc = (fa * fbc).sum()
      print(f"'{wa}', '{wb}' + '{wc}': {sim_a_bc:.3f}")
      print()
