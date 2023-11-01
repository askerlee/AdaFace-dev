from evaluation.clip_eval import CLIPEvaluator

clip = CLIPEvaluator(device='cuda')
triplets = [ ['camel', 'giraffe', 'cashmere'], 
             ['snail', 'ladybug', 'winding'], 
             ['dietitian', 'pharmacist', 'nutritious'], 
             ['snake', 'twisted', 'gecko'],
             ['reflections of earth', 'sphere', 'civilization'],
             ['fear', 'scream', 'wolf']
           ]  

for triplet in triplets:
    wa, wb, wc = triplet
    print(f"Triplet: '{wa}' vs '{wb}'\t'{wc}'")
    sim_ab, sim_ac = clip.text_pairwise_similarity((wa,), (wb, wc), reduction='none')[0]
    print(f"'{wa}', '{wb}': {sim_ab:.3f}, '{wa}', '{wc}': {sim_ac:.3f}")
    fa, fb, fc = [ clip.get_text_features(w) for w in triplet ]
    fbc = fb + fc
    fbc = fbc / fbc.norm(dim=-1, keepdim=True)
    sim_a_bc = (fa * fbc).sum()
    print(f"'{wa}', '{wb}' + '{wc}': {sim_a_bc:.3f}")
    print()
