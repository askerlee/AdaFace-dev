import torch
import sys
# "logs/VGGface2_HQ_masks2024-09-11T00-11-41_zero3-ada/checkpoints/embeddings_gs-21000.pt"
ckpt = torch.load(sys.argv[1])
subj_basis_generator = ckpt['string_to_subj_basis_generator_dict']['z']
layers = subj_basis_generator.prompt2token_proj.text_model.encoder.layers
num_deduped_chunks = 0
N = 4

def calc_and_print_stats(ts, ts_name=None):
    if ts_name is not None:
        print(f"{ts_name}: ", end='')
    print("max: %.4f, min: %.4f, mean: %.4f, std: %.4f" %(ts.max(), ts.min(), ts.abs().mean(), ts.std()))

for layer_idx, layer in enumerate(layers):
    v_diff_stds = []
    v_diff_std_ratios = []
    old_v_shape = list(layer.self_attn.v_proj.weight.shape)
    vs = layer.self_attn.v_proj.weight.chunk(N, dim=0)
    dups = [False] * N

    for i in range(N):
        for j in range(i+1, N):
            if (vs[i] == vs[j]).all():
                dups[j] = True
            v_diff_std = (vs[j] - vs[i]).std()
            v_diff_stds.append(v_diff_std)
            v_diff_std_ratios.append(v_diff_std / (vs[i].std() + 1e-6))

    deduped_vs = [vs[i] for i in range(N) if not dups[i]]
    if len(deduped_vs) < N:
        deduped_v = torch.cat(deduped_vs, dim=0)
        v_biases = layer.self_attn.v_proj.bias.chunk(N, dim=0)
        # We don't check for duplicates in biases, because they only play a minor role.
        deduped_v_biases = [v_biases[i] for i in range(N) if not dups[i]]
        deduped_v_bias = torch.cat(deduped_v_biases, dim=0)
        layer.self_attn.v_proj.weight.data  = deduped_v
        layer.self_attn.v_proj.bias.data    = deduped_v_bias
        layer.self_attn.v_proj.out_features = deduped_v.size(0)
        print(f"Layer {layer_idx} V deduplicated: {old_v_shape} -> {list(deduped_v.shape)}")
        num_deduped_chunks += 1
        v_dedup_ratio = len(deduped_vs) / N
    else:
        print(f"Layer {layer_idx} V is different: {old_v_shape}")
        v_dedup_ratio = 1

    calc_and_print_stats(torch.tensor(v_diff_stds),       "V diff stds")
    calc_and_print_stats(torch.tensor(v_diff_std_ratios), "V diff std ratios")

    k_diff_stds = []
    k_diff_std_ratios = []
    old_k_shape = list(layer.self_attn.k_proj.weight.shape)
    ks = layer.self_attn.k_proj.weight.chunk(N, dim=0)
    dups = [False] * N
    for i in range(N):
        for j in range(i+1, N):
            if (ks[i] == ks[j]).all():
                dups[j] = True
            k_diff_std = (ks[j] - ks[i]).std()
            k_diff_stds.append(k_diff_std)
            k_diff_std_ratios.append(k_diff_std / (ks[i].std() + 1e-6))

    deduped_ks = [ks[i] for i in range(N) if not dups[i]]
    if len(deduped_ks) < N:
        deduped_k = torch.cat(deduped_ks, dim=0)
        k_biases = layer.self_attn.k_proj.bias.chunk(N, dim=0)
        # We don't check for duplicates in biases, because they only play a minor role.
        deduped_k_biases = [k_biases[i] for i in range(N) if not dups[i]]
        deduped_k_bias = torch.cat(deduped_k_biases, dim=0)
        layer.self_attn.k_proj.weight.data  = deduped_k
        layer.self_attn.k_proj.bias.data    = deduped_k_bias
        layer.self_attn.k_proj.out_features = deduped_k.size(0)
        print(f"Layer {layer_idx} K deduplicated: {old_k_shape} -> {list(deduped_k.shape)}")
        num_deduped_chunks += 1
        k_dedup_ratio = len(deduped_ks) / N
    else:
        print(f"Layer {layer_idx} K is different: {old_k_shape}")
        k_dedup_ratio = 1

    calc_and_print_stats(torch.tensor(k_diff_stds), "K diff stds")
    calc_and_print_stats(torch.tensor(k_diff_std_ratios), "K diff std ratios")

    if v_dedup_ratio < 1 and k_dedup_ratio < 1:
        print(f"Layer {layer_idx} v dedup ratio: {v_dedup_ratio:.2f}, k dedup ratio: {k_dedup_ratio:.2f}")
        if v_dedup_ratio != k_dedup_ratio:
            print(f"Warning: deduplication ratios differ")
        else:
            orig_layer_multiplier = subj_basis_generator.prompt2token_proj_attention_multipliers[layer_idx]
            if int(orig_layer_multiplier * v_dedup_ratio) != orig_layer_multiplier * v_dedup_ratio:
                print(f"Warning: deduplication ratio is not a multiple of original layer multiplier")
            else:
                new_layer_multiplier = int(orig_layer_multiplier * v_dedup_ratio)
                print(f"Layer {layer_idx} multiplier: {orig_layer_multiplier} -> {new_layer_multiplier}")
                subj_basis_generator.prompt2token_proj_attention_multipliers[layer_idx] = new_layer_multiplier
                
if num_deduped_chunks == 0:
    print("No deduplication was performed")
    sys.exit(0)
else:
    print(f"Deduplicated {num_deduped_chunks}/{len(layers) * N} chunks")
    ckpt_new_path = sys.argv[1].replace('.pt', 'a.pt')
    torch.save(ckpt, ckpt_new_path)
    print(f"Saved deduplicated checkpoint to {ckpt_new_path}")
