import webdataset as wds
comp_wds_path = "../img2dataset/mscoco/00000.tar"
comp_wds = wds.WebDataset(comp_wds_path).shuffle(100).decode("pil").to_tuple("jpg;png", "json")
comp_wds_iter = iter(comp_wds)

all_count = 0
too_small_count = 0
size = 512

for bg_img, bg_json in comp_wds_iter:
    bg_prompt = bg_json['caption'].lower()
    orig_h, orig_w = bg_json['original_height'], bg_json['original_width']
    edge_ratio = max(size / orig_h, size / orig_w)
    if edge_ratio >= 1.3:
        is_too_small = True
        too_small_count += 1
    else:
        is_too_small = False    
    if is_too_small and too_small_count < 10:
        print(bg_json['url'])

    all_count += 1

print(all_count, too_small_count)
