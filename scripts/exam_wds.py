import webdataset as wds
comp_wds_path = "../img2dataset/mscoco/00000.tar"
comp_wds = wds.WebDataset(comp_wds_path).shuffle(100).decode("pil").to_tuple("jpg;png", "json")
comp_wds_iter = iter(comp_wds)

print_count = 0

for bg_img, bg_json in comp_wds_iter:
    bg_prompt = bg_json['caption'].lower()
    if 'portrait' in bg_prompt:
        print(bg_prompt)
        print(bg_json['url'])
        print_count += 1
        if print_count >= 100:
            break
        