import webdataset as wds
import re
comp_wds_path = "../img2dataset/mscoco/00000.tar"
comp_wds = wds.WebDataset(comp_wds_path).shuffle(100).decode("pil").to_tuple("jpg;png", "json")
comp_wds_iter = iter(comp_wds)

all_count = 0
nonhuman_count = 0
size = 512

single_human_pat = "man|woman|person|boy|girl|child|kid|baby|adult|guy|lady|gentleman|lady|male|female|human"
single_role_pat  = "cook|chef|waiter|waitress|doctor|nurse|policeman|policewoman|fireman|firewoman|firefighter|teacher|student|professor|driver|pilot|farmer|worker|artist|painter|photographer|dancer|singer|musician|player|athlete|player|biker|cyclist|bicyclist"
plural_human_pat = "men|women|people|boys|girls|children|kids|babies|adults|guys|ladies|gentlemen|ladies|males|females|humans"
plural_role_pat  = "cooks|chefs|waiters|waitresses|doctors|nurses|policemen|policewomen|firemen|firewomen|firefighters|teachers|students|professors|drivers|pilots|farmers|workers|artists|painters|photographers|dancers|singers|musicians|players|athletes|players|bikers|cyclists|bicyclists"
animal_pat       = "cat|cats|dog|dogs"
human_pat = "|".join([single_human_pat, single_role_pat, plural_human_pat, plural_role_pat, animal_pat])

for bg_img, bg_json in comp_wds_iter:
    bg_prompt = bg_json['caption'].lower()
    if re.search(human_pat, bg_prompt):
        not_human = False
        nonhuman_count += 1
    else:
        not_human = True

    if not_human and nonhuman_count < 300:
        print(bg_prompt)

    all_count += 1

print(all_count, nonhuman_count)
