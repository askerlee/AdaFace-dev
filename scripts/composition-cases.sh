#!/usr/bin/fish
# " | " separated fields like "subject | prompt | folder | dreambooth-class-token".
set -l cases "taylorswift | as aikido girl, sudamerican girl, clear face, casual, white training clothes with black hakama and black belt | taylorswift-aikido | girl, instagram"
set -a cases "zendaya | underwater among fish, highly detailed, art by Miho Hirano | zendaya-underwater | girl, instagram"
set -a cases "zendaya | in a red dress traveling in indonesia, clear face | zendaya-indonesia | girl, instagram"
set -a cases "selenagomez | smelling a flower, roses everywhere, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha | selenagomez-roses | girl, instagram"
set -a cases "lilbub | swimming underwater, surrounded by tropical fish and coral reefs | lilbub-fish | tabby cat, instagram"
set -a cases "jenniferlawrence | by ilya kuvshinov, clear face, cloudy sky background lush landscape illustration concept art anime key visual by makoto shinkai, sharp focus | jenniferlawrence-anime | girl, instagram"
set -a cases "taylorswift | by ilya kuvshinov, clear face, cloudy sky background lush landscape illustration concept art anime key visual by makoto shinkai, sharp focus | taylorswift-anime | girl, instagram"
set -a cases "jiffpom | wearing superman's uniform, cute face | jiffpom-superman | pom dog, instagram"
set -a cases "masatosakai | having a cup of coffee | masatosakai-coffee | asian man, instagram"
# No "instagram" in the DreamBooth class for princessmonstertruck, otherwise "surfing" will be ignored by DreamBooth.
set -a cases "princessmonstertruck | surfing on the sea, clear face | princessmonstertruck-surf | black persian cat"
set -a cases "iainarmitage | as ben skywalker with lightsaber, star wars, by artgerm and moebius, hyperrealism, highly detailed, 8k, intricate, closeup, dynamic dramatic dark moody lighting, shadows, artstation, concept art, octane render, 8k | iainarmitage-starwars | young boy, instagram"
set -a cases "keanureeves | as obi-wan with lightsaber, star wars, by artgerm and moebius, hyperrealism, highly detailed, 8k, intricate, closeup, dynamic dramatic dark moody lighting, shadows, artstation, concept art, octane render, 8k | keanureeves-starwars | man, instagram"
set -a cases "selenagomez | as jedi woman with lightsaber, star wars, by artgerm and moebius, beautiful, hyperrealism, highly detailed, 8k, intricate, closeup, dynamic dramatic dark moody lighting, shadows, artstation, concept art, octane render, 8k | selenagomez-starwars | girl, instagram"
set -a cases "taylorswift | as jedi woman with lightsaber, star wars, by artgerm and moebius, beautiful, hyperrealism, highly detailed, 8k, intricate, closeup, dynamic dramatic dark moody lighting, shadows, artstation, concept art, octane render, 8k | taylorswift-starwars | girl, instagram"
set -a cases "michelleyeoh | a portrait of a {}, clear face, posing with a tabby cat, by justin gerard and greg rutkowski, digital art, realistic painting, dnd, character design, trending on artstation | michelleyeoh-cat | asian woman, instagram"
set -a cases "zendaya | a portrait of a cute brunette {}, clear face, posing with a tabby cat, by justin gerard and greg rutkowski, digital art, realistic painting, dnd, character design, trending on artstation | zendaya-cat | girl, instagram"
# Temporary clear the previous cases, so as to only generate the following cases. 
# Undo clearing by commenting out the following line.
# set cases
set -a cases "jiffpom | cute furry {} sitting on flying cloud, 3D render, cute, kawaii, isolated on white background, digital art, ethereal, volumetric lighting, dreamy, pastel colors, illustration for childrens book, hyperrealistic | jiffpom-cloud | pom dog, instagram"
set -Ux cases $cases
