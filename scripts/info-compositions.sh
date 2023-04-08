#!/usr/bin/fish
# " | " separated fields like "subject | prompt | folder | dreambooth-class-token".
# set -l cases "taylorswift | as aikido girl, sudamerican girl, clear face, casual, white training clothes with black hakama and black belt | taylorswift-aikido | girl, instagram"
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
set -a cases "taylorswift | neutral face, wearing blue silk dress with diamond studded lily iris flowers, fashion model style, high resolution, realistic anatomy photography | taylorswift-bluedress | girl, instagram"
set -a cases "tomholland | wearing creative jewel suit with high collar and purple blue gemstone wings and glass crown, fashion model style, high resolution, realistic anatomy photography | tomholland-jewelsuit | white young man, instagram"
set -a cases "jiffpom | cute furry {} sitting on flying cloud, 3D render, digital art, ethereal, volumetric lighting, dreamy, pastel colors, illustration for childrens book, hyperrealistic | jiffpom-cloud | pom dog, instagram"
set -a cases "keanureeves | dressed as mario, digital art, hyperdetailed, illustration, trending on artstation, matte painting, CGSociety, pinterest | keanureeves-supermario | white man, instagram"
set -a cases "timotheechalamet | dressed as mario, digital art, hyperdetailed, illustration, trending on artstation, matte painting, CGSociety, pinterest | tomholland-supermario | white young man, instagram"
set -a cases "lilbub | a cute little {} in the shape of a ball, pixar style, 4k, portrait, forest background, cinematic lighting, award winning creature portrait photography | lilbub-ball | tabby cat, instagram"
# Temporary clear the previous cases, so as to only generate the following cases. 
# Undo clearing by commenting out the following line.
set cases
set -a cases "jenniferlawrence | beautiful Gold Knightess redhead hysterically laughing out loud and dancing, visible eye laughter lines, visible smile lines, funny weird facial expression, tightly closed eyes, open gaping mouth, close up face, long flowing hair, photorealistic, wearing intricately designed high chroma tank top, perfect clean defined underarms, chiaroscuro solid colors, divine elegance, perfect teeth, beautiful intricate halo | jenniferlawrence-laugh | girl, instagram"
set -a cases "alita | as a Disney Princess | alita-disney | girl, instagram"
set -a cases "alita | leading an army on a holy war on Mars, clear face | alita-mars | girl, instagram"
set -Ux cases $cases
