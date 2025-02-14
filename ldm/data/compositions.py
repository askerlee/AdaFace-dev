import exrex
import numpy as np

# animal compositions are used by humans or animals, not for objects
animal_action_regexs = \
[ "doing (makeup|housekeeping|gardening|exercise)",
  "carrying a (bag|backpack|luggage|laptop|book|briefcase|purse|suitcase|bouquet|baby|cat|dog|teddy bear)",
  "holding a (mobile phone|book|cup of water|piece of paper|flower|bouquet|pen|sign|cat|dog|teddy bear|baby|rock|leaf|mushroom|stick|fruit)",
  "(sitting|sleeping) (on a table|on a chair|on a bench|on a tank|in a wheelchair|on the ground|on flying cloud)",
  "swimming (in a pool|underwater|in the ocean|in a lake|in a river)( among tropical fishes)",
  "pushing a (door|table|car|wheelchair|stroller|shopping cart|bicycle|motorcycle|scooter)",
  "running (in a forest|in a park|at the beach|over forest leaves|on a trail|under the moon|on a treadmill)",
  "walking (in a forest|in a park|at the beach|over forest leaves|on a trail|under the moon|on a treadmill)",
  "throwing (a ball|a rock|water|a dart|a frisbee|a knife|a javelin|a tennis ball)",
  "catching (a ball|an arrow|a butterfly|a fish|a leaf)",
  "kicking a (ball|bottle|tree|rock|punching bag|pole|box)",
  "playing (a card game|a video game|a piano|a violin|basketball|tennis)",
  "riding a (bike|motorcycle|scooter|horse|car|bus|train|boat)",
  "(kissing|hugging|holding) a (baby|cat|dog)",
  "standing (besides a tree|besides a car|in a river|on a table|on a stair|on a board|on a box)",
  "opening a (door|window|book|bottle|jar|box|envelope|bag|pouch|wallet|suitcase)",
  "pointing at (the sky|the sun|the beach|the mountains|the forest)",
  "looking at (a book|a mobile phone|the screen|the sky|the sun|the beach|a UFO|a painting|a clock|a mirror)",
  "drinking (a bottle of water|a cup of wine|beer|milk|a glass of juice|a cup of tea)",
  "eating (a sandwich|an ice cream|a pizza|a burger|pasta|cake|sushi|soup|tacos)",
]

animal_dresses = [
  "wearing a (tshirt|stormtrooper costume|superman costume|ironman armor|ski outfit|astronaut outfit|suit|baseball cap)",
  "wearing (a red hat|a santa hat|a rainbow scarf|a black top hat and a monocle|pink glasses|a yellow shirt|aikido uniform|green robe)",
  # This is kind of static but only for humans/animals. So we put it here.
  "in a (chef outfit|firefighter outfit|police outfit|a purple wizard outfit|dress|suit|stormtrooper costume|superman costume)",
]

# static compositions are used by both humans/animals and objects
static_action_regexs = \
[ 
  "leaning (against a wall|against a tree|against a table|on a chair|on top of a car)",
  "flying (in the sky|under the sunset|in the outer space|over water|over a building)",
  # Split a regex with too many candidate patterns into two lines, 
  # to avoid under-representation of the patterns, as the regexs are unifomly sampled.
  "on (an airplane|a bus|a busy street|a grass|a roof|an escalator|a train)",
  "on (a boat|a bike|a roller coaster|a scooter)",
  "in (a car|a meeting|a class|a wedding|a dinner|a concert|a gym|a library|a park)",
  "in (a mall|a movie theater|a hotel room|Hong Kong|Tokyo|New York)",
  "at (a beach|a table|a park|a concert|a gym|a library|a mall|a movie theater|a hotel room|a theme park)",
  "next to (a tree|a car|a river|a lake|a mountain|an ocean|a playground|a statue|a panda)",
  "made of (metal|stainless steel|fractal flame|marble|rubber|bronze|ice)",
  # Prompts below are from DreamBooth evaluation dataset
  #LINK - https://github.com/google/dreambooth/blob/main/dataset/prompts_and_classes.txt
  "(in the jungle|in the snow|on a cobblestone street|floating on top of water|floating in an ocean of milk)",
  "on top of (pink fabric|a wooden floor|green grass with sunflowers around it|a mirror|the sidewalk in a crowded street|a dirt road|a white rug|a purple rug in a forest)",
]

static_appearances = [
  # To avoid misalignment issues, we don't use "a red/purple z" as prompts.
  "that is (red|purple|shiny|cube|wet)",
]

all_action_regexs = static_action_regexs + animal_action_regexs
all_dress_regexs  = static_appearances   + animal_dresses
all_composition_regexs    = all_action_regexs    + all_dress_regexs
static_composition_regexs = static_action_regexs + static_appearances

# Removed "eye level shot", "close-up view", "zoomed-in view"; sometimes it generates a huge eye.
all_shots = [ "side view", "zoomed-out view", "full body view", "middle shot", "long shot", "wide shot" ]

# added "style/art" behind some prompt
all_styles = [ "cartoon style", "animation", "anime art", "comic book art", "steampunk art", "oil on canvas", "oil painting",
               "sci-fi movie", "sculpture", "bronze sculpture", "abyss art", "blade runner style", "cyberpunk art",
               "synthwave", "pencil sketch", "pastel colors", "childrens book's illustration", "pixar movie",
               "as a crochet figure", "as a 3d model", "D&D sci-fi",
               "pop art", "portrait art", "watercolour painting", "chalk art", "concepture art", "bauhaus style", 
               "photorealistic painting", "surrealism painting", "impressionism", "expressionism", "abstract art", "minimalism",
               "low poly", "cubism style", "funko pop",
               "concept art", "realistic painting", "character design", "anime sketch", 
               "trending in artstation", "vivid colors", "semirealism", "octane render",
               "unreal 5", "digital painting", "illustration",  "volumetric lighting", "dreamy", 
               "cinematic", "surreal", "pixelate", "macabre"
            ]

#add time prompts
all_time = [ "futuristic", "modern", "ancient", "antique", "retro", "old-fashioned", "youthful" ]

#add light prompts
all_light = [ "daylight", "moonlight", "night sky", "natural light", "front light", 
              "backlight", "soft light", "hard light", "moody light", "dramatic light", 
              "dynamic light", "natural light", "at night", "neon light" ]

all_art_by = [ "miho hirano", "makoto shinkai", "artgerm",  "greg rutkowski", "magali villeneuve",
               "mark ryden", "hayao miyazaki", "agnes Lawrence", "disney animation studio"]

#add background prompts
all_backgrounds = [ "a beach", "a table", "a park", "a concert", "a gym", "a library", "a mall", "a movie theater", "a hotel room", 
                    "a theme park", "a city", "a mountain", "a blue house", "a wheat field", "a tree and autumn leaves", 
                    "the Eiffel Tower", "a jungle",  "underwater", "a red cube", "a purple cube", "a building", 
                    "night view of the tokyo street"
                  ]

def sample_compositions(N, subj_type):
    compos_prompts = []
    modifiers = []

    if subj_type == 'animal':
        composition_regexs = all_composition_regexs
    elif subj_type == 'object':
        composition_regexs = static_composition_regexs
    else:
        raise ValueError(f"Unknown subject type: {subj_type}")
    
    K = len(composition_regexs)

    # Lower variations during training, to focus on the main semantics.
    # 0.75: option 0 (without certain components), 
    # 0.25: option 1 (with    certain components).
    option_probs     = [0.75, 0.25]
    shot_probs       = [0.3,  0.7]
    # 0.4:  option 0 (without background),
    # 0.6:  option 1 (with    background).
    background_probs = [0.4,  0.6]

    for i in range(N):
        idx = np.random.choice(K)

        composition = exrex.getone(composition_regexs[idx])

        style_probs = [0.3, 0.2, 0.5]
        has_styles = np.random.choice([0, 1, 2], p=style_probs)
        if has_styles == 2:     # 50% with 1 or 2 styles
            num_styles = np.random.choice([1, 2])
            styles = np.random.choice(all_styles, size=num_styles, replace=False)
            # style = np.random.choice(all_styles) + ' '
            style = "in " + " and ".join(styles) + " style"
        elif has_styles == 1:   # 20% with photorealistic as the style
            style = "photorealistic"
        elif has_styles == 0:   # 30% without style
            style = ""

        has_shot = np.random.choice([0, 1], p=shot_probs)
        if has_shot:   # p = 0.7 has a random shot
            shot = np.random.choice(all_shots)
        else:           # p = 0.3 has no shot
            shot = ""

        has_art_by = np.random.choice([0, 1], p=option_probs)
    
        if has_art_by:
            num_art_by = np.random.choice([1, 2, 3])
            art_bys = np.random.choice(all_art_by, size=num_art_by, replace=False)
            art_by = "art by " + " and ".join(art_bys)
        else:
            art_by = ""

        has_background = np.random.choice([0, 1], p=background_probs)
        if has_background:
            background = np.random.choice(all_backgrounds)
            background = "with " + background + " as background"
        else:
            background = ""

        has_time_theme = np.random.choice([0, 1], p=option_probs)
        if has_time_theme:
            time = np.random.choice(all_time) 
        else:
            time = ""
        
        has_light = np.random.choice([0, 1], p=option_probs)
        has_light =1
        if has_light:
            light = np.random.choice(all_light)
            light = "with " + light 
        else:
            light = ""

        modifier = ", ".join(filter(lambda s: len(s) > 0, [time, style, shot, light, art_by]))
        #compos_prompt = f"{composition}{obj_loc2}{background}"
        compos_prompt = ", ".join(filter(lambda s: len(s) > 0, [composition, background]))
        modifiers.append(modifier)
        compos_prompts.append(compos_prompt)
    return compos_prompts, modifiers

if __name__ == "__main__":
    compos_prompts, modifiers = sample_compositions(20, 'animal')
    for i in range(20):
        print(f"{i+1}:\t{modifiers[i]}")
        print(f"    \t{compos_prompts[i]}")
        print()

