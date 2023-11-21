import exrex
import numpy as np

# animal compositions are used by humans/animals only
animal_action_regexs = \
[ "lifting a (rock|box|barbell|cat|dog)",
  "doing (makeup|housekeeping|gardening|exercise)",
  "carrying a (bag|backpack|luggage|laptop|book|briefcase|purse|suitcase|bouquet|baby|cat|dog|teddy bear)",
  "holding a (mobile phone|book|cup of water|piece of paper|flower|bouquet|pen|sign|cat|dog|teddy bear|baby|rock|leaf|mushroom|stick|fruit)",
  "(sitting|sleeping) (on a table|on a chair|on a bench|on a tank|in a wheelchair|on the ground|on flying cloud)",
  "swimming (in a pool|underwater|in the ocean|in a lake|in a river)( among tropical fishes)?",
  "pushing a (door|table|car|wheelchair|stroller|shopping cart|bicycle|motorcycle|scooter)",
  "running (in a forest|at the beach|over forest leaves|on a trail|under the moon|on a treadmill)",
  "walking (in a forest|at the beach|over forest leaves|on a trail|under the moon|on a treadmill)",
  "throwing (a ball|a rock|water|a dart|a frisbee|a knife|a javelin)",
  "catching (a ball|an arrow|a butterfly|a fish|a leaf)",
  "kicking a (ball|bottle|tree|rock|punching bag|pole|box)",
  "playing (a card game|a video game|a piano|a violin|basketball|tennis)",
  "riding a (bike|motorcycle|scooter|horse|car|bus|train|boat)",
  "(kissing|hugging|holding) a (boy|girl|baby|lady|man|cat)",
  "dancing with a (boy|girl|lady|man|villager)",
  "standing (besides a friend|besides a tree|besides a car|in a river|on a table|on a stair|on a board|on a box)",
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
  "on (a boat|a bike|a roller coaster|a ski lift|a hot air balloon|a scooter)",
  "in (a car|a meeting|a class|a wedding|a dinner|a concert|a gym|a library|a park)",
  "in (a mall|a movie theater|a hotel room|Hong Kong|Tokyo|New York)",
  "at (a beach|a table|a park|a concert|a gym|a library|a mall|a movie theater|a hotel room|a theme park)",
  "next to (a friend|a tree|a car|a river|a lake|a mountain|an ocean|a playground|a statue|a panda)",
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

# Prompt with locations will be combined with a common animal/human.
# E.g. "a z at the left, a dog in the center"
all_locations = [ "at the left", "at the right", "at the top", "at the bottom", 
                  "in the center", "in the middle", "at the upper left", "at the upper right",
                  "at the lower left", "at the lower right", "in the background", 
                ]

coexist_objects = [ "person", "man",  "woman",   "girl",    "boy",   "baby",       "crowd", "villager", 
                     "cat",   "dog",  "bird",    "panda",  "monkey", "chimpanzee", "gorilla", "bear",  
                     "horse", "sheep", "elephant", "lion"
                     # No need to include non-animals below. They tend not to mix features with subjects.
                     # "stone", "tree",  "flower", "rock", "mountain", "grass",     "cloud", "sun",
                     # "moon",  "stars", "fire",   "lake", "ocean",    "river",     "beach", "village",
                     # "house", "car",   "bus",    "train", "boat",    "bike",      "building", "tower" 
                  ] 

# added "style/art" behind some prompt
all_styles = [ "cartoon style", "animation", "anime art", "comic book art", "steampunk art", "oil on canvas", "oil painting",
               "sci-fi movie", "sculpture", "bronze sculpture", "abyss art", "blade runner style", "cyberpunk art",
               "synthwave", "pencil sketch", "pastel colors", "childrens book's illustration", "pixar movie",
               "as a crochet figure", "as a 3d model", "closeup shot", "close view", "D&D sci-fi",
               "pop art", "portrait art", "watercolour painting", "chalk art", "concepture art", "bauhaus style", 
               "photorealistic painting", "surrealism painting", "impressionism", "expressionism", "abstract art", "minimalism",
               "low poly", "cubism style",
             ]
# concept art|realistic painting|character design|anime sketch|trending in artstation|hyper realistic|vivid colors|clear face|detailed face
all_modifiers = [ "concept art", "realistic painting", "character design", "anime sketch", 
                  "trending in artstation", "hyper realistic", "vivid colors", "clear face", 
                  "detailed face", "semirealism", "hyperrealistic", "highly detailed", "octane render",
                  "unreal 5", "photorealistic", "sharp focus", "digital painting", "illustration",
                  "volumetric lighting", "dreamy", "cinematic", 
                  "surreal", "hd", "4k", "8k", "3d", "4d", "pixelate", "blur", 
                  "beautiful", "very beautiful", "symmetrical", "macabre", "at night" 
                ]

#add time prompts
all_time = [ "futuristic", "modern", "ancient", "antique","retro","old-fashioned", "youthful" ]

#add light prompts
all_light = [ "daylight", "moonlight", "night sky", "natural light", "front light", 
              "backlight", "soft light", "hard light", "moody light", "dramatic light", 
              "dynamic light", "natural light" ]


all_art_by = [ "miho hirano", "makoto shinkai", "artgerm",  "greg rutkowski", "magali villeneuve",
               "mark ryden", "hayao miyazaki", 
               #add artist 
               "agnes Lawrence","disney animation studio"]

#add background prompts
all_backgrounds = [ "a beach", "a table", "a park", "a concert", "a gym", "a library", "a mall", "a movie theater", "a hotel room", "a theme park",
                    "a city", "a mountain", "a blue house", "a wheat field", "a tree and autumn leaves", "the Eiffel Tower", "a jungle", "the snow",
                    "a cobblestone street", "underwater", "an ocean of milk", "pink fabric", "a wooden floor", "green grass with sunflowers around it",
                    "a mirror", "the sidewalk in a crowded street", "a dirt road", "a white rug", "a purple rug in a forest", "a red cube", "a purple cube",
                    "a building", 
                  ]

Debug_Prompts = False #True
PRESET_DEBUG_PROMPTS = [ 'on a cobblestone street', 'on top of a wooden floor' ]

def sample_compositions(N, subj_type, is_training=False):
    compositions = []

    if Debug_Prompts:
        K = len(PRESET_DEBUG_PROMPTS)
        for i in range(N):
            idx = np.random.choice(K)
            compositions.append(PRESET_DEBUG_PROMPTS[idx])
        return compositions
    
    if subj_type == 'animal':
        composition_regexs = all_composition_regexs
    elif subj_type == 'object':
        composition_regexs = static_composition_regexs
    else:
        raise ValueError(f"Unknown subject type: {subj_type}")
    
    K = len(composition_regexs)

    if is_training:
        # Lower variations during training, to focus on the main semantics.
        # 0.75: option 0 (without certain components), 
        # 0.25: option 1 (with    certain components).
        option_probs     = [0.75, 0.25]
        # 0.4:  option 0 (without background),
        # 0.6:  option 1 (with    background).
        background_probs = [0.4,  0.6]
    else:
        option_probs = [0.3, 0.7]
        background_probs = option_probs

    for i in range(N):
        idx = np.random.choice(K)

        composition = exrex.getone(composition_regexs[idx])
        # Disable another object in the image for non-animal subjects,
        # to avoid the spotlight of the non-animal subject being stolen by the other object.
        if subj_type == 'animal':
            has_another_obj = np.random.choice([0, 1], p=[0.9, 0.1])
        else:
            has_another_obj = False

        if has_another_obj:
            object2   = np.random.choice(coexist_objects)
            location2 = np.random.choice(all_locations)
            obj_loc2  = ", a " + object2 + " " + location2
        else:
            obj_loc2  = ""

        # Choose a few common styles
        has_styles = np.random.choice([0, 1], p=option_probs)
        if has_styles:
            num_styles = np.random.choice([1, 2])
            styles = np.random.choice(all_styles, size=num_styles, replace=False)
            # style = np.random.choice(all_styles) + ' '
            style = ", in " + " and ".join(styles) + " style"
        else:
            style = ""

        has_modifiers = np.random.choice([0, 1], p=option_probs)
        if has_modifiers:
            num_modifiers = np.random.choice([1, 2, 3])
            modifiers = np.random.choice(all_modifiers, size=num_modifiers, replace=False)
            modifier = ", " + ", ".join(modifiers)          
        else:
            modifier = ""

        has_art_by = np.random.choice([0, 1], p=option_probs)
    
        if has_art_by:
            num_art_by = np.random.choice([1, 2, 3])
            art_bys = np.random.choice(all_art_by, size=num_art_by, replace=False)
            art_by = ", art by " + " and ".join(art_bys)
        else:
            art_by = ""

        has_background = np.random.choice([0, 1], p=background_probs)
        if has_background:
            background = np.random.choice(all_backgrounds)
            background = ", with " + background + " as background"
        else:
            background = ""

        has_time_theme = np.random.choice([0, 1], p=option_probs)
        if has_time_theme:
            time = np.random.choice(all_time) 
            time = ", " + time
        else:
            time = ""
        
        has_light = np.random.choice([0, 1], p=option_probs)
        has_light =1
        if has_light:
            light = np.random.choice(all_light)
            light = ", with " + light 
        else:
            light = ""

        if is_training:
            composition = f"{composition}{modifier}{time}{style}{background}{art_by}{light}{obj_loc2}"
        else:
            image = ", " + np.random.choice(['photo', 'drawing', 'illustration', 'picture'])
            composition = f"{modifier}{time}{style}{image} of z {composition}{background}{art_by}{light}{obj_loc2}"
            if composition.startswith(", "):
                composition = composition[2:]
        
        compositions.append(composition)
        
    return compositions

if __name__ == "__main__":
    print("Test:")
    print("\n".join(sample_compositions(20, 'animal')))
    print()
    print("Training:")
    print("\n".join(sample_compositions(20, 'animal', is_training=True)))
