import exrex
import numpy as np

# dynamic compositions are used by humans/animals only
dynamic_composition_regexs = \
[ "lifting a (rock|box|barbell|cat|dog)",
  "doing (makeup|housekeeping|gardening|exercise)",
  "carrying a (bag|backpack|luggage|laptop|book|briefcase|purse|suitcase|bouquet|baby|cat|dog|teddy bear)",
  "holding a (mobile phone|book|cup of water|piece of paper|flower|bouquet|pen|sign|cat|dog|teddy bear|baby)",
  "(sitting|sleeping) (on a table|on a chair|on a bench|on a tank|in a wheelchair|on the ground|on flying cloud)",
  "jumping on a (table|stair|board|chair|bed|box|trampoline)",
  "punching a (tree|wall|table|punching bag)",
  "swimming (in a pool|underwater|in the ocean|in a lake|in a river)( among tropical fishes)?",
  "pushing a (door|table|car|wheelchair|stroller|shopping cart|bicycle|motorcycle|scooter)",
  "running (in a forest|at the beach|over forest leaves|on a trail|under the moon|on a treadmill)",
  "walking (in a forest|at the beach|over forest leaves|on a trail|under the moon|on a treadmill)",
  "throwing (a ball|a rock|water|a book|a bottle|a cat|a dart|a frisbee|a grenade|a knife|a javelin)",
  "catching (a ball|an arrow|a butterfly|a fish|a leaf|a cat|a rabbit|a thief|a prey)",
  "kicking a (ball|bottle|tree|rock|punching bag|telephone booth)",
  "playing (a card game|a video game|a piano|a violin|basketball|tennis)",
  "riding a (bike|motorcycle|scooter|horse|car|bus|train|boat)",
  "(kissing|hugging|holding) a (boy|girl|baby|lady|man|cat)",
  "dancing with a (boy|girl|lady|man|villager)",
  "standing (besides a friend|besides a tree|besides a car|in a river|on a table|on a stair|on a board|on a box)",
  "picking up a (rock|leaf|mushroom|stick|flower|shell|pebble|twig|fruit)",
  "opening a (door|window|book|bottle|jar|box|envelope|bag|pouch|wallet|suitcase)",
  "pointing at (the sky|the sun|the beach|the mountains|the forest)",
  "looking at (a book|a mobile phone|the screen|the sky|the sun|the beach|a UFO|a map|a painting|a photo|a clock|a mirror|a telescope|a microscope)",
  "wearing a (tshirt|stormtrooper costume|superman costume|ironman armor|ski outfit|astronaut outfit|medal|suit|tie|baseball cap)",
  "drinking (a bottle of water|a cup of wine|a can of beer|a glass of juice|a cup of tea|a bottle of milk)",
  "eating (a sandwich|an ice cream|barbecue|a pizza|a burger|a bowl of pasta|a piece of cake|a sushi roll|a bowl of soup|a plate of tacos)",
  "wearing (a red hat|a santa hat|a rainbow scarf|a black top hat and a monocle|pink glasses|a yellow shirt|aikido training clothes|green robe)",
  # This is kind of static but only for humans/animals. So we put it here.
  "in a (chef outfit|firefighter outfit|police outfit|a purple wizard outfit|dress|suit|tshirt|stormtrooper costume|superman costume)",
]

# static compositions are used by both humans/animals and objects
static_composition_regexs = \
[ 
  "leaning (against a wall|against a tree|against a table|on a chair|on top of a car)",
  "flying (in the sky|under the sunset|in the outer space|over water|over a building)",
  "on (an airplane|a bus|a busy street|a grass|a roof|an escalator|a train|a boat|a bike|a roller coaster|a ski lift|a hot air balloon|a scooter)",
  "in (a car|a meeting|a class|a wedding|an elevator|a dinner|a concert|a gym|a library|a park|a mall|a movie theater|a hotel room|Hong Kong|Tokyo|New York)",
  "at (a beach|a table|a park|a concert|a gym|a library|a mall|a movie theater|a hotel room|a theme park)",
  "next to (a friend|a tree|a car|a river|a lake|a mountain|an ocean|a playground|a statue|a panda)",
  "made of (metal|stainless steel|fractal flame|marble|rubber|bronze|ice)",
  # Prompts below are from DreamBooth evaluation dataset
  #LINK - https://github.com/google/dreambooth/blob/main/dataset/prompts_and_classes.txt
  "(in the jungle|in the snow|on a cobblestone street|floating on top of water|floating in an ocean of milk)",
  "with a (city|mountain|blue house|wheat field|a tree and autumn leaves|Eiffel Tower) in the background",
  "on top of (pink fabric|a wooden floor|green grass with sunflowers around it|a mirror|the sidewalk in a crowded street|a dirt road|a white rug|a purple rug in a forest)",
  # To avoid misalignment issues, we don't use "a red/purple z" as prompts.
  "that is (red|purple|shiny|cube|wet)",
]

all_composition_regexs = static_composition_regexs + dynamic_composition_regexs

# Prompt with locations will be combined with a common animal/human.
# E.g. "a z at the left, a dog in the center"
all_locations = [ "at the left", "at the right", "at the top", "at the bottom", 
                  "in the center", "in the middle", "at the upper left", "at the upper right",
                  "at the lower left", "at the lower right", "in the background", "in the foreground",
                  ]

coexist_objects = [ "person", "man",  "woman",   "girl",    "boy",   "baby",       "crowd", "villager", 
                     "cat",   "dog",  "bird",    "panda",  "monkey", "chimpanzee", "gorilla", "bear",  
                     "horse", "sheep", "elephant", "lion",
                     # No need to include non-animals below. They tend not to mix features with subjects.
                     # "stone", "tree",  "flower", "rock", "mountain", "grass",     "cloud", "sun",
                     # "moon",  "stars", "fire",   "lake", "ocean",    "river",     "beach", "village",
                     # "house", "car",   "bus",    "train", "boat",    "bike",      "building", "tower" 
                  ] 

all_styles = [ "cartoon", "animation", "anime", "comic book", "steampunk", "oil on canvas", "oil painting",
               "sci-fi movie", "scuplture", "bronze sculpture", "abyss art", "blade runner", "cyberpunk",
               "synthwave", "pencil sketch", "pastel colors", "illustration for childrens book", "pixar movie",
               "as a crochet figure", "as a 3d model", "closeup shot", "close view" 
             ]
# concept art|realistic painting|character design|anime sketch|trending in artstation|hyper realistic|vivid colors|clear face|detailed face
all_modifiers = [ "concept art", "realistic painting", "character design", "anime sketch", 
                  "trending in artstation", "hyper realistic", "vivid colors", "clear face", 
                  "detailed face", "semirealism", "hyperrealistic", "highly detailed", "octane render",
                  "unreal 5", "photorealistic", "sharp focus", "digital painting", "illustration",
                  "volumetric lighting", "dreamy", "illustration", "cinematic"
                ]

all_art_by = [ "miho hirano", "makoto shinkai", "artgerm",  "greg rutkowski", "magali villeneuve",
               "mark ryden", "hayao miyazaki" ]

def sample_compositions(N, is_animal):
    compositions = []
    if is_animal:
        composition_regexs = all_composition_regexs
    else:
        composition_regexs = static_composition_regexs
        
    K = len(composition_regexs)
    for i in range(N):
        idx = np.random.choice(K)
        composition = exrex.getone(composition_regexs[idx])
        has_location = np.random.choice([0, 1])
        if has_location:
            loc1, loc2 = np.random.choice(all_locations, 2, replace=False)
            location1 = loc1 + " "
            object2   = np.random.choice(coexist_objects)
            obj_loc2  = ", a " + object2 + " " + loc2
        else:
            location1 = ""
            obj_loc2  = ""

        has_styles = np.random.choice([0, 1])
        if has_styles:
            num_styles = np.random.choice([1, 2, 3])
            styles = [ np.random.choice(all_styles) for i in range(num_styles) ]
            style = ", in " + " and ".join(styles) + " style"
        else:
            style = ""

        has_modifiers = np.random.choice([0, 1])
        if has_modifiers:
            num_modifiers = np.random.choice([1, 2, 3])
            modifiers = [ np.random.choice(all_modifiers) for i in range(num_modifiers) ]
            modifier = ", " + ", ".join(modifiers)
        else:
            modifier = ""

        has_art_by = np.random.choice([0, 1])
        if has_art_by:
            num_art_by = np.random.choice([1, 2])
            art_bys = [ np.random.choice(all_art_by) for i in range(num_art_by) ]
            art_by = ", art by " + " and ".join(art_bys)
        else:
            art_by = ""

        compositions.append(f"{location1}{composition}{style}{modifier}{art_by}{obj_loc2}")

    return compositions

if __name__ == "__main__":
    print("\n".join(sample_compositions(100, True)))
