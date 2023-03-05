import exrex
import numpy as np

composition_regexs = \
[ "lifting a (rock|box|barbell|cat|dog)",
  "doing (makeup|housekeeping|gardening|exercise)",
  "carrying a (bag|backpack|luggage|laptop|book|briefcase|purse|suitcase|bouquet|baby|cat|dog|teddy bear)",
  "holding a (mobile phone|book|cup of water|piece of paper|flower|bouquet|pen|sign|cat|dog|teddy bear|baby)",
  "sitting (on a table|on a chair|on a bench|on a tank|in a wheelchair|on the ground)",
  "leaning (against a wall|against a tree|against a table|on a chair|on a car)",
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
  "kissing a (boy|girl|baby|lady|man|cat)",
  "dancing with a (boy|girl|lady|man|villager)",
  "standing (besides a friend|besides a tree|besides a car|in a river|on a table|on a stair|on a board|on a box)",
  "picking up a (rock|leaf|mushroom|stick|flower|shell|pebble|twig|fruit)",
  "opening a (door|window|book|bottle|jar|box|envelope|bag|pouch|wallet|suitcase)",
  "pointing at (the sky|the sun|the beach|the mountains|the forest)",
  "looking at (a book|a mobile phone|the screen|the sky|the sun|the beach|a UFO|a map|a painting|a photo|a clock|a mirror|a telescope|a microscope)",
  "flying (in the sky|under the sunset|in the outer space|over water|over a building)",
  "wearing a (tshirt|stormtrooper costume|superman costume|ironman armor|ski outfit|astronaut outfit|medal|suit|tie|baseball cap)",
  "drinking (a bottle of water|a cup of wine|a can of beer|a glass of juice|a cup of tea|a bottle of milk)",
  "eating (a sandwich|an ice cream|barbecue|a pizza|a burger|a bowl of pasta|a piece of cake|a sushi roll|a bowl of soup|a plate of tacos)",
  "on (an airplane|a bus|a busy street|a grass|a roof|an escalator|a train|a boat|a bike|a roller coaster|a ski lift|a hot air balloon|a scooter)",
  "in (a car|a meeting|a class|a dress|a suit|a tshirt|a stormtrooper costume|a superman costume|a wedding|an elevator|a dinner|a concert|a gym|a library|a park|a mall|a movie theater|a hotel room|Hong Kong|Tokyo|New York)",
  "at (a beach|a table|a park|a concert|a gym|a library|a mall|a movie theater|a hotel room|a theme park)",
  "next to (a friend|a tree|a car|a river|a lake|a mountain|an ocean|a playground|a statue|a panda)",
  "made of (metal|stainless steel|fractal flame|marble|rubber|bronze|ice)",
]

all_styles = [ "cartoon", "animation", "anime", "comic book", "steampunk", "oil on canvas", "oil painting",
               "sci-fi movie", "scuplture", "bronze sculpture", "abyss art", "blade runner", "cyberpunk",
               "synthwave", "pencil sketch", 
             ]
# concept art|realistic painting|character design|anime sketch|trending in artstation|hyper realistic|vivid colors|clear face|detailed face
all_modifiers = [ "concept art", "realistic painting", "character design", "anime sketch", 
                 "trending in artstation", "hyper realistic", "vivid colors", "clear face", 
                 "detailed face", "semirealism", "hyperrealistic", "highly detailed", "octane render",
                 "unreal 5", "photorealistic", "sharp focus", "digital painting", "illustration"  
                ]

all_art_by = [ "miho hirano", "makoto shinkai", "artgerm",  "greg rutkowski", "magali villeneuve" ]

def sample_compositions(N):
    compositions = []
    K = len(composition_regexs)
    for i in range(N):
        idx = np.random.choice(K)
        composition = exrex.getone(composition_regexs[idx])

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

        compositions.append(f"{composition}{style}{modifier}{art_by}")

    return compositions

if __name__ == '__main__':
    print('\n'.join(sample_compositions(100)))
