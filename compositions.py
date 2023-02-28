import exrex
import numpy as np

composition_regexs = \
[ "lift a (rock|box|barbell|cat|dog)",
  "carry a (bag|backpack|cat|dog|teddy bear)",
  "hold a (mobile phone|book|cup of water|piece of paper|sign|cat|dog|teddy bear|baby)",
  "sit (on a table|on a chair|on a bench|on a tank|in a wheelchair)",
  "lean (against a wall|against a tree|against a table|on a chair)",
  "jump on a (table|stair|board|chair|bed|box)",
  "punch a (tree|wall|table|punching bag)",
  "swim (in a pool|underwater|in the ocean)",
  "push a (door|table|car|wheelchair|stroller|shopping cart|bicycle|motorcycle|scooter)",
  "run (in a forest|at the beach|over forest leaves|under the moon)",
  "walk (in a forest|at the beach|over forest leaves|under the moon)",
  "throw (a ball|a rock|water|a book)",
  "catch (a ball|an arrow|a butterfly|a fish|a leaf)",
  "kick a (ball|bottle|tree|rock|punching bag)",
  "play (a card game|a video game|a piano|a violin)",
  "kiss a (boy|girl|baby|lady|cat)",
  "dance with a (boy|girl|lady)",
  "stand (besides a friend|besides a tree|besides a car|in a river|on a table|on a stair|on a board|on a box)",
  "pick up a (rock|leaf|mushroom)",
  "open a (door|window|book|bottle)",
  "point at (the sky|the sun|the beach)",
  "look at (a book|a mobile phone|the sky|the sun|a UFO|the beach)",
  "fly (in the sky|under the sunset|in the outer space|over water|over a building)",
  "wear a (tshirt|stormtrooper costume|superman costume|medal|suit|tie)",
  "drink (a bottle of water|a cup of wine|a can of beer)",
  "eat (a sandwich|an ice cream|barbecue)",
  "on (an airplane|a bus|a busy street|a grass|a roof|an escalator)",
  "in (a car|a meeting|a class|a dress|a suit|a wedding|an elevator|a dinner|Hong Kong|Tokyo|New York)",
  "at (a beach|a table|a park)",
  "besides (a friend|a tree|a car)",
]

def sample_compositions(N):
    compositions = []
    K = len(composition_regexs)
    for i in range(100):
        idx = np.random.choice(K)
        composition = exrex.getone(composition_regexs[idx])
        compositions.append(composition)
    return compositions

if __name__ == '__main__':
    print(sample_compositions(100))
