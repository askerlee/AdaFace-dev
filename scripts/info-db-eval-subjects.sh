#!/usr/bin/fish
#                   1               2               3                 4               5       6       7       8       9       10                11       12      13      14      15      16      17      18              19              20                      21              22              23              24      25              26              27              28      29      30
set -l subjects     backpack        backpack_dog    bear_plushie      berry_bowl      can     candle  cat     cat2    clock   colorful_sneaker  dog      dog2    dog3    dog5    dog6    dog7    dog8    duck_toy        fancy_boot      grey_sloth_plushie      monster_toy     pink_sunglasses poop_emoji      rc_car  red_cartoon     robot_toy       shiny_sneaker   teapot  vase    wolf_plushie
set -l db_prompts   backpack        backpack        "stuffed animal"  bowl            can     candle  cat     cat     clock   sneaker           dog      dog     dog     dog     dog     dog     dog     toy             boot            "stuffed animal"        toy             glasses         toy             toy     cartoon         toy             sneaker         teapot  vase    "stuffed animal"
set -l ada_prompts  backpack        backpack        "stuffed animal"  bowl            can     candle  cat     cat     clock   sneaker           dog      dog     dog     dog     dog     dog     dog     toy             boot            "stuffed animal"        toy             glasses         toy             toy     cartoon         toy             sneaker         teapot  vase    "stuffed animal"
set -l ada_weights  1               1               "1 2"             1               1       1       1       1       1       1                 1        1       1       1       1       1       1       1               1               "1 2"                   1               1               1               1       1               1               1               1       1       "1 2"

set -Ux subjects        $subjects
set -Ux db_prompts      $db_prompts
set -Ux ada_prompts     $ada_prompts
set -Ux ada_weights     $ada_weights
# No suffix for the DreamBooth eval set, as they are objects/animals, as opposed to faces.
set -Ux db_suffix       ""
set -Ux data_folder     db-eval-dataset
