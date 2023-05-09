#!/usr/bin/fish
#                     1               2                 3                 4               5       6               7       8       9                10                11       12      13      14      15      16      17      18              19              20                      21              22              23              24         25                  26              27              28               29              30
set -l subjects       backpack        backpack_dog      bear_plushie      berry_bowl      can     candle          cat     cat2    clock            colorful_sneaker  dog      dog2    dog3    dog5    dog6    dog7    dog8    duck_toy        fancy_boot      grey_sloth_plushie      monster_toy     pink_sunglasses poop_emoji      rc_car     red_cartoon        robot_toy       shiny_sneaker   teapot            vase            wolf_plushie
set -l db_prompts     "red backpack"  "cute backpack"   "stuffed animal"  "berry bowl"    can     "glass candle"  cat     cat     "yellow clock"   sneaker           dog      dog     dog     dog     dog     dog     dog     toy             boot            "stuffed animal"        "monster toy"   glasses         toy             "toy car"  "cartoon monster"  toy             sneaker         "chinese teapot"  "stylish vase"  "stuffed animal"
set -l cls_tokens     backpack        backpack          toy               bowl            can     candle          cat     cat     clock            sneaker           dog      dog     dog     dog     dog     dog     dog     toy             boot            toy                     toy             glasses         toy             toy        cartoon            toy             sneaker         teapot            vase            toy             
# ada_prompts only differ from db_prompts in "stuffed toy" vs "stuffed animal". 
# Because individual words, rather than the whole phrase in ada_prompts impact the final embeddings. 
# "stuffed animal" is not an animal, so "stuffed toy" better suites AdaPrompt.
# But a prompt phrase like "stuffed animal" can be understood as a whole phrase by SD/DreamBooth.
set -l ada_prompts    "red backpack"  "cute backpack"   "stuffed toy"     "berry bowl"    can     "glass candle"  cat     cat     "yellow clock"   sneaker           dog      dog     dog     dog     dog     dog     dog     toy             boot            "stuffed toy"           "monster toy"   glasses         toy             "toy car"  "cartoon monster"  toy             sneaker         "chinese teapot"  "stylish vase"  "stuffed toy"
set -l ada_weights    "1 2"           "1 2"             "1 2"             "1 2"           1       "1 2"           1       1       "1 2"            1                 1        1       1       1       1       1       1       1               1               "1 2"                   "1 2"           1               1               "2 1"      "2 1"              1               1               "1 2"             "1 2"           "1 2"
set -l broad_classes  0               0                 0                 0               0       0               1       1       0                0                 1        1       1       1       1       1       1       0               0               0                       0               0               0               0          2                  0               0               0                 0               0
# No subjects are human faces. $are_faces instructs the generation script 
# whether to compute face similarity.
set -l are_faces      0               0                 0                 0               0       0               0       0       0                0                 0        0       0       0       0       0       0       0               0               0                       0               0               0               0          0                  0               0               0                 0               0

# sel_set contains a few selected challenging test subjects.
#                     backpack_dog  berry_bowl  can  candle  rc_car robot_toy
set -l sel_set        2             4           5    6       24     26

#                     objects    animals           cartoon characters
set -l lrs            3e-4       6e-4              3e-4
set -l z_prefixes     ""         "portrait of"     ""
set -l maxiters       3500       4500              3500
# Individual LR for each class in the broad classes, according to their difficulties / inherent complexity.
# A prefix of "portrait of" for animals/humans suggests SD to focus on the face area of the subject.

set -Ux subjects            $subjects
set -Ux db_prompts          $db_prompts
set -Ux ada_prompts         $ada_prompts
set -Ux ada_weights         $ada_weights
set -Ux cls_tokens          $cls_tokens
set -Ux broad_classes       $broad_classes
set -Ux are_faces           $are_faces
set -Ux sel_set             $sel_set
set -Ux lrs                 $lrs
# No suffix for the DreamBooth eval set, as they are objects/animals, as opposed to faces.
set -Ux db_suffix           ""
set -Ux z_prefixes          $z_prefixes
set -Ux data_folder         dbeval-dataset
