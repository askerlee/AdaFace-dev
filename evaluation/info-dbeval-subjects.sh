#!/usr/bin/fish
#                     1                 2                    3                 4               5             6               7               8               9                10                11           12              13               14           15                16              17              18              19              20                      21              22              23              24              25                 26              27              28               29                    30
set -g subjects       backpack          backpack_dog         bear_plushie      berry_bowl      can           candle          cat             cat2            clock            colorful_sneaker  dog          dog2            dog3             dog5         dog6              dog7            dog8            duck_toy        fancy_boot      grey_sloth_plushie      monster_toy     pink_sunglasses poop_emoji      rc_car          red_cartoon        robot_toy       shiny_sneaker   teapot            vase                  wolf_plushie
set -g db_prompts    "red backpack"    "cute backpack"      "stuffed toy"     "berry bowl"    can           "glass candle"  cat             cat             "yellow clock"   sneaker           dog          dog             dog              dog          dog               dog             dog             toy             boot            "stuffed toy"           "monster toy"   glasses         toy             "toy car"       "cartoon sprite"  toy              sneaker         "chinese teapot"  "stylish vase"        "stuffed toy"
set -g cls_strings    $db_prompts
set -g ada_prompts    $db_prompts
set -g class_names    backpack          backpack             toy               bowl            can           candle          cat             cat             clock            sneaker           dog          dog             dog              dog          dog               dog             dog             toy             boot            toy                     toy             glasses         toy             toy             sprite             toy             sneaker         teapot            vase                  toy             

set -g ada_weights    "1 2"             "1 2"                "1 2"             "1 2"           1             "1 2"           1               1               "1 2"            1                 1            1               1                1            1                 1               1               1               1               "1 2"                   "1 2"           1               1               "2 1"           "2 1"             1                1               "1 2"             "1 2"                 "1 2"
set -g bg_init_words  "rock tree woman" "window grass table" "ground grass"    "cloth"         "rock ground"  "ground grass" "shade grass"   "floor blanket" "hand cloth"     "ground grass"    "sky tree"   "ground grass"  "beach mountain" "sofa couch"  "orange yellow"  "beach flower"  "grass blanket" "ground beach"  "ground grass"  "ground grass"          "window ground" "cloth"         "ground grass"  "ground floor"  "white"           "ground rock"    "ground grass"  "table plate"     "table window white"  "ground grass"  
set -g broad_classes  0                 0                   0                   0               0               0               1               1               0                0                 1            1               1               1           1                   1               1               0               0               0                       0               0               0               0               2                  0               0               0                 0                      0
# No subjects are human faces. $are_faces instructs the generation script 
# whether to compute face similarity.
set -g are_faces      0                 0                   0                   0               0               0               0               0               0                0                 0            0               0               0           0                   0               0               0               0               0                       0               0               0               0               0                  0               0               0                 0                      0

# sel_set contains a few selected challenging test subjects.
#                     backpack_dog  berry_bowl  can  candle  rc_car robot_toy
set -g sel_set        2             4           5    6       24     26

#                       objects    animals           cartoon characters
set -g lrs              7e-4       1e-3              7e-4
set -g inf_z_prefixes   ""         "portrait of"     ""
set -g maxiters         1500       2000              1500
# Individual LR for each class in the broad classes, according to their difficulties / inherent complexity.
# A prefix of "portrait of" for animals/humans suggests SD to focus on the face area of the subject.

# No suffix for the DreamBooth eval set, as they are objects/animals, as opposed to faces.
set -g db_suffix           ""
set -g data_folder         subjects-dreambench
set -g misc_train_opts     --use_fp_trick 0
set -g misc_infer_opts
set -g resume_from_ckpt     0
# Objects in the same group share the same resumed_ckpt. Number of subjects in each group:
#                           2         3                4                          3             1       8    2    7
set -g resumed_ckpt_keys    backpack  bowl,can,candle  clock,glasses,teapot,vase  boot,sneaker  sprite  toy  cat  dog
#set -g resumed_ckpt_values  resumed_ckpts/backpack_dog2024-01-12T17-34-49_backpack_dog-ada/checkpoints/embeddings_gs-1500.pt