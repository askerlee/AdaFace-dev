#!/usr/bin/fish
#                     1               2                 3                  4                                5                   6                    7                  8                       9 
set -l subjects       cat_statue      clock             colorful_teapot    elephant                         mug_skulls          physics_mug          red_teapot         round_bird              thin_bird
set -l db_prompts     "cat figurine"  "alarm clock"     "colorful teapot"  "openwork elephant figurine"     "face skull mug"    "brown formula mug"  "painted teapot"   "rustic bird figurine"  "abstract bird figurine"       
set -l cls_tokens     figurine        clock             teapot             figurine                         mug                 mug                  teapot             figurine                figurine
# ada_prompts only differ from db_prompts in "stuffed toy" vs "stuffed animal". 
# Because individual words, rather than the whole phrase in ada_prompts impact the final embeddings. 
# "stuffed animal" is not an animal, so "stuffed toy" better suites AdaPrompt.
# But a prompt phrase like "stuffed animal" can be understood as a whole phrase by SD/DreamBooth.
set -l ada_prompts    $db_prompts
set -l ada_weights    "1 2"           "1 2"             "1 2"               "1 1 2"                          "1 1 2"            "1 1 2"              "1 2"              "1 1 2"                 "1 1 2" 
set -l broad_classes  0               0                 0                   0                                 0                  0                    0                 0                       0
# No subjects are human faces. $are_faces instructs the generation script 
# whether to compute face similarity.
set -l are_faces      0               0                 0                   0                                 0                  0                    0                 0                       0

# sel_set contains a few selected challenging test subjects.
#                     cat_statue      mug_skulls
set -l sel_set        1               5

#                     objects    animals           cartoon characters
set -l lrs            3e-4       6e-4              3e-4
set -l z_prefixes     ""         "portrait of"     ""
set -l maxiters       3500       4000              3500
# Individual LR for each class in the broad classes, according to their difficulties / inherent complexity.
# A prefix of "portrait of" for animals/humans suggests SD to focus on the face area of the subject.

set -g subjects            $subjects
set -g db_prompts          $db_prompts
set -g ada_prompts         $ada_prompts
set -g ada_weights         $ada_weights
set -g cls_tokens          $cls_tokens
set -g broad_classes       $broad_classes
set -g are_faces           $are_faces
set -g sel_set             $sel_set
set -g lrs                 $lrs
# No suffix for the DreamBooth eval set, as they are objects/animals, as opposed to faces.
set -g db_suffix           ""
set -g z_prefixes          $z_prefixes
set -g maxiters            $maxiters
set -g data_folder         ti-dataset
