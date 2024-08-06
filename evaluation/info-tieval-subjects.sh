#!/usr/bin/fish
#                               1               2                 3                  4                       5                   6                  7                  8                9 
set -g subjects                 cat_statue      clock             colorful_teapot    elephant                mug_skulls          physics_mug        red_teapot         round_bird       thin_bird
set -g cls_delta_strings        "cat figurine"  "alarm clock"     "colorful teapot"  "elephant figurine"     "skull mug"         "formula mug"      "painted teapot"   "bird figurine"  "abstract figurine"       
set -g class_names              figurine        clock             teapot             figurine                mug                 mug                teapot             figurine         figurine
set -g broad_classes            0               0                 0                   0                      0                   0                    0                  0              0
# No subjects are human faces. $are_faces instructs the generation script 
# whether to compute face similarity.
set -g are_faces                0               0                 0                   0                      0                   0                    0                  0              0

# sel_set contains a few selected challenging test subjects.
#                     cat_statue      mug_skulls
set -g sel_set        1               5
set -g data_folder         subjects-ti
set -g misc_train_opts     --use_fp_trick 0
set -g misc_infer_opts
