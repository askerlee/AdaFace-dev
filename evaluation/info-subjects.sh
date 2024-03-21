#!/usr/bin/fish
#                               1                   2                     3                4            5                   6               7             8                     9               10              11              12                   13                 14             15                    16           17             18                 19                  20                  21            22             23                  24                 25    
set -g subjects                 alexachung          alita                 caradelevingne   donnieyen    gabrielleunion      iainarmitage    jaychou      jenniferlawrence       jiffpom         keanureeves     lilbub          lisa                 masatosakai        michelleyeoh   princessmonstertruck  ryangosling  sandraoh       selenagomez        smritimandhana      spikelee            stephenchow   taylorswift    timotheechalamet    tomholland         zendaya
set -g cls_delta_strings        "woman"             "cyborg young girl"   "young woman"    "asian man"  "black woman"       "young boy"     "asian man"  "young woman"          "pom dog"       "man"           "tabby cat"     "asian young woman"  "asian man"        "asian woman"  "black persian cat"   "man"        "asian woman"  "young girl"       "indian girl"       "black man"         "asian man"   "woman"        "young man"         "young man"        "young girl"
set -g class_names              girl                girl                  girl             man          woman               boy             man          girl                   dog             man             cat             girl                 man                woman          cat                   man          woman          girl               girl                man                 man           girl           man                 man                girl
# all bg tokens are initialized with "unknown" now. No need to use customized bg init words.
#set -g bg_init_strings        "room wall"         "dim blur"            "cover stage"    "wall stage"  "wall stage flower"  "wall stage"  "stage blur" "stage grass building" "room couch bed" "stage wall"   "room floor"    "wall color"        "wall stage plant"  "stage crowd" "room floor window"    "stage wall" "stage cover"  "stage cover room" "room window wall"  "stage room green"  "stage wall"  "stage crowd" "stage wall"         "stage blur"       "stage wall blur"
# broad_classes are all 1, i.e., humans/animals.
set -g broad_classes            1                  1                     1                1            1                   1               1            1                      1               1               1               1                    1                  1              1                     1            1              1                  1                   1                   1             1              1                   1                     1           
# Most subjects are human faces, except for the 3 cats/dogs. $are_faces instructs the generation script 
# whether to compute face similarity.
# $are_faces are used only for evaluation, not for training.
set -g are_faces                1                  1                     1                1            1                   1            1              1                       0               1               0               1                    1                  1              0                     1            1              1                  1                   1                   1             1              1                   1                     1
#                     objects    humans/animals    cartoon characters
set -g lrs            7e-4       1e-3              7e-4
set -g inf_z_prefixes     ""         "face portrait of"     ""
set -g maxiters       1500       2000              1500
# All subjects are humans/animals. The other two classes are listed for completeness.
# Individual LR for each class in the broad classes, according to their difficulties / inherent complexity.
# A prefix of "portrait of" for animals/humans suggests SD to focus on the face area of the subject.

# caradelevingne jenniferlawrence masatosakai michelleyeoh selenagomez taylorswift tomholland
set -g sel_set              3 8 13 14 18 22 24
# "instagram" for the main dataset, to focus on faces.
set -g db_suffix           ", instagram"
set -g data_folder         subjects-celebrity
set -g misc_train_opts     
set -g misc_infer_opts      
set -g resume_from_ckpt     0
set -g resumed_ckpt_keys    boy cat dog girl man woman
#set -g resumed_ckpt_values  resumed_ckpts/iainarmitage2024-01-12T16-08-12_iainarmitage-ada/checkpoints/embeddings_gs-2000.pt