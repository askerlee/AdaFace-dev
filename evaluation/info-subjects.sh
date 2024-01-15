#!/usr/bin/fish
#                   1                   2                     3                4            5                   6               7             8                     9               10              11              12                   13                 14             15                    16           17             18                 19                  20                  21            22             23                  24                 25    
set -g subjects     alexachung          alita                 caradelevingne   donnieyen    gabrielleunion      iainarmitage    jaychou      jenniferlawrence       jiffpom         keanureeves     lilbub          lisa                 masatosakai        michelleyeoh   princessmonstertruck  ryangosling  sandraoh       selenagomez        smritimandhana      spikelee            stephenchow   taylorswift    timotheechalamet    tomholland         zendaya
set -g db_prompts   "young woman"       "cyborg young girl"   "young woman"    "asian man"  "black woman"       "young boy"     "asian man"  "young woman"          "pom dog"       "white man"     "tabby cat"     "asian young woman"  "asian man"        "asian woman"  "black persian cat"   "white man"  "asian woman"  "young girl"       "indian girl"       "black man"         "asian man"   "young woman"  "white young man"   "white young man"  "young girl"
set -g cls_strings  $db_prompts
set -g ada_prompts  $db_prompts
set -g class_names   girl                girl                  girl             man          woman               boy             man          girl                   dog             man             cat             girl                 man                woman          cat                   man          woman          girl               girl                man                 man           girl           man                 man                girl

set -g ada_weights  "1 2"               "1 1 2"               "1 2"            "1 2"        "1 2"               "1 2"           "1 2"        "1 2"                  "1 1"           "1 2"           "1 2"           "1 1 2"              "1 2"              "1 2"          "1 1 3"               "1 2"        "1 2"          "1 2"              "1 2"               "1 2"               "1 2"         "1 2"          "1 1 2"             "1 1 2"            "1 2"
set -g bg_init_words  "room wall"       "dim blur"            "cover stage"    "wall stage"  "wall stage flower"  "wall stage"  "stage blur" "stage grass building" "room couch bed" "stage wall"   "room floor"    "wall color"        "wall stage plant"  "stage crowd" "room floor window"    "stage wall" "stage cover"  "stage cover room" "room window wall"  "stage room green"  "stage wall"  "stage crowd" "stage wall"         "stage blur"       "stage wall blur"
# broad_classes are all 1, i.e., humans/animals.
set -g broad_classes  1                 1                     1                1            1                   1               1            1                      1               1               1               1                    1                  1              1                     1            1              1                  1                   1                   1             1              1                   1                     1           
# Most subjects are human faces, except for the 3 cats/dogs. $are_faces instructs the generation script 
# whether to compute face similarity.
# $are_faces are used only for evaluation, not for training.
set -g are_faces     1                  1                     1                1            1                   1            1              1                       0               1               0               1                    1                  1              0                     1            1              1                  1                   1                   1             1              1                   1                     1
#                     objects    humans/animals    cartoon characters
set -g lrs            7e-4       1e-3              7e-4
set -g inf_z_prefixes     ""         "face portrait of"     ""
set -g maxiters       1500       2000              1500
# All subjects are humans/animals. The other two classes are listed for completeness.
# Individual LR for each class in the broad classes, according to their difficulties / inherent complexity.
# A prefix of "portrait of" for animals/humans suggests SD to focus on the face area of the subject.

# donnieyen jenniferlawrence jiffpom lilbub lisa michelleyeoh selenagomez smitrimandhana taylorswift zendaya
set -g sel_set              4 8 9 11 12 14 18 19 22 25
# "instagram" for the main dataset, to focus on faces.
set -g db_suffix           ", instagram"
set -g data_folder         data
set -g misc_train_opts     
set -g misc_infer_opts      
set -g resume_from_ckpt     0
set -g resumed_ckpt_keys    boy cat dog girl man woman
set -g resumed_ckpt_values  resumed_ckpts/iainarmitage2024-01-12T16-08-12_iainarmitage-ada/checkpoints/embeddings_gs-2000.pt \
                            resumed_ckpts/lilbub2024-01-12T16-08-19_lilbub-ada/checkpoints/embeddings_gs-2000.pt \
                            resumed_ckpts/jiffpom2024-01-12T21-30-12_jiffpom-ada/checkpoints/embeddings_gs-2000.pt \
                            resumed_ckpts/lisa2024-01-12T17-53-24_lisa-ada/checkpoints/embeddings_gs-2000.pt \
                            resumed_ckpts/donnieyen2024-01-12T21-29-01_donnieyen-ada/checkpoints/embeddings_gs-2000.pt \
                            resumed_ckpts/lisa2024-01-12T17-53-24_lisa-ada/checkpoints/embeddings_gs-2000.pt
