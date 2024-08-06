#!/usr/bin/fish
#                           1             2                 3                4          5               6               7           8                   9               10      11              12              13              14          15              16              17          18              19              20                  21              22              
set -g subjects             alexachung    alita             caradelevingne   donnieyen  gabrielleunion  iainarmitage    jaychou     jenniferlawrence    keanureeves     lisa    masatosakai     michelleyeoh    ryangosling     sandraoh    selenagomez     smritimandhana  spikelee    stephenchow     taylorswift     timotheechalamet    tomholland      zendaya
# cls_delta_strings are used during training.
set -g cls_delta_strings    woman         "young woman"     "young woman"    man        woman           "young boy"     man         "young woman"       man             woman   man             woman           man             woman       "young woman"   "young woman"   man         man             woman           "young man"         "young man"     "young woman"
# class_names are used during evaluation.
set -g class_names          girl          girl              girl             man        woman           boy             man         girl                man             girl    man             woman           man             woman       girl            girl            man         man             girl            man                 man             girl
# all bg tokens are initialized with "unknown" now. No need to use customized bg init words.
# broad_classes are all 1, i.e., humans/animals.
set -g broad_classes         1             1                 1               1          1               1               1           1                   1               1       1               1               1               1           1               1               1           1               1                1                  1               1
# Most subjects are human faces, except for the 3 cats/dogs. $are_faces instructs the generation script 
# whether to compute face similarity.
# $are_faces are used only for evaluation, not for training.
set -g are_faces             1             1                 1               1          1                1              1           1                   1               1       1               1               1               1           1               1               1           1               1                1                  1               1
set -g data_folder         subjects-celebrity
set -g misc_train_opts     
set -g misc_infer_opts      
