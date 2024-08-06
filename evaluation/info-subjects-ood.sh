#!/usr/bin/fish
#                           1               2               3           4       5       6               7           8               9               10              11          12              
set -g subjects             andrewng	    daphnekoller    feifeili	hinton	ilya	kaiminghe	    lebronjames	mengchen	    qinwenzheng	    simonebiles	    yannlecun	yusufdikec
# cls_delta_strings are used during training.
set -g cls_delta_strings    man             woman           woman       man     man     "young man"     man         "young woman"   woman           woman           man         man
# class_names are used when the prompt needs to be added with a suffix.
set -g class_names          man             woman           woman       man     man     man             man         woman           woman           woman           man         man
# all bg tokens are initialized with "unknown" now. No need to use customized bg init words.
# broad_classes are all 1, i.e., humans/animals.
set -g broad_classes         1             1                 1          1       1       1               1           1               1               1               1           1               
# Most subjects are human faces, except for the 3 cats/dogs. $are_faces instructs the generation script 
# whether to compute face similarity.
# $are_faces are used only for evaluation, not for training.
set -g are_faces             1             1                 1          1       1       1              1           1                1               1               1           1               
set -g data_folder         subjects-ood
set -g misc_train_opts     
set -g misc_infer_opts      
