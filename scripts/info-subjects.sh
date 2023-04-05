#!/usr/bin/fish
#                    1                    2                         3             4            5               6           7             8            9             10          11            12                    13             14          15                  16          17             18              19          20             21          22              23              24                   25    
set -l subjects     alexachung          alita                 caradelevingne   donnieyen   gabrielleunion iainarmitage jaychou      jenniferlawrence jiffpom    keanureeves      lilbub       lisa                masatosakai michelleyeoh  princessmonstertruck ryangosling sandraoh      selenagomez   smritimandhana spikelee    stephenchow   taylorswift  timotheechalamet   tomholland            zendaya
set -l db_prompts   girl                girl                  girl             "asian man" "black woman"  "young boy"  "asian man"  girl             "pom dog"  "white man"      cat          "asian girl"        "asian man" "asian woman" "black persian cat"  "white man" "asian woman" girl          "indian girl"  "black man" "asian man"   girl         "white young man"  "white young man"     girl
set -l cls_tokens   girl                girl                  girl             man         woman          boy          man          girl             dog        man              cat          girl                man         woman         cat                  man         woman         girl          girl           man         man           girl         man                man                   girl
set -l ada_prompts  "young girl woman"  "cyborg young girl"   "young girl"     "asian man" "black woman"  "young boy"  "asian man"  "young woman"    "pom dog"  "keanu cool man" "tabby cat"  "asian young girl"  "asian man" "asian woman" "black persian cat"  "white man" "asian woman" "young girl"  "indian girl"  "black man" "asian man"   "cute girl"  "french young man" "young handsome man"  "young girl zendaya"
set -l ada_weights  "1 2 2"             "1 1 2"               "1 2"            "1 2"       "1 2"          "1 2"        "1 2"        "1 2"            "1 1"      "2 1 2"          "1 2"        "1 1 2"             "1 2"       "1 2"         "1 1 3"              "1 2"       "1 2"         "1 2"         "1 2"          "1 2"        "1 2"        "1 2"        "1 1 2"            "1 1 2"               "1 2 2"
# broad_classes are all 1, i.e., humans/animals.
set -l broad_classes  1                 1                     1                1           1              1            1            1                1          1                1            1                   1           1             1                    1           1             1             1              1           1             1            1                  1                     1           

#                     objects    humans/animals    cartoon characters
set -l lrs            3e-4       8e-4              3e-4

# donnieyen jenniferlawrence jiffpom lilbub lisa michelleyeoh selenagomez smitrimandhana taylorswift zendaya
set -l sel_set      4 8 9 11 12 14 18 19 22 25
set -Ux subjects        $subjects
set -Ux db_prompts      $db_prompts
set -Ux ada_prompts     $ada_prompts
set -Ux ada_weights     $ada_weights
set -Ux cls_tokens      $cls_tokens
# "instagram" for the main dataset, to focus on faces.
set -Ux db_suffix       ", instagram"
set -Ux sel_set         $sel_set
set -Ux lrs             $lrs
set -Ux data_folder     data
