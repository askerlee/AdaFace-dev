#!/usr/bin/fish
#                    1                    2                         3             4            5               6           7             8            9             10          11            12                    13             14          15                  16          17             18              19          20             21            22              23          24                   25    
set -l subjects     alexachung          alita                 caradelevingne   donnieyen   gabrielleunion iainarmitage jaychou     jenniferlawrence jiffpom    keanureeves      lilbub       lisa                masatosakai michelleyeoh  princessmonstertruck ryangosling sandraoh      selenagomez    smritimandhana spikelee    stephenchow   taylorswift  timotheechalamet  tomholland            zendaya
set -l db_prompts      girl             girl                  girl             "asian man" "black woman"  "young boy"   "asian man" girl            "pom dog"  "white man"         cat       "asian girl"        "asian man" "asian woman" "black persian cat"  "white man" "asian woman"    girl        "indian girl"  "black man" "asian man"    girl        "white young man"  "white young man"     girl
set -l ada_prompts  "young girl woman"  "cyborg young girl"  "young girl"      "asian man" "black woman"  "young boy"  "asian man" "young woman"    "pom dog"  "keanu cool man" "tabby cat"  "asian young girl"  "asian man" "asian woman" "black persian cat"  "white man" "asian woman" "young girl"   "indian girl"  "black man" "asian man"   "cute girl"  "french young man" "young handsome man" "young girl zendaya"
set -l ada_weights   "1 2 2"             "1 1 2"              "1 2"            "1 2"       "1 2"          "1 2"        "1 2"       "1 2"            "1 1"      "2 1 2"          "1 2"        "1 1 2"             "1 2"        "1 2"        "1 1 3"              "1 2"        "1 2"        "1 2"          "1 2"          "1 2"        "1 2"        "1 2"        "1 1 2"            "1 1 2"              "1 2 2"

set -Ux subjects        $subjects
set -Ux db_prompts      $db_prompts
set -Ux ada_prompts     $ada_prompts
set -Ux ada_weights     $ada_weights
# "instagram" for the main dataset, to focus on faces.
set -Ux db_suffix       ", instagram"
set -Ux data_folder     data
