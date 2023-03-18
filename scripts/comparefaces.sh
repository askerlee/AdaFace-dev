#!/usr/bin/fish
set fish_trace 1
set -l subjects alexachung         caradelevingne corgi        donnieyen   gabrielleunion iainarmitage jaychou     jenniferlawrence jiffpom    keanureeves      lilbub       lisa                masatosakai michelleyeoh  princessmonstertruck ryangosling sandraoh      selenagomez    smritimandhana spikelee    stephenchow   taylorswift  timotheechalamet  tomholland            zendaya
set type $argv[1]

for i in (seq 1 25)
    set subject $subjects[$i]
    python3 scripts/comparefaces.py data/$subject samples-$type/$subject
end
