#!/usr/bin/fish
argparse --ignore-unknown --min-args 1 --max-args 20 'selset' -- $argv
or begin
    echo "Usage: $0 [--selset] (ada|db|ti|lora) arg2 arg3 ..."
    exit 1
end

if [ "$argv[1]" = 'ada' ]; or [ "$argv[1]" = 'db' ]; or [ "$argv[1]" = 'ti' ]; or [ "$argv[1]" = 'lora' ]
    set method $argv[1]
else
    echo "Usage: $0 [--selset] (ada|db|ti|lora) arg2 arg3 ..."
    exit 1
end

set EXTRA_ARGS $argv[2..-1]

set info_file evaluation/info-dbeval-subjects.sh

if set -q _flag_selset
    source $info_file
    if not set -q _flag_selset
        echo "Error: 'sel_set' is not specified in $info_file."
        exit 1
    end

    set LEN (count $sel_set)
    # Pass the --selset flag to train-subjects.sh.
    set EXTRA_ARGS "--selset" $EXTRA_ARGS
else
    set LEN (count $subjects)
end

set L1 1
set H1 (math ceil $LEN / 2)
set L2 (math ceil $LEN / 2 + 1)
set H2 $LEN

if [ $method = 'ada' ]; or [ $method = 'ti' ]; or [ $method = 'db' ]
    # In db eval training images, the subjects are usually small.
    # So set rand_scaling_range = (0.9, 1.1), so that the generated images tend to be larger, 
    # and have higher DINO/CLIP scores.
    if [ $method = 'ada' ]; or [ $method = 'ti' ]
        set EXTRA_ARGS --min_rand_scaling 0.9 --max_rand_scaling 1.1 --prompt_delta_reg_weight 0.01 --embedding_reg_weight 0.01 $EXTRA_ARGS
    end
    screen -dm -L -Logfile train-$method-0-(date +%m%d%H%M).txt fish scripts/train-subjects.sh $method $L1 $H1 --gpu 0 --subjfile $info_file $EXTRA_ARGS 
    screen -dm -L -Logfile train-$method-1-(date +%m%d%H%M).txt fish scripts/train-subjects.sh $method $L2 $H2 --gpu 1 --subjfile $info_file $EXTRA_ARGS
else
    screen -dm -L -Logfile train-$method-0-(date +%m%d%H%M).txt fish scripts/train-subjects-lora.sh $flag_selset $L1 $H1 --gpu 0 --subjfile $info_file $EXTRA_ARGS
    screen -dm -L -Logfile train-$method-1-(date +%m%d%H%M).txt fish scripts/train-subjects-lora.sh $flag_selset $L2 $H2 --gpu 1 --subjfile $info_file $EXTRA_ARGS
end
