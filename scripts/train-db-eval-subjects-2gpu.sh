#!/usr/bin/fish
argparse --ignore-unknown --min-args 1 --max-args 10 'selset' -- $argv
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

if set -q _flag_selset
    set L1 1
    set H1 4
    set L2 5
    set H2 7
    # Pass the --selset flag to train-subjects.sh.
    set flag_selset "--selset"
else
    set L1 1
    set H1 15
    set L2 16
    set H2 30
    set flag_selset ""
end

set EXTRA_ARGS $argv[2..-1]

if [ $method = 'ada' ]; or [ $method = 'ti' ]; or [ $method = 'db' ]
    # In db eval training images, the subjects are usually small.
    # So set rand_scaling_range = (0.9, 1.1), so that the generated images tend to be larger, 
    # and have higher DINO/CLIP scores.
    if [ $method = 'ada' ]; or [ $method = 'ti' ]
        set EXTRA_ARGS --min_rand_scaling 0.9 --max_rand_scaling 1.1 --composition_delta_reg_weight 0.01 $EXTRA_ARGS
    end
    screen -dm -L -Logfile train-$method-0-(date +%m%d%H%M).txt fish scripts/train-subjects.sh $flag_selset $method $L1 $H1 --gpu 0 --subjfile scripts/info-db-eval-subjects.sh $EXTRA_ARGS 
    screen -dm -L -Logfile train-$method-1-(date +%m%d%H%M).txt fish scripts/train-subjects.sh $flag_selset $method $L2 $H2 --gpu 1 --subjfile scripts/info-db-eval-subjects.sh $EXTRA_ARGS
else
    screen -dm -L -Logfile train-$method-0-(date +%m%d%H%M).txt fish scripts/train-subjects-lora.sh $flag_selset $L1 $H1 --gpu 0 --subjfile scripts/info-db-eval-subjects.sh $EXTRA_ARGS
    screen -dm -L -Logfile train-$method-1-(date +%m%d%H%M).txt fish scripts/train-subjects-lora.sh $flag_selset $L2 $H2 --gpu 1 --subjfile scripts/info-db-eval-subjects.sh $EXTRA_ARGS
end
