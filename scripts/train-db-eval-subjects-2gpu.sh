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

if [ $method = 'ada' ]; or [ $method = 'ti' ]; or [ $method = 'db' ]
    # Set min_rand_scaling = 0.9, so that the generated images tend to be larger, 
    # and have higher DINO/CLIP scores.
    screen -dm -L -Logfile train-$method-0-(date +%m%d%H%M).txt fish scripts/train-subjects.sh $flag_selset $method $L1 $H1 --gpu 0 --subjfile scripts/info-db-eval-subjects.sh --min_rand_scaling 0.9 $argv[2..-1]
    screen -dm -L -Logfile train-$method-1-(date +%m%d%H%M).txt fish scripts/train-subjects.sh $flag_selset $method $L2 $H2 --gpu 1 --subjfile scripts/info-db-eval-subjects.sh --min_rand_scaling 0.9 $argv[2..-1]
else
    screen -dm -L -Logfile train-$method-0-(date +%m%d%H%M).txt fish scripts/train-subjects-lora.sh $flag_selset $L1 $H1 --gpu 0 --subjfile scripts/info-db-eval-subjects.sh $argv[2..-1]
    screen -dm -L -Logfile train-$method-1-(date +%m%d%H%M).txt fish scripts/train-subjects-lora.sh $flag_selset $L2 $H2 --gpu 1 --subjfile scripts/info-db-eval-subjects.sh $argv[2..-1]
end
