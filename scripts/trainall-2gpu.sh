#!/usr/bin/fish
if [ "$argv[1]" = 'ada' ]; or [ "$argv[1]" = 'db' ]; or [ "$argv[1]" = 'ti' ]; or [ "$argv[1]" = 'lora' ]
    set method $argv[1]
else
    echo "Usage: $0 (ada|db|ti|lora)"
    exit 1
end

screen -dm -L -Logfile train$method-0-(date +%m%d%H%M).txt fish scripts/train-range-$method.sh --gpu 0 1 13
screen -dm -L -Logfile train$method-1-(date +%m%d%H%M).txt fish scripts/train-range-$method.sh --gpu 1 14 25
