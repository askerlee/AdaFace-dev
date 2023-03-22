#!/usr/bin/fish
if [ "$argv[1]" = 'ada' ]; or [ "$argv[1]" = 'db' ]; or [ "$argv[1]" = 'ti' ]; or [ "$argv[1]" = 'lora' ]
    set method $argv[1]
else
    echo "Usage: $0 (ada|db|ti|lora) arg2 arg3 ..."
    exit 1
end

if [ $method = 'ada' ]; or [ $method = 'ti' ]; or [ $method = 'db' ]
    # ... $argv[2..-1]: pass the rest of the arguments to train-subjects.sh.
    screen -dm -L -Logfile train-$method-0-(date +%m%d%H%M).txt fish scripts/train-subjects.sh $method 1 13  --gpu 0 --subjfile scripts/info-subjects.sh $argv[2..-1]
    screen -dm -L -Logfile train-$method-1-(date +%m%d%H%M).txt fish scripts/train-subjects.sh $method 14 25 --gpu 1 --subjfile scripts/info-subjects.sh $argv[2..-1] 
else
    screen -dm -L -Logfile train-$method-0-(date +%m%d%H%M).txt fish scripts/train-subjects-lora.sh 1 13  --gpu 0 --subjfile scripts/info-subjects.sh $argv[2..-1]
    screen -dm -L -Logfile train-$method-1-(date +%m%d%H%M).txt fish scripts/train-subjects-lora.sh 14 25 --gpu 1 --subjfile scripts/info-subjects.sh $argv[2..-1]
end
