#!/usr/bin/fish
if [ "$argv[1]" = 'ada' ]; or [ "$argv[1]" = 'db' ]; or [ "$argv[1]" = 'ti' ]; or [ "$argv[1]" = 'lora' ]
    set method $argv[1]
else
    echo "Usage: $0 (ada|db|ti|lora) arg2 arg3 ..."
    exit 1
end

if [ $method = 'ada' ]; or [ $method = 'ti' ]; or [ $method = 'db' ]
    screen -dm -L -Logfile train-$method-0-(date +%m%d%H%M).txt fish scripts/train-subjects.sh $method  1 15 --gpu 0 --subjfile scripts/info-db-eval-subjects.sh $argv[2..-1]
    screen -dm -L -Logfile train-$method-1-(date +%m%d%H%M).txt fish scripts/train-subjects.sh $method 16 30 --gpu 1 --subjfile scripts/info-db-eval-subjects.sh $argv[2..-1]
else
    screen -dm -L -Logfile train-$method-0-(date +%m%d%H%M).txt fish scripts/train-subjects-lora.sh  1 15 --gpu 0 --subjfile scripts/info-db-eval-subjects.sh $argv[2..-1]
    screen -dm -L -Logfile train-$method-1-(date +%m%d%H%M).txt fish scripts/train-subjects-lora.sh 16 30 --gpu 1 --subjfile scripts/info-db-eval-subjects.sh $argv[2..-1]
end
