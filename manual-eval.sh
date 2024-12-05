#!/bin/bash
if [ -z "$2" ]; then
    echo "Usage: $0 <PREFIX> <ITER-SIG> <PROMPT-SUFFIX>"
    exit 1
fi
# if $3 is provided
if [ -n "$3" ]; then
    SUF=-$3
fi

LOGFILE='manual-eval.log'
PRE=$1
ITER=$2
output=$(deepface verify2 --detector_backend retinaface -img1_path ~/test/${PRE}-1.jpg -img2_path \
 ~/test/${PRE}-adaface$ITER${SUF}-1.png,~/test/${PRE}-adaface$ITER${SUF}-2.png,~/test/${PRE}-adaface$ITER${SUF}-3.png,~/test/${PRE}-adaface$ITER${SUF}-4.png)
#/home/shaohua/test/liming-adaface120309-24000-jedi-1.png: 0.482
#Average distance: 0.552
# Use awk to extract the numbers like 0.xxx
distances=$(echo "$output" | sed -n 's/.*\([0-9]\+\.[0-9]\+\).*/\1/p')
echo $PRE$SUF $ITER $distances >> $LOGFILE
grep $PRE$SUF $LOGFILE | tail -n 5
