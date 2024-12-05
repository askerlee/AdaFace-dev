#!/bin/bash
if [ -z "$2" ]; then
    echo "Usage: $0 <ITER-SIG> <PROMPT-SUFFIX> <PREFIX>"
    exit 1
fi
# if $3 is provided, then it implies $2 is PROMPT-SUFFIX.
# Otherwise, $2 is PREFIX.
if [ -n "$3" ]; then
    SUF=-$2
    PRE=$3
else
    SUF=
    PRE=$2
fi

ITER=$1
LOGFILE='manual-eval.log'

output=$(deepface verify2 --detector_backend retinaface -img1_path ~/test/${PRE}-1.jpg -img2_path \
 ~/test/${PRE}-adaface$ITER${SUF}-1.png,~/test/${PRE}-adaface$ITER${SUF}-2.png,~/test/${PRE}-adaface$ITER${SUF}-3.png,~/test/${PRE}-adaface$ITER${SUF}-4.png 2>&1)
 if [ $? -ne 0 ]; then
    echo "Deepface failed. Check arguments please"
    echo $output|tail -n1
    exit 1
fi

#/home/shaohua/test/liming-adaface120309-24000-jedi-1.png: 0.482
#Average distance: 0.552
# Use awk to extract the numbers like 0.xxx
distances=$(echo "$output" | sed -n 's/.*: \(0\.[0-9]\+\).*/\1/p')
echo $PRE$SUF $ITER $distances >> $LOGFILE
grep $PRE$SUF $LOGFILE | tail -n 5
