#!/bin/bash
if [ -z "$2" ]; then
    echo "Usage: $0 <ITER-SIG> <PROMPT-SIG> <SUBJECT>"
    exit 1
fi
# if $3 is provided, then it implies $2 is PROMPT-SIG.
# Otherwise, $2 is SUBJECT.
if [ -n "$3" ]; then
    PROMPT_SIG=-$2
    SUBJECT=$3
else
    PROMPT_SIG=
    SUBJECT=$2
fi

ITER=$1
LOGFILE='manual-eval.log'

output=$(deepface verify2 --detector_backend retinaface -img1_path ~/test/${SUBJECT}-1.jpg -img2_path \
 ~/test/${SUBJECT}-adaface$ITER${PROMPT_SIG}-1.png,~/test/${SUBJECT}-adaface$ITER${PROMPT_SIG}-2.png,~/test/${SUBJECT}-adaface$ITER${PROMPT_SIG}-3.png,~/test/${SUBJECT}-adaface$ITER${PROMPT_SIG}-4.png 2>&1)
 if [ $? -ne 0 ]; then
    echo "Deepface failed. Check arguments please"
    echo $output|tail -n1
    exit 1
fi

#/home/shaohua/test/liming-adaface120309-24000-jedi-1.png: 0.482
#Average distance: 0.552
# Use awk to extract the numbers like 0.xxx
distances=$(echo "$output" | sed -n 's/.*: \(0\.[0-9]\+\).*/\1/p')
echo $SUBJECT${PROMPT_SIG} $ITER $distances >> $LOGFILE
grep -E "^$SUBJECT${PROMPT_SIG}" $LOGFILE | tail -n 10
