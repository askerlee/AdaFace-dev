#!/bin/bash

if [ "$#" -lt 2 ];
then
 echo "Usage: $0 file1 file2 ..."
 exit
fi

if [[ $1 =~ "/" ]]; then
   folder=""
else
   folder="outputs/txt2img-samples"
fi

feh  --scale-down -g 1600x200+5+30  "$folder$1" &
feh  --scale-down -g 1600x200+0+250 "$folder$2" &
if [ ! -z "$3" ]
then
 feh  --scale-down -g 1600x200+0+470 "$folder$3" &
fi
if [ ! -z "$4" ]
then
 feh  --scale-down -g 1600x200+0+690 "$folder$4" &
fi
if [ ! -z "$5" ]
then
 feh  --scale-down -g 1600x200+0+910 "$folder$5" &
fi
