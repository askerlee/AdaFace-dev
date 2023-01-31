#!/bin/bash

if [ $# -ne 2 ]
then
 echo "Usage: $0 file1 file2"
 exit
fi

feh  --scale-down -g 1600x200+5+30 "$1" &
feh  --scale-down -g 1600x200+0+250 "$2" &
