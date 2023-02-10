#!/bin/bash

if [ "$#" -lt 2 ];
then
 echo "Usage: $0 file1 file2 ..."
 exit
fi

feh  --scale-down -g 1600x200+5+30 "outputs/txt2img-samples/$1" &
feh  --scale-down -g 1600x200+0+250 "outputs/txt2img-samples/$2" &
if [ ! -z "$3" ]
then
 feh  --scale-down -g 1600x200+0+470 "outputs/txt2img-samples/$3" &
fi
if [ ! -z "$4" ]
then
 feh  --scale-down -g 1600x200+0+690 "outputs/txt2img-samples/$4" &
fi
