#!/usr/bin/fish
set scriptname (status -f)

if test (count $argv) -lt 2
 echo "Usage: $scriptname file1 file2 ..."
 exit
end

if string match -eq "/" $argv[1] 
   set folder ""
else
   set folder "samples-ada/"
end

# screen dimensions: 1600x1200
set WIDTH 1600
set HEIGHT 1200
set max_images 5
set padding 20
set img_height (math (math $HEIGHT - 30) / $max_images - $padding)

set start_idx (count $argv)
set end_idx (math max 1, (math $start_idx - $max_images + 1))
set h_offset $padding

for img_idx in (seq $start_idx -1 $end_idx)
  #echo $argv[$img_idx]
  feh  --scale-down -g {$WIDTH}x200+0+{$h_offset}  "$folder$argv[$img_idx]" &
  set h_offset (math $h_offset + $img_height)
end 

