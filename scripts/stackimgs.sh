#!/usr/bin/fish
set scriptname (status -f)
if test (count $argv) -lt 1
 echo "Usage: $scriptname file1 file2 ..."
 exit
end

set default_folder "samples-ada/"

# screen dimensions: 1600x1080
set WIDTH 1600
set HEIGHT 1080
set max_images 5
set padding 10
set img_height (math (math $HEIGHT - 20) / $max_images - $padding)

set start_idx (count $argv)
set end_idx (math max 1, (math $start_idx - $max_images + 1))
set h_offset $padding

for img_idx in (seq $start_idx -1 $end_idx)
  set img_path $argv[$img_idx]
  # test if img_path exists
   if test -f $img_path
      set folder ""
   else
      set folder $default_folder
   end

  #echo $argv[$img_idx]
  feh  --scale-down -g {$WIDTH}x200+0+{$h_offset}  "$folder$argv[$img_idx]" &
  set h_offset (math $h_offset + $img_height)
end 

