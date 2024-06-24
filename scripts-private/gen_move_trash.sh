mkdir -p /data/username/trash
rm /data/username/trash/*

while read -l file_path
    # If $file_path doesn't exist
    if ! test -e $file_path
        echo "File not found: $file_path" >&2
        continue
    end
    
    # /data/username/VGGface2_HQ_masks/n001154/0342_01.jpg -> /data/username/trash/n001154-0342_01.jpg
    set trash_path (string replace "VGGface2_HQ_masks" "trash" $file_path)
    set trash_path (string replace "/0" -- "-0" $trash_path)
    echo $file_path - $trash_path >&2
    if test -e $trash_path
        echo "Duplicate file: $file_path" >&2
        continue
    end

    cp $file_path $trash_path

    # Extract the directory path
    set dir_path (dirname $file_path)

    # Replace the part of the path to form the new directory path
    set new_dir_path (string replace "VGGface2_HQ_masks" "VGGface2_HQ_masks_trash" $dir_path)

    # Create the new directory (if it doesn't exist)
    echo mkdir -p $new_dir_path
    # Move the file to the new directory
    echo mv $file_path $new_dir_path/(basename $file_path)
    set mask_path (string replace ".jpg" "_mask.png" $file_path)
    if test -e $mask_path
        echo mv $mask_path $new_dir_path/(basename $mask_path)
    end
end
