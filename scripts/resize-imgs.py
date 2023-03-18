import os
from PIL import Image

def resize_images_in_subdirectories(root_dir, size=(256, 256)):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            filepath = os.path.join(subdir, file)

            try:
                img = Image.open(filepath)
                img_resized = img.resize(size, Image.ANTIALIAS)
                img_resized.save(filepath)
                print(f"Resized image: {filepath}")
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

if __name__ == "__main__":
    root_directory = "data-256"  # Replace with the path to your root directory
    resize_images_in_subdirectories(root_directory)
