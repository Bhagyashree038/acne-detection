import os
from PIL import Image

# Function to remove corrupted images
def remove_invalid_images(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            try:
                with Image.open(os.path.join(root, file)) as img:
                    img.verify()  # Verify if it's a valid image
            except (IOError, SyntaxError) as e:
                print(f"Removing corrupted image: {file}")
                os.remove(os.path.join(root, file))

# Function to convert images to RGB (useful for palette images with transparency)
def convert_to_rgb(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(root, file)
                with Image.open(img_path) as img:
                    img = img.convert('RGB')  # Convert to RGB
                    img.save(img_path)

# Specify the dataset directory
dataset_dir = 'dataset'

# Step 1: Remove invalid/corrupted images
remove_invalid_images(dataset_dir)

# Step 2: Convert all images to RGB
convert_to_rgb(dataset_dir)

print("Image preprocessing is complete.")
