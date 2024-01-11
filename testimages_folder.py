import pandas as pd
import os
import shutil

df = pd.read_csv('./flickr30k_processed/test.csv')
image_paths = df['image'].apply(lambda x: os.path.join('./flickr30k_processed/', x))
image_paths = image_paths.drop_duplicates()

output_folder = './flickr30k_processed/test_images'
os.makedirs(output_folder, exist_ok=True)

for image_path in image_paths:
    shutil.copy(image_path, output_folder)
