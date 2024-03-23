from PIL import Image
import os
import numpy as np


# This script will take in raw images placed in the directory folder, 
# and save a standardize version in the new_directory.
# The images are currently resized to 400x400 and maintains aspect ration
# by applying a white border.

directory = './bears_raw'
new_directory = './bears'


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    

image_sizes = []
for folder in os.listdir(directory):
    if folder not in ['black', 'grizzly', 'teddy']:
        continue

    print(f'Converting images in {folder} folder.')

    os.makedirs(f'{new_directory}/{folder}', exist_ok=True)
    for filename in os.listdir(f'{directory}/{folder}'):
        if 'g' in filename.lower(): #will convert png and jpg in this case
            im = Image.open(f'{directory}/{folder}/{filename}')
            im = expand2square(im, (255, 255, 255)).resize((400, 400))
            image_sizes.append(np.array(im).shape)
            name = f"{filename.split('.')[0]}.jpg"
            rgb_im = im.convert('RGB')
            rgb_im.save(f'{new_directory}/{folder}/{name}')

avg_size = np.array(image_sizes).mean(axis=0).astype(int)
print(f'Average image size: {avg_size}')
print(f'Average ratio: {avg_size[0]/avg_size[1]}')
print('Data setup complete.')