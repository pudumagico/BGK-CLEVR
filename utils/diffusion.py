import os
import base64
import requests
from datetime import datetime
import time
import json
import urllib.request

# Set the API URL for Stable Diffusion WebUI
api_url = "http://127.0.0.1:7860/api/generate"
webui_server_url = 'http://127.0.0.1:7860'

# Folder paths
image_folder = "test_dataset/0"
mask_folder = "test_dataset/0"
output_folder = "NEW"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

out_dir = 'api_out'
out_dir_t2i = os.path.join(out_dir, 'txt2img')
out_dir_i2i = os.path.join(out_dir, 'img2img')
os.makedirs(out_dir_t2i, exist_ok=True)
os.makedirs(out_dir_i2i, exist_ok=True)

def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


def encode_file_to_base64(path):
    with open(path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')


def decode_and_save_base64(base64_str, save_path):
    with open(save_path, "wb") as file:
        file.write(base64.b64decode(base64_str))


def call_api(api_endpoint, **payload):
    data = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(
        f'{webui_server_url}/{api_endpoint}',
        headers={'Content-Type': 'application/json'},
        data=data,
    )
    response = urllib.request.urlopen(request)
    return json.loads(response.read().decode('utf-8'))

def call_img2img_api(posneg, image_id, **payload):
    response = call_api('sdapi/v1/img2img', **payload)
    # print(response['parameters'])
    # exit()
    for index, image in enumerate(response.get('images')):
        save_path = os.path.join(out_dir_i2i, f'{posneg}/{image_id}.png')
        decode_and_save_base64(image, save_path)

# Loop through all files in the image folder
for filename in os.listdir(image_folder):
    if filename.endswith(".png"):
        image_id = filename.split(".")[0]
        image_path = os.path.join(image_folder, filename)
        mask_path = os.path.join(mask_folder, f"{image_id}_mask.jpg")
        
        if os.path.exists(mask_path):
            with open(image_path, "rb") as img_file, open(mask_path, "rb") as mask_file:
                # Encode the image and mask as base64 strings
                image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                mask_base64 = base64.b64encode(mask_file.read()).decode('utf-8')
            
                payload = {
                    "prompt": "fuzzy",
                    "seed": 1,
                    "steps": 10,
                    "width": 512,
                    "height": 512,
                    "denoising_strength": 0.55,
                    "n_iter": 1,
                    "init_images": [image_base64],
                    "mask": mask_base64,
                    "mask_blur": 0,
                    "inpainting_fill": 1,
                    # "inpaint_full_res": True,
                    "inpainting_mask": 1,
                    "inpaint_masked": 1,
                    # "inpaint_full_res_padding": 0,
                }

                call_img2img_api(0, image_id, **payload)

image_folder = "test_dataset/1"
mask_folder = "test_dataset/1"
# Loop through all files in the image folder
for filename in os.listdir(image_folder):
    if filename.endswith(".png"):
        image_id = filename.split(".")[0]
        image_path = os.path.join(image_folder, filename)
        mask_path = os.path.join(mask_folder, f"{image_id}_mask.jpg")
        
        if os.path.exists(mask_path):
            with open(image_path, "rb") as img_file, open(mask_path, "rb") as mask_file:
                # Encode the image and mask as base64 strings
                image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                mask_base64 = base64.b64encode(mask_file.read()).decode('utf-8')
            
                payload = {
                    "prompt": "fuzzy",
                    "seed": 1,
                    "steps": 10,
                    "width": 512,
                    "height": 512,
                    "denoising_strength": 0.55,
                    "n_iter": 1,
                    "init_images": [image_base64],
                    "mask": mask_base64,
                    "mask_blur": 0,
                    "inpainting_fill": 1,
                    # "inpaint_full_res": True,
                    "inpainting_mask": 1,
                    "inpaint_masked": 1,
                    # "inpaint_full_res_padding": 0,
                }

                call_img2img_api(1, image_id, **payload)


print("Processing complete.")
