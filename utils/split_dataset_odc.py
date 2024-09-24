import os
import shutil
import random

def get_files(folder, extension):
    return [f for f in os.listdir(folder) if f.endswith(extension)]

def rename_files(folder_0, folder_1):
    images_0 = get_files(folder_0, '.png')
    images_1 = get_files(folder_1, '.png')
    length_1 = len(images_1)
    
    for image in images_0:
        base_name = os.path.splitext(image)[0]
        new_name = f"{int(base_name) + int(length_1)}.png"
        os.rename(os.path.join(folder_0, image), os.path.join(folder_0, new_name))
        
        label = f"{base_name}.txt"
        if label in get_files(folder_0, '.txt'):
            new_label_name = f"{int(base_name) + int(length_1)}.txt"
            os.rename(os.path.join(folder_0, label), os.path.join(folder_0, new_label_name))

def split_dataset(root_folder, train_ratio=0.8):
    folder_0 = os.path.join(root_folder, '0')
    folder_1 = os.path.join(root_folder, '1')
    
    rename_files(folder_0, folder_1)
    
    all_images = get_files(folder_0, '.png') + get_files(folder_1, '.png')
    all_images.sort()
    all_labels = get_files(folder_0, '.txt') + get_files(folder_1, '.txt')
    all_labels.sort()
    combined = list(zip(all_images, all_labels))
    random.shuffle(combined)
    
    train_size = int(train_ratio * len(combined))
    train_set = combined[:train_size]
    val_set = combined[train_size:]
    
    train_folder = os.path.join(root_folder, 'train')
    val_folder = os.path.join(root_folder, 'val')
    
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    
    for img, lbl in train_set:
        shutil.copy(os.path.join(folder_0, img), os.path.join(train_folder, img)) if os.path.exists(os.path.join(folder_0, img)) else shutil.copy(os.path.join(folder_1, img), os.path.join(train_folder, img))
        shutil.copy(os.path.join(folder_0, lbl), os.path.join(train_folder, lbl)) if os.path.exists(os.path.join(folder_0, lbl)) else shutil.copy(os.path.join(folder_1, lbl), os.path.join(train_folder, lbl))
    
    for img, lbl in val_set:
        shutil.copy(os.path.join(folder_0, img), os.path.join(val_folder, img)) if os.path.exists(os.path.join(folder_0, img)) else shutil.copy(os.path.join(folder_1, img), os.path.join(val_folder, img))
        shutil.copy(os.path.join(folder_0, lbl), os.path.join(val_folder, lbl)) if os.path.exists(os.path.join(folder_0, lbl)) else shutil.copy(os.path.join(folder_1, lbl), os.path.join(val_folder, lbl))

if __name__ == "__main__":
    root_folder = 'test_dataset'
    split_dataset(root_folder)
