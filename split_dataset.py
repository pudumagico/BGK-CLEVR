import os
import shutil
import random

# def split_dataset(dataset_dir, split_percentage):
#     # Define the paths
#     images_dir = os.path.join(dataset_dir, 'images')
#     labels_dir = os.path.join(dataset_dir, 'labels')
#     train_images_dir = os.path.join(dataset_dir, 'train', 'images')
#     train_labels_dir = os.path.join(dataset_dir, 'train', 'labels')
#     val_images_dir = os.path.join(dataset_dir, 'val', 'images')
#     val_labels_dir = os.path.join(dataset_dir, 'val', 'labels')

#     # Create train and validation directories if they don't exist
#     os.makedirs(train_images_dir, exist_ok=True)
#     os.makedirs(train_labels_dir, exist_ok=True)
#     os.makedirs(val_images_dir, exist_ok=True)
#     os.makedirs(val_labels_dir, exist_ok=True)

#     # Get list of all images and labels
#     images = [f for f in os.listdir(images_dir) if f.endswith('.png')]
#     labels = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

#     # Sort and match images and labels
#     images.sort()
#     labels.sort()

#     # Shuffle the dataset
#     data = list(zip(images, labels))
#     random.shuffle(data)

#     # Calculate split index
#     split_index = int(len(data) * split_percentage)

#     # Split the data
#     train_data = data[:split_index]
#     val_data = data[split_index:]

#     # Copy the files to the train and val directories
#     for img, lbl in train_data:
#         shutil.copy2(os.path.join(images_dir, img), train_images_dir)
#         shutil.copy2(os.path.join(labels_dir, lbl), train_labels_dir)

#     for img, lbl in val_data:
#         shutil.copy2(os.path.join(images_dir, img), val_images_dir)
#         shutil.copy2(os.path.join(labels_dir, lbl), val_labels_dir)

#     print(f"Dataset split into {len(train_data)} training and {len(val_data)} validation samples.")

# # Example usage
# split_percentage = 0.8  # 80% training, 20% validation
# dataset_dir = 'yolo_dataset_test'  # Replace with the path to your dataset
# split_dataset(dataset_dir, split_percentage)

import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(dataset_folder, train_ratio=0.8):
    # Paths for positive and negative examples
    pos_path = os.path.join(dataset_folder, '1')
    neg_path = os.path.join(dataset_folder, '0')
    
    # Function to get images and labels from a given path
    def get_images_and_labels(path):
        images = [f for f in os.listdir(path) if f.endswith('.png')]
        labels = [os.path.join(path, 'labels', f.replace('.png', '.txt')) for f in images]
        images = [os.path.join(path, f) for f in images]
        return images, labels

    # Get images and labels for both classes
    pos_images, pos_labels = get_images_and_labels(pos_path)
    neg_images, neg_labels = get_images_and_labels(neg_path)
    
    # Split the positive and negative examples separately
    pos_train_images, pos_val_images, pos_train_labels, pos_val_labels = train_test_split(
        pos_images, pos_labels, train_size=train_ratio, random_state=42
    )
    neg_train_images, neg_val_images, neg_train_labels, neg_val_labels = train_test_split(
        neg_images, neg_labels, train_size=train_ratio, random_state=42
    )
    
    # Create directories for training and validation
    for category in ['train', 'val']:
        for label in ['0', '1']:
            os.makedirs(os.path.join(dataset_folder, category, label, 'labels'), exist_ok=True)
    
    # Function to move files to the destination folder
    def move_files(images, labels, dest_dir):
        for img, lbl in zip(images, labels):
            shutil.copy(img, os.path.join(dest_dir, os.path.basename(img)))
            shutil.copy(lbl, os.path.join(dest_dir, 'labels', os.path.basename(lbl)))

    # Move the files
    move_files(pos_train_images, pos_train_labels, os.path.join(dataset_folder, 'train', '1'))
    move_files(pos_val_images, pos_val_labels, os.path.join(dataset_folder, 'val', '1'))
    move_files(neg_train_images, neg_train_labels, os.path.join(dataset_folder, 'train', '0'))
    move_files(neg_val_images, neg_val_labels, os.path.join(dataset_folder, 'val', '0'))
    
    print(f"Dataset split completed. Training set: {len(pos_train_images) + len(neg_train_images)} images, Validation set: {len(pos_val_images) + len(neg_val_images)} images.")


# Example usage
dataset_folder = '/home/nhiguera/Research/cclevr/datasets/no_adj_same_color'
split_dataset(dataset_folder)