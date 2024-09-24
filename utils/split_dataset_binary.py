import os
import shutil
import random

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
            # shutil.copy(lbl, os.path.join(dest_dir, 'labels', os.path.basename(lbl)))

    # Move the files
    move_files(pos_train_images, pos_train_labels, os.path.join(dataset_folder, 'train', '1'))
    move_files(pos_val_images, pos_val_labels, os.path.join(dataset_folder, 'val', '1'))
    move_files(neg_train_images, neg_train_labels, os.path.join(dataset_folder, 'train', '0'))
    move_files(neg_val_images, neg_val_labels, os.path.join(dataset_folder, 'val', '0'))
    
    print(f"Dataset split completed. Training set: {len(pos_train_images) + len(neg_train_images)} images, Validation set: {len(pos_val_images) + len(neg_val_images)} images.")


# Example usage
dataset_folder = './test_dataset'
split_dataset(dataset_folder)
