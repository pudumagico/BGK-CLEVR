import os
import shutil

def create_directories(root_folder):
    # Define the paths for the new directories
    images_dir = os.path.join(root_folder, 'images')
    labels_dir = os.path.join(root_folder, 'labels')
    
    # Create new directories
    for directory in [images_dir, labels_dir]:
        for subdir in ['train', 'val']:
            os.makedirs(os.path.join(directory, subdir), exist_ok=True)
    
    return images_dir, labels_dir

def copy_files(src_folder, images_dir, labels_dir):
    for subdir in ['train', 'val']:
        src_subdir = os.path.join(src_folder, subdir)
        dest_images_subdir = os.path.join(images_dir, subdir)
        dest_labels_subdir = os.path.join(labels_dir, subdir)
        
        for item in os.listdir(src_subdir):
            item_path = os.path.join(src_subdir, item)
            if os.path.isfile(item_path):
                if item.endswith('.png'):
                    shutil.copy(item_path, os.path.join(dest_images_subdir, item))
                elif item.endswith('.txt'):
                    shutil.copy(item_path, os.path.join(dest_labels_subdir, item))

def main(root_folder):
    images_dir, labels_dir = create_directories(root_folder)
    copy_files(root_folder, images_dir, labels_dir)
    print("Files have been successfully copied.")

if __name__ == "__main__":
    root_folder = 'test_dataset'
    main(root_folder)
