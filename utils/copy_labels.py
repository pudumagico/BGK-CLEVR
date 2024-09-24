import os
import shutil

def copy_txt_files(source_folder, destination_folder):
    # Check if the source and destination folders exist
    if not os.path.exists(source_folder):
        print(f"The source folder '{source_folder}' does not exist.")
        return
    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"The destination folder '{destination_folder}' was created.")

    # Iterate through the files in the source folder
    for filename in os.listdir(source_folder):
        # Check if the file has a .txt extension
        if filename.endswith('.txt'):
            # Construct the full file path
            source_file = os.path.join(source_folder, filename)
            destination_file = os.path.join(destination_folder, filename)
            
            # Copy the file to the destination folder
            shutil.copy(source_file, destination_file)
            print(f"Copied '{filename}' to '{destination_folder}'")

# Example usage:
copy_txt_files('test_dataset/1', 'api_out/img2img/1')
