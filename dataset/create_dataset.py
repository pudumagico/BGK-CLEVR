import argparse
import csv
import glob
import os
import shutil

def main(input_folder, output_folder):
    
    if not os.path.exists(output_folder):
  
        os.makedirs(output_folder)
        os.makedirs(f'{output_folder}/images')

    annnotation_file = open(f'{output_folder}/ann.csv', 'w')
    writer = csv.writer(annnotation_file)

    for file in glob.glob(f'{input_folder}/positive/*'):
        dest = file.split('/')[-1]
        
        writer.writerow([f'positive_{dest}', 1])
        shutil.copy(file, f'{output_folder}/images/positive_{dest}')
    
    for file in glob.glob(f'{input_folder}/negative/*'):
        dest = file.split('/')[-1]

        writer.writerow([f'negative_{dest}', 0])
        shutil.copy(file, f'{output_folder}/images/negative_{dest}')

    # for file in glob.glob(f'{input_folder}/positive/labels/*'):    
    #     dest = file.split('/')[-1]
        
    #     shutil.copy(file, f'{output_folder}/images/labels/positive_{dest}')
    
    # for file in glob.glob(f'{input_folder}/negative/labels/*'):
    #     dest = file.split('/')[-1]

    #     shutil.copy(file, f'{output_folder}/images/labels/negative_{dest}')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--input_folder', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    
    args = parser.parse_args()

    main(args.input_folder, args.output_folder)

