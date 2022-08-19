import argparse
import csv
import glob
import os
import shutil

def main(input_folder, output_folder):
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--input_folder', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    
    args = parser.parse_args()

    main(args.input_folder, args.output_folder)

