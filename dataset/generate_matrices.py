import argparse
import itertools
import math
import os
import sys
import re
import yaml

import torch
import numpy as np

from tqdm import tqdm

PIXEL_SCALE = 128
BKG_COLOR = [255/255,255/255,255/255]

# COLORS_PS = {
# 'pink': [240/255,161/255,211/255],
# 'blue': [159/255,223/255,244/255],
# 'red': [242/255,133/255,119/255],
# 'green': [102/255,226/255,219/255]
# }

COLORS = {
'red': [230/255,124/255,115/255],
'yellow': [247/255,203/255,77/255],
'green': [65/255,179/255,117/255],
'blue': [123/255,170/255,247/255],
'purple': [186/255,103/255,200/255]
}

SHAPES = ['triangle','circle','cross','square', 'diamond']

CLASSES = list(' '.join(e) for e in itertools.product(SHAPES, COLORS))
CLASSES = dict(enumerate(CLASSES))
digit_mapping = inv_map = {v: k for k, v in CLASSES.items()}
DIFF = 0.35

# # Input string
# input_strings = [
#     "obj(4,3,diamond,yellow) obj(4,1,triangle,purple) obj(4,0,square,purple) obj(3,4,square,purple) obj(2,4,triangle,red) obj(2,0,cross,purple) obj(1,3,diamond,blue) obj(1,2,cross,red) obj(0,3,triangle,red) obj(0,2,circle,red) obj(0,0,triangle,green) obj(4,4,circle,green) obj(4,2,cross,green) obj(3,3,square,yellow) obj(3,1,cross,blue) obj(3,0,diamond,red) obj(2,3,cross,blue) obj(2,2,cross,yellow) obj(1,1,triangle,blue) obj(0,4,triangle,purple) obj(0,1,cross,green) obj(3,2,circle,purple) obj(1,4,cross,yellow) obj(1,0,triangle,green) obj(2,1,square,green)",
#     # Add more input strings as needed
# ]

# Function to parse the input string and convert to matrix of digits
def parse_input_string(input_string):
    elements = input_string.split(' ')
    matrix = np.full((5, 5), -1)  # Initialize a 5x5 matrix with -1

    for element in elements:
        # Remove 'obj(' and ')' and split the string by ','
        data = element[4:-1].split(',')
        row, col = int(data[0]), int(data[1])
        shape, color = data[2], data[3]
        key = f"{shape} {color}"
        digit = digit_mapping[key]
        matrix[row, col] = digit
    
    return matrix

# # Process each input string and store the matrices
# matrices = [parse_input_string(input_string) for input_string in input_strings]

# # Convert the list of matrices to a 4D torch tensor
# tensor = torch.tensor(matrices).unsqueeze(1).float()

# # Save the tensor to a file
# torch.save(tensor, 'matrices.pt')

# # Display the tensor to verify
# print(tensor)