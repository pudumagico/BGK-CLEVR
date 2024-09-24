import argparse
import itertools
import math
import os
import sys
import re
import yaml
import random
import numpy as np
import torch
import cairo
import csv
import clingo

from PIL import Image
from tqdm import tqdm


PIXEL_SCALE = 128
BKG_COLOR = [255/255,255/255,255/255]

COLORS = {
'red': [230/255,124/255,115/255],
'yellow': [247/255,203/255,77/255],
'green': [65/255,179/255,117/255],
'blue': [123/255,170/255,247/255],
'purple': [186/255,103/255,200/255]
}

SHAPES = ['triangle','circle','cross','square', 'diamond']
SIZES = ['small','medium','big']

CLASSES = list(' '.join(e) for e in itertools.product(SHAPES, COLORS, SIZES))
CLASSES = dict(enumerate(CLASSES))
digit_mapping = inv_map = {v: k for k, v in CLASSES.items()}

DIFF = 0.35

def gen_image(i, model, grid_size, output_folder, grid_move):

    image_folder = f"/home/nhiguera/Research/cclevr/pollock/{grid_size}"

    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    # Choose an image at random
    random_image = random.choice(image_files)

    # Load the background image using PIL
    image_path = os.path.join(image_folder, random_image)
    image = Image.open(image_path)

    # Convert the image to a format suitable for Cairo
    image = image.convert("RGBA")
    image_data = np.array(image)
    surface = cairo.ImageSurface.create_for_data(image_data, cairo.FORMAT_ARGB32, grid_size*PIXEL_SCALE, grid_size*PIXEL_SCALE)
    ctx = cairo.Context(surface)
    ctx.scale(PIXEL_SCALE, PIXEL_SCALE)

    labels = []

    x_width = PIXEL_SCALE 
    y_width = PIXEL_SCALE 

    random_move = []
    for obj in model:
        
        width, height, shape, color, size = obj.strip('obj(').strip(')').split(',')
        width = int(width)
        height = int(height)

        cx = width + 0.5
        cy = height + 0.5
        scale = 1
        # Adjust DIFF based on size
        if size == 'small':
            scale = 0.5
            diff = DIFF * 0.5
        elif size == 'medium':
            diff = DIFF
        elif size == 'big':
            scale = 1.5
            diff = DIFF * 1.5
        else:
            diff = DIFF  # default to medium size if size is unknown

        if grid_move:
            cx += random.uniform(-0.25, 0.25)
            cy += random.uniform(-0.25, 0.25)
            random_move.append((cx,cy))

        if shape == 'square':
            ctx.rectangle(cx - diff, cy - diff, diff*2, diff*2)

        elif shape == 'circle':
            ctx.arc(cx, cy, diff, 0, 2*math.pi)

        elif shape == 'cross':
            ctx.move_to(cx - diff, cy - diff)
            ctx.line_to(cx + diff, cy + diff)
            ctx.move_to(cx - diff, cy + diff)
            ctx.line_to(cx + diff, cy - diff)

            ctx.set_source_rgb(*COLORS[color])
            ctx.set_line_width(0.05)
            ctx.stroke()

        elif shape == 'triangle':
            h = diff * math.sqrt(3) / 2
            ctx.move_to(cx - diff, cy + h)
            ctx.line_to(cx + diff, cy + h)
            ctx.line_to(cx, cy - h/2 - 0.4*diff)

        elif shape == 'diamond':
            ctx.move_to(cx, cy - diff)
            ctx.line_to(cx + diff, cy)
            ctx.line_to(cx, cy + diff)
            ctx.line_to(cx - diff, cy)
            ctx.close_path()
        
        else:
            continue

        ctx.set_source_rgb(*COLORS[color])
        ctx.fill()

        shape_color_size = shape + ' ' + color + ' ' + size
        class_index = list(CLASSES.keys())[list(CLASSES.values()).index(shape_color_size)]
        x_center = PIXEL_SCALE / 2 + PIXEL_SCALE * int(width)
        y_center = PIXEL_SCALE / 2 + PIXEL_SCALE * int(height)
        
        labels.append(f'{class_index} {cx/(grid_size)} {cy/(grid_size)} {scale*x_width/(grid_size*PIXEL_SCALE)} {scale*y_width/(grid_size*PIXEL_SCALE)}\n')

    surface.write_to_png(output_folder + '/' + str(i) + '.png')

    return labels, random_move


def gen_mask_image(i, model, grid_size, output_folder, grid_move, random_move):

    # Create a blank black image
    image_data = np.zeros((grid_size * PIXEL_SCALE, grid_size * PIXEL_SCALE, 4), dtype=np.uint8)
    image_data[:, :, 3] = 255  # Set alpha channel to fully opaque

    # Create a Cairo surface with the blank image
    surface = cairo.ImageSurface.create_for_data(image_data, cairo.FORMAT_ARGB32, grid_size * PIXEL_SCALE, grid_size * PIXEL_SCALE)
    ctx = cairo.Context(surface)
    ctx.scale(PIXEL_SCALE, PIXEL_SCALE)

    for j, obj in enumerate(model):
        
        width, height, shape, _, size = obj.strip('obj(').strip(')').split(',')
        width = int(width)
        height = int(height)
        
        if random_move:
            rm = random_move[j]
            cx = rm[0]
            cy = rm[1]
        else:
            cx = width + 0.5
            cy = height + 0.5

        # Adjust DIFF based on size
        if size == 'small':
            diff = DIFF * 0.5
        elif size == 'medium':
            diff = DIFF
        elif size == 'big':
            diff = DIFF * 1.5
        else:
            diff = DIFF  # default to medium size if size is unknown

        if shape == 'square':
            ctx.rectangle(cx - diff, cy - diff, diff * 2, diff * 2)
        elif shape == 'circle':
            ctx.arc(cx, cy, diff, 0, 2 * math.pi)
        elif shape == 'cross':
            ctx.move_to(cx - diff, cy - diff)
            ctx.line_to(cx + diff, cy + diff)
            ctx.move_to(cx - diff, cy + diff)
            ctx.line_to(cx + diff, cy - diff)
            ctx.set_line_width(0.05)
            ctx.set_source_rgb(1, 1, 1)  # Set the color to white
            ctx.stroke()
            continue  # Skip the fill step for crosses
        elif shape == 'triangle':
            h = diff * math.sqrt(3) / 2
            ctx.move_to(cx - diff, cy + h)
            ctx.line_to(cx + diff, cy + h)
            ctx.line_to(cx, cy - h / 2 - 0.4 * diff)
            ctx.close_path()
        elif shape == 'diamond':
            ctx.move_to(cx, cy - diff)
            ctx.line_to(cx + diff, cy)
            ctx.line_to(cx, cy + diff)
            ctx.line_to(cx - diff, cy)
            ctx.close_path()
        else:
            continue

        ctx.set_source_rgb(1, 1, 1)  # Set the color to white
        ctx.fill()

    surface.write_to_png(os.path.join(output_folder, f'{i}_mask.jpg'))


def gen_labels(models, grid_size, output_folder):
    labels_file = open(f"{output_folder}/labels.txt","w+")
    x_width = PIXEL_SCALE / 2
    y_width = PIXEL_SCALE / 2
    for model in models:
        for obj in model:
            width, heigth, shape, color, size = obj.strip('obj(').strip(')').split(',')
            shape_color_size = shape + ' ' + color + ' ' + size
            class_index = list(CLASSES.keys())[list(CLASSES.values()).index(shape_color_size)]
            x_center = PIXEL_SCALE / 2 + PIXEL_SCALE * int(width)
            y_center = PIXEL_SCALE / 2 + PIXEL_SCALE * int(heigth)
            labels_file.write(f'{class_index} {x_center/(grid_size*PIXEL_SCALE)} {y_center/(grid_size*PIXEL_SCALE)} {x_width/(grid_size*PIXEL_SCALE)} {y_width/(grid_size*PIXEL_SCALE)}\n')

def gen_label(i, model, grid_size, output_folder):
    labels_file = open(f"{output_folder}/labels/{i}.txt","w+")
    x_width = PIXEL_SCALE 
    y_width = PIXEL_SCALE 
    for obj in model:
        width, heigth, shape, color, size = obj.strip('obj(').strip(')').split(',')
        shape_color_size = shape + ' ' + color + ' ' + size
        class_index = list(CLASSES.keys())[list(CLASSES.values()).index(shape_color_size)]
        x_center = PIXEL_SCALE / 2 + PIXEL_SCALE * int(width)
        y_center = PIXEL_SCALE / 2 + PIXEL_SCALE * int(heigth)
        labels_file.write(f'{class_index} {x_center/(grid_size*PIXEL_SCALE)} {y_center/(grid_size*PIXEL_SCALE)} {x_width/(grid_size*PIXEL_SCALE)} {y_width/(grid_size*PIXEL_SCALE)}\n')

# def get_label(model, grid_size):
#     # labels_file = open(f"{output_folder}/labels/{i}.txt","w+")
#     x_width = PIXEL_SCALE 
#     y_width = PIXEL_SCALE 
#     bounding_boxes = []
#     labels = []
#     model = model.split(' ')
#     for obj in model:
#         width, heigth, shape, color, size = obj.strip('obj(').strip(')').split(',')
#         shape_color_size = shape + ' ' + color + ' ' + size
#         class_index = list(CLASSES.keys())[list(CLASSES.values()).index(shape_color_size)]
#         x_center = PIXEL_SCALE / 2 + PIXEL_SCALE * int(width)
#         y_center = PIXEL_SCALE / 2 + PIXEL_SCALE * int(heigth)

#         bounding_boxes.append([x_center/(grid_size*PIXEL_SCALE), y_center/(grid_size*PIXEL_SCALE), x_width/(grid_size*PIXEL_SCALE), y_width/(grid_size*PIXEL_SCALE)])
#         labels.append(class_index)
#         # labels_file.write(f'{class_index} {x_center/(grid_size*PIXEL_SCALE)} {y_center/(grid_size*PIXEL_SCALE)} {x_width/(grid_size*PIXEL_SCALE)} {y_width/(grid_size*PIXEL_SCALE)}\n')
#     return bounding_boxes, labels

def get_label(model, grid_size):
    x_width = PIXEL_SCALE 
    y_width = PIXEL_SCALE 
    bounding_boxes = []
    labels = []
    model = model.split(' ')
    
    for obj in model:
        width, height, shape, color, size = obj.strip('obj(').strip(')').split(',')
        shape_color_size = shape + ' ' + color + ' ' + size
        class_index = list(CLASSES.keys())[list(CLASSES.values()).index(shape_color_size)]
        
        x_center = PIXEL_SCALE / 2 + PIXEL_SCALE * int(width)
        y_center = PIXEL_SCALE / 2 + PIXEL_SCALE * int(height)

        # Adjust x_width and y_width based on size
        if size == 'small':
            diff = DIFF * 0.5
        elif size == 'medium':
            diff = DIFF
        elif size == 'big':
            diff = DIFF * 1.5
        else:
            diff = DIFF  # default to medium size if size is unknown

        adjusted_x_width = diff * 2
        adjusted_y_width = diff * 2

        bounding_boxes.append([x_center/(grid_size*PIXEL_SCALE), y_center/(grid_size*PIXEL_SCALE), adjusted_x_width/(grid_size*PIXEL_SCALE), adjusted_y_width/(grid_size*PIXEL_SCALE)])
        labels.append(class_index)

    return bounding_boxes, labels

def gen_yaml(data_dict, output_folder):

    num_classes = len(data_dict)
    class_names = [data_dict[i] for i in range(num_classes)]

    data = {
        'train': 'train',
        'val': 'val',
        'nc': num_classes,
        'names': class_names
    }

    with open(f'{output_folder}/data.yaml', 'w+') as file:
        yaml.dump(data, file, default_flow_style=False)

def gen_csv(models, classes, output_folder, binary_label, grid_size):
    headers = ["image_path", "bounding_boxes", "labels", "binary_label"]
    
    # Create a CSV file and write the headers and rows
    with open(f"{output_folder}/annotations.csv", 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Write the headers
        csvwriter.writerow(headers)
        
        # Write the rows
        for i, model in enumerate(models):
            model = [str(symbol) for symbol in model]
            model = ' '.join(model)
            bboxes, labels = get_label(model, grid_size)
            row = [f"{i}.png", f"{bboxes}", labels, binary_label]  # Default values for bounding_boxes, labels, and binary_label
            csvwriter.writerow(row)

def parse_input_string(input_string):
    input_string = [str(symbol) for symbol in input_string]
    input_string = ' '.join(input_string)
    elements = input_string.split(' ')
    matrix = np.full((5, 5), -1)  # Initialize a 5x5 matrix with -1

    for element in elements:
        # Remove 'obj(' and ')' and split the string by ','
        data = element[4:-1].split(',')
        row, col = int(data[0]), int(data[1])
        shape, color, size = data[2], data[3], data[4]
        key = f"{shape} {color} {size}"
        digit = digit_mapping[key]
        matrix[row, col] = digit
    
    return matrix

def save_list_to_file(string_list, filename):
    """
    Saves a list of strings to a file, with each string on a new line.
    
    :param string_list: List of strings to save
    :param filename: Name of the file to save the strings to
    """
    with open(filename, 'w') as file:
        for item in string_list:
            file.write(f"{item}\n")

def generate_uniform_model(grid_size, constraints, constraint_flags, existing_models):
    def is_valid(model, grid_size, constraints, constraint_flags):
        for constraint, constraint_flag in zip(constraints, constraint_flags):
        # for constraint in constraints:
            if constraint == "no_adjacent_same_shape":
                if constraint_flag == 1:
                    for x in range(grid_size):
                        for y in range(grid_size):
                            shape = model[(x, y)][0]
                            if x < grid_size - 1 and model[(x + 1, y)][0] == shape:
                                return False
                            if y < grid_size - 1 and model[(x, y + 1)][0] == shape:
                                return False
                else:
                    found = False
                    for x in range(grid_size):
                        for y in range(grid_size):
                            shape = model[(x, y)][0]
                            if x < grid_size - 1 and model[(x + 1, y)][0] == shape:
                                found = True
                            if y < grid_size - 1 and model[(x, y + 1)][0] == shape:
                                found = True
                    if not found:
                        return False
            elif constraint == "no_adjacent_same_color":
                if constraint_flag == 1:
                    for x in range(grid_size):
                        for y in range(grid_size):
                            color = model[(x, y)][1]
                            if x < grid_size - 1 and model[(x + 1, y)][1] == color:
                                return False
                            if y < grid_size - 1 and model[(x, y + 1)][1] == color:
                                return False
                else:
                    found = False
                    for x in range(grid_size):
                        for y in range(grid_size):
                            color = model[(x, y)][1]
                            if x < grid_size - 1 and model[(x + 1, y)][1] == color:
                                found = True
                            if y < grid_size - 1 and model[(x, y + 1)][1] == color:
                                found = True
                    if not found:
                        return False
            elif constraint == "same_shape_in_row":
                if constraint_flag == 1:
                    for y in range(grid_size):
                        shapes = {model[(x, y)][0] for x in range(grid_size)}
                        if len(shapes) != 1:
                            return False
                else:
                    for y in range(grid_size):
                        shapes = {model[(x, y)][0] for x in range(grid_size)}
                        if len(shapes) == 1:
                            return False
            elif constraint == "same_color_in_column":
                if constraint_flag == 1:
                    for x in range(grid_size):
                        colors = {model[(x, y)][1] for y in range(grid_size)}
                        if len(colors) != 1:
                            return False
                else:
                    for x in range(grid_size):
                        colors = {model[(x, y)][1] for y in range(grid_size)}
                        if len(colors) == 1:
                            return False
            elif constraint == "different_shape_in_row":
                if constraint_flag == 1:
                    for y in range(grid_size):
                        shapes = {model[(x, y)][0] for x in range(grid_size)}
                        if len(shapes) != grid_size:
                            return False
                else:
                    for y in range(grid_size):
                        shapes = {model[(x, y)][0] for x in range(grid_size)}
                        if len(shapes) == grid_size:
                            return False
            elif constraint == "different_color_in_column":
                if constraint_flag == 1:
                    for x in range(grid_size):
                        colors = {model[(x, y)][1] for y in range(grid_size)}
                        if len(colors) != grid_size:
                            return False
                else:
                    for x in range(grid_size):
                        colors = {model[(x, y)][1] for y in range(grid_size)}
                        if len(colors) == grid_size:
                            return False

            elif constraint == "different_shape_in_column":
                if constraint_flag == 1:
                    for x in range(grid_size):
                        shapes = {model[(x, y)][0] for y in range(grid_size)}
                        if len(shapes) != grid_size:
                            return False
                else:
                    for x in range(grid_size):
                        shapes = {model[(x, y)][0] for y in range(grid_size)}
                        if len(shapes) == grid_size:
                            return False
            elif constraint == "different_color_in_row":
                if constraint_flag == 1:
                    for y in range(grid_size):
                        colors = {model[(x, y)][1] for x in range(grid_size)}
                        if len(colors) != grid_size:
                            return False
                else:
                    for y in range(grid_size):
                        colors = {model[(x, y)][1] for x in range(grid_size)}
                        if len(colors) == grid_size:
                            return False
            elif constraint == "different_shape_in_diagonal":
                if constraint_flag == 1:
                    shapes_main = {model[(i, i)][0] for i in range(grid_size)}
                    shapes_anti = {model[(i, grid_size-i-1)][0] for i in range(grid_size)}
                    if len(shapes_main) != grid_size or len(shapes_anti) != grid_size:
                        return False
                else:
                    shapes_main = {model[(i, i)][0] for i in range(grid_size)}
                    shapes_anti = {model[(i, grid_size-i-1)][0] for i in range(grid_size)}
                    if len(shapes_main) == grid_size and len(shapes_anti) == grid_size:
                        return False
            elif constraint == "different_color_in_diagonal":
                if constraint_flag == 1:
                    colors_main = {model[(i, i)][1] for i in range(grid_size)}
                    colors_anti = {model[(i, grid_size-i-1)][1] for i in range(grid_size)}
                    if len(colors_main) != grid_size or len(colors_anti) != grid_size:
                        return False
                else:
                    colors_main = {model[(i, i)][1] for i in range(grid_size)}
                    colors_anti = {model[(i, grid_size-i-1)][1] for i in range(grid_size)}
                    if len(colors_main) == grid_size and len(colors_anti) == grid_size:
                        return False
            elif constraint == "at_least_one_of_each_shape_in_row":
                if constraint_flag == 1:
                    for y in range(grid_size):
                        shapes = {model[(x, y)][0] for x in range(grid_size)}
                        if len(shapes) != len(SHAPES):
                            return False
                else:
                    for y in range(grid_size):
                        shapes = {model[(x, y)][0] for x in range(grid_size)}
                        if len(shapes) == len(SHAPES):
                            return False
            elif constraint == "at_least_one_of_each_color_in_column":
                if constraint_flag == 1:
                    for x in range(grid_size):
                        colors = {model[(x, y)][1] for y in range(grid_size)}
                        if len(colors) != len(COLORS):
                            return False
                else:
                    for x in range(grid_size):
                        colors = {model[(x, y)][1] for y in range(grid_size)}
                        if len(colors) == len(COLORS):
                            return False
            elif constraint == "same_shape_in_corners":
                if constraint_flag == 1:
                    shape = model[(0, 0)][0]
                    if any(model[(x, y)][0] != shape for x, y in [(0, grid_size-1), (grid_size-1, 0), (grid_size-1, grid_size-1)]):
                        return False
                else:
                    shape = model[(0, 0)][0]
                    if all(model[(x, y)][0] == shape for x, y in [(0, grid_size-1), (grid_size-1, 0), (grid_size-1, grid_size-1)]):
                        return False
            elif constraint == "same_color_in_corners":
                if constraint_flag == 1:
                    color = model[(0, 0)][1]
                    if any(model[(x, y)][1] != color for x, y in [(0, grid_size-1), (grid_size-1, 0), (grid_size-1, grid_size-1)]):
                        return False
                else:
                    color = model[(0, 0)][1]
                    if all(model[(x, y)][1] == color for x, y in [(0, grid_size-1), (grid_size-1, 0), (grid_size-1, grid_size-1)]):
                        return False
            elif constraint == "different_shape_in_center":
                if grid_size % 2 == 0:
                    if constraint_flag == 1:
                        shapes = {model[(grid_size//2-1, grid_size//2-1)][0], model[(grid_size//2-1, grid_size//2)][0], model[(grid_size//2, grid_size//2-1)][0], model[(grid_size//2, grid_size//2)][0]}
                        if len(shapes) != 4:
                            return False
                    else:
                        shapes = {model[(grid_size//2-1, grid_size//2-1)][0], model[(grid_size//2-1, grid_size//2)][0], model[(grid_size//2, grid_size//2-1)][0], model[(grid_size//2, grid_size//2)][0]}
                        if len(shapes) == 4:
                            return False
            elif constraint == "different_color_in_center":
                if grid_size % 2 == 0:
                    if constraint_flag == 1:
                        colors = {model[(grid_size//2-1, grid_size//2-1)][1], model[(grid_size//2-1, grid_size//2)][1], model[(grid_size//2, grid_size//2-1)][1], model[(grid_size//2, grid_size//2)][1]}
                        if len(colors) != 4:
                            return False
                    else:
                        colors = {model[(grid_size//2-1, grid_size//2-1)][1], model[(grid_size//2-1, grid_size//2)][1], model[(grid_size//2, grid_size//2-1)][1], model[(grid_size//2, grid_size//2)][1]}
                        if len(colors) == 4:
                            return False
            elif constraint == "specific_color_in_row":
                specific_color = 'red'  # example
                if constraint_flag == 1:
                    for y in range(grid_size):ans
                else:
                    for y in range(grid_size):
                        colors = {model[(x, y)][1] for x in range(grid_size)}
                        if specific_color in colors:
                            return False
            elif constraint == "specific_shape_in_column":
                specific_shape = 'circle'  # example
                if constraint_flag == 1:
                    for x in range(grid_size):
                        shapes = {model[(x, y)][0] for y in range(grid_size)}
                        if specific_shape not in shapes:
                            return False
                else:
                    for x in range(grid_size):
                        shapes = {model[(x, y)][0] for y in range(grid_size)}
                        if specific_shape in shapes:
                            return False
            elif constraint == "no_same_shape_and_color_above":
                if constraint_flag == 1:
                    for x in range(1, grid_size):
                        for y in range(grid_size):
                            if model[(x, y)] == model[(x-1, y)]:
                                return False
                else:
                    for x in range(1, grid_size):
                        for y in range(grid_size):
                            if model[(x, y)] == model[(x-1, y)]:
                                return True
                    return False
            elif constraint == "no_same_shape_and_color_left":
                if constraint_flag == 1:
                    for x in range(grid_size):
                        for y in range(1, grid_size):
                            if model[(x, y)] == model[(x, y-1)]:
                                return False
                else:
                    for x in range(grid_size):
                        for y in range(1, grid_size):
                            if model[(x, y)] == model[(x, y-1)]:
                                return True
                    return False
            elif constraint == "specific_shape_in_each_row":
                specific_shape = 'square'  # example
                if constraint_flag == 1:
                    for y in range(grid_size):
                        if specific_shape not in {model[(x, y)][0] for x in range(grid_size)}:
                            return False
                else:
                    for y in range(grid_size):
                        if specific_shape in {model[(x, y)][0] for x in range(grid_size)}:
                            return False
            elif constraint == "specific_color_in_each_column":
                specific_color = 'green'  # example
                if constraint_flag == 1:
                    for x in range(grid_size):
                        if specific_color not in {model[(x, y)][1] for y in range(grid_size)}:
                            return False
                else:
                    for x in range(grid_size):
                        if specific_color in {model[(x, y)][1] for y in range(grid_size)}:
                            return False
            elif constraint == "same_shape_in_middle_row":
                if grid_size % 2 == 1:
                    middle_row = grid_size // 2
                    if constraint_flag == 1:
                        shapes = {model[(x, middle_row)][0] for x in range(grid_size)}
                        if len(shapes) != 1:
                            return False
                    else:
                        shapes = {model[(x, middle_row)][0] for x in range(grid_size)}
                        if len(shapes) == 1:
                            return False
            elif constraint == "same_color_in_middle_column":
                if grid_size % 2 == 1:
                    middle_column = grid_size // 2
                    if constraint_flag == 1:
                        colors = {model[(middle_column, y)][1] for y in range(grid_size)}
                        if len(colors) != 1:
                            return False
                    else:
                        colors = {model[(middle_column, y)][1] for y in range(grid_size)}
                        if len(colors) == 1:
                            return False
            elif constraint == "one_of_each_shape_in_diagonal":
                if constraint_flag == 1:
                    shapes_main = {model[(i, i)][0] for i in range(grid_size)}
                    shapes_anti = {model[(i, grid_size-i-1)][0] for i in range(grid_size)}
                    if len(shapes_main) != len(SHAPES) or len(shapes_anti) != len(SHAPES):
                        return False
                else:
                    shapes_main = {model[(i, i)][0] for i in range(grid_size)}
                    shapes_anti = {model[(i, grid_size-i-1)][0] for i in range(grid_size)}
                    if len(shapes_main) == len(SHAPES) and len(shapes_anti) == len(SHAPES):
                        return False
            elif constraint == "one_of_each_color_in_diagonal":
                if constraint_flag == 1:
                    colors_main = {model[(i, i)][1] for i in range(grid_size)}
                    colors_anti = {model[(i, grid_size-i-1)][1] for i in range(grid_size)}
                    if len(colors_main) != len(COLORS) or len(colors_anti) != len(COLORS):
                        return False
                else:
                    colors_main = {model[(i, i)][1] for i in range(grid_size)}
                    colors_anti = {model[(i, grid_size-i-1)][1] for i in range(grid_size)}
                    if len(colors_main) == len(COLORS) and len(colors_anti) == len(COLORS):
                        return False
                
        return True

    def generate_random_model(grid_size):
        model = {}
        total_cells = grid_size * grid_size
        num_shapes = len(SHAPES)
        num_colors = len(COLORS)
        num_sizes = len(SIZES)

        shape_counts = {shape: total_cells // num_shapes for shape in SHAPES}
        color_counts = {color: total_cells // num_colors for color in COLORS}
        size_counts = {color: total_cells // num_sizes for color in SIZES}

        for _ in range(total_cells % num_shapes):
            shape_counts[random.choice(SHAPES)] += 1

        for _ in range(total_cells % num_colors):
            color_counts[random.choice(list(COLORS.keys()))] += 1

        for _ in range(total_cells % num_colors):
            size_counts[random.choice(list(SIZES))] += 1

        cells = [(i, j) for i in range(grid_size) for j in range(grid_size)]
        random.shuffle(cells)

        for i, j in cells:
            shape = random.choices(list(shape_counts.keys()), weights=list(shape_counts.values()))[0]
            color = random.choices(list(color_counts.keys()), weights=list(color_counts.values()))[0]
            size = random.choices(list(size_counts.keys()), weights=list(size_counts.values()))[0]
            
            model[(i, j)] = (shape, color, size)

            shape_counts[shape] -= 1
            if shape_counts[shape] == 0:
                del shape_counts[shape]

            color_counts[color] -= 1
            if color_counts[color] == 0:
                del color_counts[color]

            size_counts[size] -= 1
            if size_counts[size] == 0:
                del size_counts[size]
        return model

    def model_to_string(model):
        return " ".join([f"obj({i},{j},{shape},{color},{size})" for (i, j), (shape, color, size) in model.items()])

    model = generate_random_model(grid_size)
    model_str = model_to_string(model)

    while not is_valid(model, grid_size, constraints, constraint_flags) or model_str in existing_models:
        model = generate_random_model(grid_size)
        model_str = model_to_string(model)

    existing_models.add(model_str)
    return model_str

def run_asp_process(program, n, constraint):
    models = []
    # print(weak_constraints)

    # Solve and get a model
    ctl = clingo.Control([f"-c c={constraint}", f"{n}"])

    # Add initial program to control object
    ctl.add("base", [], program)

    # Ground the initial program
    ctl.ground([("base", [])])
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            models.append(model.symbols(shown=True))
    if not model:
        print("No model found")

    return models


positive_folder = os.path.exists('./test_dataset/1')
negative_folder = os.path.exists('./test_dataset/0')

if not positive_folder:
    os.mkdir('./test_dataset/1', 0o777) 
if not negative_folder:
    os.mkdir('./test_dataset/0', 0o777) 

# Example usage
grid_size = 3
n_models = 1000
constraints = ["no_adj_same_shape"]  # List of constraints
constraint_flags = [1]  # Set to 1 to follow the constraint, 0 to break the constraint
existing_models = set()

main_program = open('asp_programs/main.lp').read()
constraint_programs = []
for constraint in constraints:
    constraint_program = open(f'asp_programs/{constraint}.lp').read()
    constraint_programs.append(constraint_program)

constraint_programs.append(main_program)
program = '\n'.join(constraint_programs)
positive_models = run_asp_process(program, n_models, 0)
# print(models[0])
# exit()


# Generate multiple models
# positive_models = []
for i, _ in enumerate(tqdm(range(len(positive_models)))):
    model_str = positive_models[i]
    model_str = [str(symbol) for symbol in model_str]
    model_str = ' '.join(model_str)
    labels, random_move = gen_image(i, model_str.split(' '), grid_size, './test_dataset/1', False)
    gen_mask_image(i, model_str.split(' '), grid_size, './test_dataset/1', True, random_move)
    save_list_to_file(labels, f"./test_dataset/1/{i}.txt")

gen_csv(positive_models, CLASSES, './test_dataset/1', 1, grid_size)    

positive_matrices = [parse_input_string(input_string) for input_string in positive_models]
positive_tensor = torch.tensor(positive_matrices).unsqueeze(1).float()
torch.save(positive_tensor, './test_dataset/1/matrices.pt')

# existing_models = set()
constraint_flags = [0]  # Now generate models that break the constraint
negative_models = run_asp_process(program, n_models, 1)

for i, _ in enumerate(tqdm(range(len(negative_models)))):
    model_str = negative_models[i]
    model_str = [str(symbol) for symbol in model_str]
    model_str = ' '.join(model_str)
    labels, random_move = gen_image(i, model_str.split(' '), grid_size, './test_dataset/0', False)
    gen_mask_image(i, model_str.split(' '), grid_size, './test_dataset/0', True, random_move)
    save_list_to_file(labels, f"./test_dataset/0/{i}.txt")

gen_csv(negative_models, CLASSES, './test_dataset/0', 0, grid_size)   

negative_matrices = [parse_input_string(input_string) for input_string in negative_models]
negative_tensor = torch.tensor(negative_matrices).unsqueeze(1).float()
torch.save(negative_tensor, './test_dataset/0/matrices.pt')

gen_yaml(CLASSES, 'test_dataset')