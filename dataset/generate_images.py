import argparse
import itertools
import math
import os
import sys
import re
import yaml

import cairo
import csv

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
DIFF = 0.35

def gen_image(i, model, grid_size, output_folder):

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, grid_size*PIXEL_SCALE, grid_size*PIXEL_SCALE)
    ctx = cairo.Context(surface)
    ctx.scale(PIXEL_SCALE, PIXEL_SCALE)

    ctx.rectangle(0, 0, grid_size, grid_size)
    ctx.set_source_rgb(*BKG_COLOR)
    ctx.fill()

    for obj in model:
        
        width, heigth, shape, color = obj.strip('obj(').strip(')').split(',')
        width = int(width)
        heigth = int(heigth)

        cx = width + 0.5
        cy = heigth + 0.5

        if shape == 'square':

            ctx.rectangle(cx - DIFF, cy - DIFF, DIFF*2, DIFF*2)

        elif shape == 'circle':
            ctx.arc(cx, cy, DIFF, 0, 2*math.pi)

        elif shape == 'cross':

            ctx.move_to(cx - DIFF, cy - DIFF)
            ctx.line_to(cx + DIFF, cy + DIFF)

            ctx.move_to(cx - DIFF, cy + DIFF)
            ctx.line_to(cx + DIFF, cy - DIFF)

            ctx.set_source_rgb(*COLORS[color])
            ctx.set_line_width(0.05)
            ctx.stroke()

        elif shape == 'triangle':
            
            h = DIFF*math.sqrt(3)/2

            ctx.move_to(cx - DIFF, cy + h)
            ctx.line_to(cx + DIFF, cy + h)
            ctx.line_to(cx, cy - h/2 - 0.4*DIFF)

        elif shape == 'diamond':
            ctx.move_to(cx, cy - DIFF)
            ctx.line_to(cx + DIFF, cy)
            ctx.line_to(cx, cy + DIFF)
            ctx.line_to(cx - DIFF, cy)
            ctx.close_path()
        
        else:
            continue

        ctx.set_source_rgb(*COLORS[color])
        ctx.fill()
    surface.write_to_png(output_folder + '/' + str(i) + '.png')

def gen_labels(models, grid_size, output_folder):
    labels_file = open(f"{output_folder}/labels.txt","w+")
    x_width = PIXEL_SCALE / 2
    y_width = PIXEL_SCALE / 2
    for model in models:
        for obj in model:
            width, heigth, shape, color = obj.strip('obj(').strip(')').split(',')
            shape_color = shape + ' ' + color
            class_index = list(CLASSES.keys())[list(CLASSES.values()).index(shape_color)]
            x_center = PIXEL_SCALE / 2 + PIXEL_SCALE * int(width)
            y_center = PIXEL_SCALE / 2 + PIXEL_SCALE * int(heigth)
            labels_file.write(f'{class_index} {x_center/(grid_size*PIXEL_SCALE)} {y_center/(grid_size*PIXEL_SCALE)} {x_width/(grid_size*PIXEL_SCALE)} {y_width/(grid_size*PIXEL_SCALE)}\n')

def gen_label(i, model, grid_size, output_folder):
    labels_file = open(f"{output_folder}/labels/{i}.txt","w+")
    x_width = PIXEL_SCALE 
    y_width = PIXEL_SCALE 
    for obj in model:
        width, heigth, shape, color = obj.strip('obj(').strip(')').split(',')
        shape_color = shape + ' ' + color
        class_index = list(CLASSES.keys())[list(CLASSES.values()).index(shape_color)]
        x_center = PIXEL_SCALE / 2 + PIXEL_SCALE * int(width)
        y_center = PIXEL_SCALE / 2 + PIXEL_SCALE * int(heigth)
        labels_file.write(f'{class_index} {x_center/(grid_size*PIXEL_SCALE)} {y_center/(grid_size*PIXEL_SCALE)} {x_width/(grid_size*PIXEL_SCALE)} {y_width/(grid_size*PIXEL_SCALE)}\n')

def get_label(model, grid_size):
    # labels_file = open(f"{output_folder}/labels/{i}.txt","w+")
    x_width = PIXEL_SCALE 
    y_width = PIXEL_SCALE 
    bounding_boxes = []
    labels = []
    model = model.split(' ')
    for obj in model:
        width, heigth, shape, color = obj.strip('obj(').strip(')').split(',')
        shape_color = shape + ' ' + color
        class_index = list(CLASSES.keys())[list(CLASSES.values()).index(shape_color)]
        x_center = PIXEL_SCALE / 2 + PIXEL_SCALE * int(width)
        y_center = PIXEL_SCALE / 2 + PIXEL_SCALE * int(heigth)

        bounding_boxes.append([x_center/(grid_size*PIXEL_SCALE), y_center/(grid_size*PIXEL_SCALE), x_width/(grid_size*PIXEL_SCALE), y_width/(grid_size*PIXEL_SCALE)])
        labels.append(class_index)
        # labels_file.write(f'{class_index} {x_center/(grid_size*PIXEL_SCALE)} {y_center/(grid_size*PIXEL_SCALE)} {x_width/(grid_size*PIXEL_SCALE)} {y_width/(grid_size*PIXEL_SCALE)}\n')
    return bounding_boxes, labels

def gen_yaml(data_dict, output_folder):

    num_classes = len(data_dict)
    class_names = [data_dict[i] for i in range(num_classes)]

    data = {
        'train': f'{output_folder}train',
        'val': f'{output_folder}val',
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
            bboxes, labels = get_label(model, grid_size)
            row = [f"{i}.png", f"{bboxes}", labels, binary_label]  # Default values for bounding_boxes, labels, and binary_label
            csvwriter.writerow(row)

def main(input_file, grid_size, output_folder):
    
    file = open(input_file)
    lines = file.readlines()

    if not os.path.exists(output_folder):    
        os.makedirs(output_folder)
        os.makedirs(f"{output_folder}/labels")
        
    all_models = []

    for line in lines:
        if 'obj' in line:
            all_models.append(line.strip('\n').split(' '))

    gen_yaml(CLASSES, output_folder)
    # gen_csv(all_models, CLASSES, output_folder, grid_size)
    
    for i, model in tqdm(enumerate(all_models), total=len(all_models)):
        gen_image(i, model, grid_size, output_folder)
        gen_label(i, model, grid_size, output_folder)




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--input_file', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-gs', '--grid_size', type=int, default=5)
    
    args = parser.parse_args()

    main(args.input_file, args.grid_size, args.output_folder)
