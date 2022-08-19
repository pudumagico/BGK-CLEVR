import argparse
import math
import os
import sys
import re

import cairo

PIXEL_SCALE = 512
BKG_COLOR = [148/256,146/256,147/256]

COLORS = {
'pink': [240/256,161/256,211],
'blue': [159/256,223/256,244/256],
'red': [242/256,133/256,119/256],
'green': [102/256,226/256,219/256]
}

DIFF = 0.35

def gen_image(i, model, grid_size, output_folder):

    if not os.path.exists(output_folder):    
        os.makedirs(output_folder)

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

        if shape == 'circle':
            ctx.arc(cx, cy, DIFF, 0, 2*math.pi)

        if shape == 'cross':

            ctx.move_to(cx - DIFF, cy - DIFF)
            ctx.line_to(cx + DIFF, cy + DIFF)

            ctx.move_to(cx - DIFF, cy + DIFF)
            ctx.line_to(cx + DIFF, cy - DIFF)

            ctx.set_source_rgb(*COLORS[color])
            ctx.set_line_width(0.05)
            ctx.stroke()

        if shape == 'triangle':
            
            h = DIFF*math.sqrt(3)/2

            ctx.move_to(cx - DIFF, cy + h)
            ctx.line_to(cx + DIFF, cy + h)
            ctx.line_to(cx, cy - h/2 - 0.4*DIFF)

        ctx.set_source_rgb(*COLORS[color])
        ctx.fill()


    surface.write_to_png(output_folder + '/' + str(i) + '.png')

def main(input_file, grid_size, output_folder):
    
    file = open(input_file)
    lines = file.readlines()

    all_models = []

    for line in lines:
        if 'obj' in line:
            all_models.append(line.strip('\n').split(' '))

    for i, model in enumerate(all_models):
        gen_image(i, model, grid_size, output_folder)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--input_file', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-gs', '--grid_size', type=int, required=True)
    
    args = parser.parse_args()

    main(args.input_file, args.grid_size, args.output_folder)
