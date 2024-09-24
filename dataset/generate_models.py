import random
import itertools
import os 

import torch

from tqdm import tqdm

COLORS = {
    'red': [230/255, 124/255, 115/255],
    'yellow': [247/255, 203/255, 77/255],
    'green': [65/255, 179/255, 117/255],
    'blue': [123/255, 170/255, 247/255],
    'purple': [186/255, 103/255, 200/255]
}

SHAPES = ['triangle', 'circle', 'cross', 'square', 'diamond']

CLASSES = list(' '.join(e) for e in itertools.product(SHAPES, COLORS))
CLASSES = dict(enumerate(CLASSES))
digit_mapping = inv_map = {v: k for k, v in CLASSES.items()}

# def generate_model(grid_size):
#     model = []
#     for i in range(grid_size):
#         for j in range(grid_size):
#             shape = random.choice(SHAPES)
#             color = random.choice(list(COLORS.keys()))
#             model.append(f"obj({i},{j},{shape},{color})")
#     return model

# # Example usage
# grid_size = 3
# model = generate_model(grid_size)
# print(" ".join(model))


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
                    for y in range(grid_size):
                        colors = {model[(x, y)][1] for x in range(grid_size)}
                        if specific_color not in colors:
                            return False
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

        shape_counts = {shape: total_cells // num_shapes for shape in SHAPES}
        color_counts = {color: total_cells // num_colors for color in COLORS}

        for _ in range(total_cells % num_shapes):
            shape_counts[random.choice(SHAPES)] += 1

        for _ in range(total_cells % num_colors):
            color_counts[random.choice(list(COLORS.keys()))] += 1

        cells = [(i, j) for i in range(grid_size) for j in range(grid_size)]
        random.shuffle(cells)

        for i, j in cells:
            shape = random.choices(list(shape_counts.keys()), weights=list(shape_counts.values()))[0]
            color = random.choices(list(color_counts.keys()), weights=list(color_counts.values()))[0]
            
            model[(i, j)] = (shape, color)

            shape_counts[shape] -= 1
            if shape_counts[shape] == 0:
                del shape_counts[shape]

            color_counts[color] -= 1
            if color_counts[color] == 0:
                del color_counts[color]

        return model

    def model_to_string(model):
        return " ".join([f"obj({i},{j},{shape},{color})" for (i, j), (shape, color) in model.items()])

    model = generate_random_model(grid_size)
    model_str = model_to_string(model)

    while not is_valid(model, grid_size, constraints, constraint_flags) or model_str in existing_models:
        model = generate_random_model(grid_size)
        model_str = model_to_string(model)

    existing_models.add(model_str)
    return model_str

# Example usage
grid_size = 3
constraints = ["same_color_in_middle_column"]  # List of constraints
constraint_flags = [1]  # Set to 1 to follow the constraint, 0 to break the constraint
existing_models = set()

from generate_images import gen_image, gen_csv
from generate_matrices import parse_input_string

positive_folder = os.path.exists('./test_dataset/1')
negative_folder = os.path.exists('./test_dataset/0')
if not positive_folder:
    os.mkdir('./test_dataset/1', 0o777) 
if not negative_folder:
    os.mkdir('./test_dataset/0', 0o777) 

# Generate multiple models
positive_models = []
for i, _ in enumerate(tqdm(range(100))):
    model_str = generate_uniform_model(grid_size, constraints, constraint_flags, existing_models)
    # print(model_str)
    positive_models.append(model_str)
    gen_image(i, model_str.split(' '), grid_size, './test_dataset/1')

gen_csv(positive_models, CLASSES, './test_dataset/1', 1, grid_size)    

positive_matrices = [parse_input_string(input_string) for input_string in positive_models]
positive_tensor = torch.tensor(positive_matrices).unsqueeze(1).float()
torch.save(positive_tensor, './test_dataset/1/matrices.pt')

existing_models = set()
constraint_flags = [0]  # Now generate models that break the constraint
negative_models = []
for i, _ in enumerate(tqdm(range(100))):
    model_str = generate_uniform_model(grid_size, constraints, constraint_flags, existing_models)
    # print(model_str)
    negative_models.append(model_str)
    gen_image(i, model_str.split(' '), grid_size, './test_dataset/0')

gen_csv(negative_models, CLASSES, './test_dataset/0', 0, grid_size)    
negative_matrices = [parse_input_string(input_string) for input_string in negative_models]
negative_tensor = torch.tensor(negative_matrices).unsqueeze(1).float()
torch.save(negative_tensor, './test_dataset/0/matrices.pt')
