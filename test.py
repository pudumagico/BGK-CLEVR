import clingo
import re
from collections import defaultdict

def sort_objects(string):
    # Extract the individual objects using regex
    objects = re.findall(r'obj\((\d+),(\d+),(\w+),(\w+)\)', string)
    
    # Convert to a list of tuples with integers for the coordinates
    objects = [(int(x), int(y), shape, color) for x, y, shape, color in objects]
    
    # Sort the list by the first (x-coordinate) and second (y-coordinate) elements
    objects.sort(key=lambda obj: (obj[0], obj[1]))
    
    # Convert back to the original string format
    sorted_string = ', '.join([f'obj({x},{y},{shape},{color})' for x, y, shape, color in objects])
    
    return sorted_string

def element_to_set(element):
    # Convert each element's objects into a set of tuples
    element_set = {frozenset(obj) for obj in element}
    return element_set

def elements_to_unique_set(elements_list):
    # Convert each element to a frozenset (unordered, unique collection of objects)
    elements_set = {frozenset(element_to_set(element)) for element in elements_list}
    return elements_set

def generate_full_model_weak_constraint(model):
    constraint = ":~ " + ", ".join(f"obj({atom.arguments[0]},{atom.arguments[1]},{atom.arguments[2]},{atom.arguments[3]})"
                                   for atom in model if atom.name == "obj")
    constraint = sort_objects(constraint)
    constraint = ":~ " + constraint + '. [10]'
    # print(constraint)
    return constraint


def get_model(ctl):
    models = []
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            models.append(model.symbols(shown=True))
    return models[-1]

def generate_weak_constraints(model):
    constraints = []
    # print(model)
    for atom in model:
        if atom.name == "obj":
            X, Y, S, C = atom.arguments
            # Generate a weak constraint to avoid this specific atom
            constraints.append(f":~ obj({X},{Y},{S},{C}). [1@1]")
    return constraints

def run_asp_process(n):
    models = []
    weak_constraints = []
    for i in range(n):
        # print(weak_constraints)

        # Solve and get a model
        ctl = clingo.Control()

        # Add initial program to control object
        ctl.add("base", [], program)
        if weak_constraints:
            string_wc = "\n".join(weak_constraints)
            ctl.add("base", [], string_wc)

        # Ground the initial program
        ctl.ground([("base", [])])
        model = get_model(ctl)
        if not model:
            print("No model found")
            break

        # Output the model
        print(f"Model {i+1}: {model}")
        models.append(str(model))

        # Generate weak constraints to avoid this model
        weak_constraints.append(generate_full_model_weak_constraint(model))
        # print(weak_constraints)
        # for wc in weak_constraints_incumbent:
        #     weak_constraints.append(wc)
        # print(weak_constraints_incumbent)
        # weak_constraints = aggregate_weak_constraints(weak_constraints)
        # print(weak_constraints)
    # for wc in weak_constraints:
    #     print(wc)
    return models

# Initial ASP program
program = """
#const w=5.
#const h=5.

width(0..w-1).
heigth(0..h-1).

shape(triangle;circle;cross;square;diamond).
color(yellow;blue;green;red;purple).

{obj(X,Y,S,C) : shape(S), color(C)} = 1 :- width(X), heigth(Y).

#const c=1.
:- c=1, obj(X1, Y1, S, _), obj(X2, Y2, S, _), (X1, Y1) != (X2, Y2).

% Example 2 Converse: At least one pair of objects in the grid must have the same shape
some_same_shape :- obj(X1, Y1, S, _), obj(X2, Y2, S, _), (X1, Y1) != (X2, Y2).
:- c=0, not some_same_shape.

#show obj/4.
"""

# Run the ASP process N times
N = 1000
models = run_asp_process(N)

print(len(models))
models.sort()
models = set(models)
# unique_elements = elements_to_unique_set(models)
print(len(models))
