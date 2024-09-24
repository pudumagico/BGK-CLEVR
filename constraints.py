import clingo

def get_model(ctl):
    models = []
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            models.append(model.symbols(atoms=True))
            break
    return models

def generate_full_model_weak_constraint(model):
    constraint = ":~ " + ", ".join(f"obj({atom.arguments[0]},{atom.arguments[1]},{atom.arguments[2]},{atom.arguments[3]})"
                                   for atom in model if atom.name == "obj")
    constraint += '. [1@1]'
    print(constraint)
    return constraint

def run_asp_process(ctl, n):
    for i in range(n):
        # Solve and get a model
        model = get_model(ctl)[0]
        if not model:
            print("No model found")
            break

        # Output the model
        print(f"Model {i+1}: {model}")

        # Generate a weak constraint to avoid this full model
        full_model_constraint = generate_full_model_weak_constraint(model)

        # Add the constraint to the control object
        ctl.add("base", [], full_model_constraint)

        # Ground the program again with the new constraints
        ctl.ground([("base", [])])

# Initial ASP program
program = """
#const w=5.
#const h=5.

width(0..w-1).
heigth(0..h-1).

shape(triangle;circle;cross;square;diamond).
color(yellow;blue;green;red;purple).

{obj(X,Y,S,C) : shape(S), color(C)} = 1 :- width(X), heigth(Y).

#show obj/4.
"""

# Initialize control object
ctl = clingo.Control()

# Add initial program to control object
ctl.add("base", [], program)

# Ground the initial program
ctl.ground([("base", [])])

# Run the ASP process N times
N = 5
run_asp_process(ctl, N)