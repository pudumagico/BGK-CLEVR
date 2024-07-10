import random

# Define the grid of objects
objects = [
    ("1,0", "cross", "pink"), ("2,0", "cross", "pink"), ("0,0", "square", "blue"),
    ("0,1", "cross", "pink"), ("2,1", "cross", "green"), ("1,1", "square", "pink"),
    ("1,2", "cross", "pink"), ("2,2", "cross", "pink"), ("0,2", "square", "pink")
]

def parse_position(pos):
    x, y = map(int, pos.split(','))
    return x, y

def is_left_of(pos1, pos2):
    x1, y1 = parse_position(pos1)
    x2, y2 = parse_position(pos2)
    return x1 < x2 and y1 == y2

def generate_random_question():
    obj1, obj2 = random.sample(objects, 2)
    pos1, shape1, color1 = obj1
    pos2, shape2, color2 = obj2

    question = f"Is the {shape1} {color1} object at position {pos1} to the left of the {shape2} {color2} object at position {pos2}?"
    answer = is_left_of(pos1, pos2)

    return question, answer

# Generate a set of random questions
def generate_questions(n=5):
    questions = []
    for _ in range(n):
        question, answer = generate_random_question()
        questions.append((question, answer))
    return questions

# Example usage
if __name__ == "__main__":
    questions = generate_questions(10)
    for question, answer in questions:
        print(question)
        print(f"Answer: {'Yes' if answer else 'No'}")
        print()
