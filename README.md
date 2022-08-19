# BGK-CLEVR

BKG-CLEVR is a 2D CLEVR-like dataset generator that uses Answer Set Programming as the engine for generating positive and negative examples.
These examples are divided between those who satisfy a given constraints and those who do not.
Furthermore, a neural network model is provided to test the capacity of neural approaches to learn the aforementioned constraint.

# Installation

Clone this repository:

`git clone https://github.com/pudumagico/BGK-CLEVR.git`

Create an environment with all packages from requirements.txt

`conda create --name bkg-clevr -c potassco conda-forge pytorch --file requirements.txt`

`source activate bkg-clevr`

# Use

The pipeline to use this framework consists of three steps:

## Encoding a constraint

We provide a main encoding that positions objects of a certain shape and color in a grid.
The size of the grid can be changed trough the use of constants.

Aside from this the user needs to input a constraint on top of the generated answer sets.
A constraint example is provided, which says that all objects in an horizontal line must be the same.
This constraint comes with is counterpart, which says that at least two objects must be of different shape in an horizontal line.
This will come in handy when we want to generate positive and negative examples.

Concretely, use the following code to generate the positive and negative answer sets:

`clingo main_encoding.lp constraints/horizontal_shape.lp -c c=1 10 > asp_models/horizontal_shape_positive.txt`

`clingo main_encoding.lp constraints/horizontal_shape.lp -c c=0 10 > asp_models/horizontal_shape_negative.txt`

Where the constant `c` enforces the positive contraint when is equal to 1 and enforces the negative constraint when is equal to 0.

## Creating a dataset

To generate use the following code

`python generate_images.py -i asp_models/horizontal_shape_positive.txt -o ./images/horizontal_shape/positive -gs 3`

`python generate_images.py -i asp_models/horizontal_shape_negative.txt -o ./images/horizontal_shape/negative -gs 3`

Where `-i` takes a text file with the answer sets inn it, `-o` is the output folder for the images and `-gs` is the grid size.

Finally we have to merge these two folders and create the annotations file which contains the name of the image and the label.
This is done with the followinng command:

`python create_dataset.py -i images/horizontal_shape -o ./datasets/horizontal_color`

## Training and Testing

For this use the following command:

`python train_and_test.py -d datasets/horizontal_color`
 