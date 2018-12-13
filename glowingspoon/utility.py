import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from distutils.util import strtobool


def boolean(x):
    return bool(strtobool(x))


def programme_parameters():
    """This function will parse the script argument and return a dictionnary with the parameters and their values"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Try to reconstruct original images from deteriorated one.",
    )

    parser.add_argument(
        "--train",
        help="If True, the neural will apply a training phase",
        default=True,
        type=boolean,
    )
    parser.add_argument(
        "--train_x",
        type=str,
        help="Give the path where the original images are located",
        default="../codes/original/",
    )
    parser.add_argument(
        "--train_y",
        type=str,
        help="Give the path where the modified images are located",
        default="../codes/modified/",
    )

    parser.add_argument(
        "--test",
        type=boolean,
        help="If True, tue neural network will be tested on image test.",
        default=False,
    )
    parser.add_argument(
        "--test_x",
        type=str,
        help="Give the path where the original images are located",
        default="../codes/original/",
    )
    parser.add_argument(
        "--test_y",
        type=str,
        help="Give the path where the modified images are located",
        default="../codes/modified/",
    )

    parser.add_argument(
        "--save_nn",
        type=str,
        help="If a value is given, the neural will be save at given path. Only work if there is a training",
    )
    parser.add_argument(
        "--load_nn",
        type=str,
        help="If a value is give, it will load the neural network from the path.",
    )

    parser.add_argument(
        "--epochs", type=int, help="Number of epochs in the traning phase.", default=1
    )
    parser.add_argument(
        "--size-block",
        type=int,
        help="The size of block for spliting images on training and test",
        default=6,
    )
    parser.add_argument(
        "--print_example",
        type=boolean,
        help="Display a random example of : modified, original and reconstructed images",
        default=True,
    )

    args = parser.parse_args()

    return args


def load_images(path_x, path_y):
    """Use matplotlib to load images from path as matrix and store it into X,y. This two variables will be returned"""
    # get the list of path images
    list_original = sorted(
        [path_x + image for image in os.listdir(path_x) if ".bmp" in image]
    )
    list_modified = sorted(
        [path_y + image for image in os.listdir(path_y) if ".bmp" in image]
    )

    # we want to reconstruct the originals from the modified, it implies that :
    #   - your features (X) are the modified
    #   - the classes (y) are the original
    X = []
    y = []
    # the following line is useless, it just allows a smooth printing from tqdm
    list_path = list(zip(list_modified, list_original))
    print("Loading images...")
    for modified_path, original_path in tqdm(list_path):
        X.append(plt.imread(modified_path))
        y.append(plt.imread(original_path))

    return np.array(X), np.array(y)


def split_into_block(matrix, size_block, unique=False):
    """It will split matrix into sub matrix of size : size_block.
    This function is very specifiy. You need :
        - a list a matrix
        - the matrix must be square
        - the matrix must be able to be divided by the size of block you specify
    """
    if unique:
        shape = [i for i in matrix.shape]
        shape = [1] + shape
    else:
        shape = matrix.shape

    ratio = int(shape[-1] / size_block)
    new_matrix = matrix.reshape(shape[0] * ratio ** 2, size_block, size_block)
    return new_matrix / 255
