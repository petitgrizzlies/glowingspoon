from glowingspoon.utility.utility import load_images, split_into_block
import torch
from skimage.measure import compare_ssim as ssim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def test(model, args):
    X, y = load_images(args.test_x, args.test_y)
    y = y / y.max().astype(np.float32)

    overall_accuracy = 0.0
    overall_improvement = 0.0
    shape_image = X[0].shape

    print(f"Testing the {X.shape[0]} testing images...")
    for small_x, small_y in tqdm(list(zip(X, y))):
        # we split the image
        splited_x = split_into_block(small_x, model.size_block, unique=True)
        shape = splited_x.shape
        # apply model
        guessed_y = model(
            torch.Tensor(splited_x).reshape(shape[0], 1, shape[1], shape[2])
        ).reshape(shape_image[0], shape_image[0])
        # get the new image
        guessed_y = guessed_y.detach().numpy()
        # compute structural similarity and structural improvement
        diff = ssim(guessed_y, small_y)
        overall_accuracy += diff
        overall_improvement += np.abs(diff - ssim(small_x.astype(np.float32), small_y))

    print(
        f"Overall structural similarity on test images: {overall_accuracy / X.shape[0]}\n"
        f"Overall structural improvement between modified to reconstruct : {overall_improvement / X.shape[0]}"
    )

    if args.print_example:
        example = np.random.randint(X.shape[0])
        plt.subplot(1, 3, 1)
        plt.imshow(X[example], cmap="Greys")
        plt.title("Modified")
        plt.xlabel(
            f"Similarity with original = {ssim(X[example].astype(np.float32), y[example])}"
        )

        plt.subplot(1, 3, 2)
        plt.imshow(y[example] / y[example].max(), cmap="Greys")
        plt.title("Original")

        plt.subplot(1, 3, 3)
        splited_x = split_into_block(X[example], model.size_block, unique=True)
        shape = splited_x.shape
        guessed_y = model(
            torch.Tensor(splited_x).reshape(shape[0], 1, shape[1], shape[2])
        ).reshape(shape_image[0], shape_image[0])
        guessed_y = guessed_y.detach().numpy()

        plt.imshow(guessed_y, cmap="Greys")
        plt.title("Reconstructed")
        plt.xlabel(f"Similarity with original = {ssim(guessed_y, y[example])}")

        plt.show()
