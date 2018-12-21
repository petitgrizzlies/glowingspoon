from utility import load_images, split_into_block
from model import ImageDataset
from torch.utils.data import DataLoader
import torch
from skimage.measure import compare_ssim as ssim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def train(model, args, device, batch_size=1000):

    X, y = load_images(args.train_x, args.train_y)

    new_X = split_into_block(X, model.size_block)
    new_y = split_into_block(y, model.size_block)

    # we reshape the matrix to match the neural network input.
    shape = new_X.shape
    new_X = torch.Tensor(new_X, device=device).reshape(shape[0], 1, shape[1], shape[2])
    new_y = torch.Tensor(new_y, device=device).reshape(shape[0], 1, shape[1], shape[2])

    # define the optimizer and the loss
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

    # create the dataset and batch loader
    image_dataset = ImageDataset(X=new_X, y=new_y)
    dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True)

    print(f"Training the neural network on the {X.shape[0]} training images")
    for epoch in tqdm(range(args.epochs)):
        running_loss = 0.0
        for index, data in enumerate(dataloader):

            y_pred = model(data["features"]).to(device)

            loss = criterion(y_pred, data["classe"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print mean loss for epoch
            running_loss += loss.item()
            print(f"{index} / {int(shape[0] / batch_size)}", end="\r")
        print(f"epoch : {epoch}, mean loss : {running_loss / shape[0]}")
        running_loss = 0.0

    if args.save_nn:
        torch.save(model.state_dict(), args.save_nn)

    return model


def test(model, args, device):
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
            torch.Tensor(splited_x, device=device).reshape(
                shape[0], 1, shape[1], shape[2]
            )
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
