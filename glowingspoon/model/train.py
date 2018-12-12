from glowingspoon.utility.utility import load_images, split_into_block
from glowingspoon.model.ImageDataset import ImageDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm


def train(model, args, batch_size=1000):

    X, y = load_images(args.train_x, args.train_y)

    new_X = split_into_block(X, model.size_block)
    new_y = split_into_block(y, model.size_block)

    # we reshape the matrix to match the neural network input.
    shape = new_X.shape
    new_X = torch.Tensor(new_X).reshape(shape[0], 1, shape[1], shape[2])
    new_y = torch.Tensor(new_y).reshape(shape[0], 1, shape[1], shape[2])

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

            y_pred = model(data["features"])

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
        torch.save(model, args.save_nn)

    return model
