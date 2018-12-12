from glowingspoon.model.train import train
from glowingspoon.model.test import test
from glowingspoon.model.ConvNet import ConvNet
from glowingspoon.utility.utility import programme_parameters
import torch


def core():
    args = programme_parameters()

    if args.load_nn:
        print(f"Load model from {args.load_nn}")
        model = torch.load(args.load_nn)
        model.eval()
    else:
        model = ConvNet(args.size_block)

    if args.train:
        model = train(model, args)

    if args.test:
        test(model, args)


if __name__ == "__main__":
    core()
