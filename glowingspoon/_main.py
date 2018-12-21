from ml import train, test
from model import ConvNet
from utility import programme_parameters
import torch


def core():
    args = programme_parameters()

    model = ConvNet(args.size_block)
    if args.load_nn:
        print(f"Load model from {args.load_nn}")
        model.load_state_dict(torch.load(args.load_nn))
        model.eval()

    if args.train:
        model = train(model, args)

    if args.test:
        test(model, args)


if __name__ == "__main__":
    core()
