from ml import train, test
from model import ConvNet
from utility import programme_parameters
import torch


def core():
    args = programme_parameters()
    # define to use cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvNet(args.size_block)
    if args.load_nn:
        print(f"Load model from {args.load_nn}")
        model.load_state_dict(torch.load(args.load_nn))
        model.eval()

    if args.train:
        model = train(model, args, device)

    if args.test:
        test(model, args, device)


if __name__ == "__main__":
    core()
