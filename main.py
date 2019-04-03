import torch

from argparse import ArgumentParser


def setup_torch():
    print(f'Setting torch.device to {"cuda" if args.cuda else "cpu"}')
    if args.cuda:
        device = torch.device('cuda')

        if torch.has_cudnn:
            torch.backends.cudnn.enabled = True
            print(f'Enabled CuDNN {torch.backends.cudnn.version()}')
    else:
        device = torch.device('cpu')

    return device


if __name__ == '__main__':
    parser = ArgumentParser('PyTorch RNet')
    parser.add_argument('--cuda', action='store_true', help='Enable GPU training if available')
    parser.add_argument('--fp16', action='store_true', help='Enable mixed-precision training')

    # Arguments regarding the model
    parser.add_argument('--epochs', type=int, default=10, help='Number of full pass over the dataset to do')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0., help='Drop out probability')

    # Arguments regarding the optimizer
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.99, help='Adam first order momentum')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam second order momentum')
    parser.add_argument('--l2', type=float, default=0., help='L2 regularization factor')

    # Parse arguments
    args = parser.parse_args()

    # Enable cuda if requested and available
    args.cuda &= torch.cuda.is_available()

    # Setup torch
    args.device = setup_torch()
