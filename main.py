import torch

from argparse import ArgumentParser
from torchtext.vocab import pretrained_aliases as word_emb
from data import setup_squad

TRAIN_METRIC_LABELS = ['loss', 'loss start', 'loss end', 'exact match', 'grad norm']
TEST_METRIC_LABELS = ['loss', 'loss start', 'loss end', 'exact match', 'f1-score']


def setup_torch():
    print(f'Setting torch.device to {"cuda" if args.cuda else "cpu"}')
    if args.cuda:
        device = torch.device('cuda')

        if torch.has_cudnn:
            args.fp16 &= True
            torch.backends.cudnn.enabled = True
            print(f'Enabled CuDNN {torch.backends.cudnn.version()}')
    else:
        device = torch.device('cpu')
        args.fp16 = False

    return device


def train():
    pass


def test():
    pass


if __name__ == '__main__':
    parser = ArgumentParser('PyTorch RNet')
    parser.add_argument('--cuda', action='store_true', help='Enable GPU training if available')
    parser.add_argument('--fp16', action='store_true', help='Enable mixed-precision training')
    parser.add_argument('--debug', action='store_true', help='Enable debugging')

    # Arguments regarding the model
    parser.add_argument('--epochs', type=int, default=10, help='Number of full pass over the dataset to do')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout probability')

    parser.add_argument('--word-embedding', default='glove.840B.300d', choices=word_emb.keys(),
                        help='Word embedding to use')

    parser.add_argument('--char-embedding', type=str, default=None, choices=[None], help='Char embedding to use')
    parser.add_argument('--char-embedding-training', action='store_true', help='Flag turning on char embedding training')
    parser.add_argument('--char-embedding-size', type=int, default=32,
                        help='Size of the char embedding (only relevant if --char-embedding=None)')

    parser.add_argument('--word-encoder-size', type=int, default=64, help='Size of the word encoder representation')
    parser.add_argument('--word-encoder-layers', type=int, default=3, help='Number of stacked word encoding layer (GRU)')

    # Arguments regarding the optimizer
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.99, help='Adam first order momentum')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam second order momentum')
    parser.add_argument('--l2', type=float, default=0., help='L2 regularization factor')

    # Arguments regarding data
    parser.add_argument('--train-path', type=str, help='Training data path')
    parser.add_argument('--test-path', type=str, help='Testing data path')

    # Parse arguments
    args = parser.parse_args()

    # Enable cuda if requested and available
    args.cuda &= torch.cuda.is_available()

    # Setup torch
    args.device = setup_torch()

    # Create train/test dataset along with embeddings
    squad_train, squad_test, words_v, chars_v = setup_squad(args)
