import torch

from argparse import ArgumentParser

from apex import amp
from torch import Tensor
from torch.nn.functional import cross_entropy, softmax
from torch.optim import Adam
from torchtext.data import BucketIterator
from torchtext.vocab import pretrained_aliases as word_emb
from tqdm import trange

from data import setup_squad, sort_by_lengths
from data.evaluate import evaluate
from data.squad import convert_idx
from models import normalize_predictions, normalize_prediction
from models.rnet import RNet


TRAIN_METRIC_LABELS = ['loss', 'loss start', 'loss end', 'exact match', 'grad norm']
TEST_METRIC_LABELS = ['loss', 'loss start', 'loss end', 'exact match', 'f1-score']


def setup_torch(device_ord: int):
    print(f'Setting torch.device to {"cuda" if args.cuda else "cpu"}')
    if args.cuda:
        device = torch.device(f'cuda:{device_ord}')
        device_props = torch.cuda.get_device_properties(device)

        if torch.has_cudnn:
            args.fp16 &= True
            torch.backends.cudnn.enabled = True

            print(f'Enabled CuDNN {torch.backends.cudnn.version()}')
            print(
                f'Device used: '
                f'{torch.cuda.get_device_name(device)} '
                f'(capabilities: {device_props.major}.{device_props.minor}, '
                f'memory: {round(device_props.total_memory / 1024 ** 3, 2)}Gb)'
            )

    else:
        device = torch.device('cpu')
        args.fp16 = False

        print('Device used: cpu')

    return device


def train(iterator, model, optimizer):
    # Ensure model is in training mode
    model.train()

    # Metrics accumulator
    metrics = [[] for _ in TRAIN_METRIC_LABELS]

    for batch in iterator:
        # Reset gradients
        optimizer.zero_grad()

        # Retrieve (tokens, nb_tokens)
        doc_w, que_w = batch.context_w, batch.question_w
        doc_c, que_c = batch.context_c, batch.question_c

        # The field is tuple storing (word_indexes, nb_words)
        (doc_w, doc_w_l), (que_w, que_w_l) = doc_w, que_w
        (doc_c, doc_c_l), (que_c, que_c_l) = (doc_c[0], doc_c[-1]), (que_c[0], que_c[-1])

        # Extract answer start & end
        answer_start, answer_end = batch.answer_start, batch.answer_end

        # Forward through model
        p_pred_start, p_pred_end = model(doc_w, doc_c, doc_w_l, que_w, que_c, que_w_l)

        # Compute loss
        loss_start, loss_end = cross_entropy(p_pred_start, answer_start), cross_entropy(p_pred_end, answer_end)

        # Combine losses
        loss = loss_start + loss_end

        # Backward & optimizer step
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
        optimizer.step()

        # Compute accuracy over start and end
        with torch.no_grad():
            p_pred_start_, p_pred_end_ = softmax(p_pred_start, 1), softmax(p_pred_end, 1)
            p_pred_start_, p_pred_end_ = normalize_predictions(p_pred_start_, p_pred_end_)
            p_pred_start_, p_pred_end_ = torch.tensor(p_pred_start_, device=loss.device), \
                                         torch.tensor(p_pred_end_, device=loss.device)

            exact_correct_num = torch.sum((p_pred_start_ == answer_start) * (p_pred_end_ == answer_end))
            em = exact_correct_num.float() / float(p_pred_start.size(0))

            # ['loss', 'loss start', 'loss end', 'exact match', 'grad norm']
            for idx, value in enumerate((loss, loss_start, loss_end, em, grad_norm)):
                if isinstance(value, float):
                    metrics[idx] += [value]
                elif isinstance(value, Tensor):
                    metrics[idx] += [value.tolist()]
                else:
                    metrics[idx] += value

    return tuple(metric for metric in metrics)


def test(iterator, model):
    # Ensure model is in training mode
    model.eval()

    # Metrics accumulator
    metrics, test_preds = [[] for _ in TEST_METRIC_LABELS], {}

    # Disable gradients and iterate
    with torch.no_grad():
        for batch in iterator:

            # Retrieve (question id and context)
            qids, context = batch.id, batch.context

            # Retrieve (tokens, nb_tokens)
            doc_w, que_w = batch.context_w, batch.question_w
            doc_c, que_c = batch.context_c, batch.question_c

            # The field is tuple storing (word_indexes, nb_words)
            (doc_w, doc_w_l), (que_w, que_w_l) = doc_w, que_w
            (doc_c, doc_c_l), (que_c, que_c_l) = (doc_c[0], doc_c[-1]), (que_c[0], que_c[-1])

            # Extract answer start & end
            answer_start, answer_end = batch.answer_start, batch.answer_end

            # Forward through model
            p_pred_start, p_pred_end = model(doc_w, doc_c, doc_w_l, que_w, que_c, que_w_l)

            # Compute loss
            loss_start, loss_end = cross_entropy(p_pred_start, answer_start), cross_entropy(p_pred_end, answer_end)

            # Combine losses
            loss = loss_start + loss_end

            # Compute accuracy over start and end
            p_pred_start_, p_pred_end_ = softmax(p_pred_start, 1), softmax(p_pred_end, 1)

            # Generate prediction for eval script
            for qid, ctx, doc, p_s, p_e in zip(qids, context, doc_w, p_pred_start_, p_pred_end_):

                # Retrieve word start/end indexes
                p_s_, p_e_ = normalize_prediction(p_s, p_e)

                # Map them back to the origin text space through the word-span space
                spans = convert_idx(ctx, (words_v.itos[w] for w in doc if w > 1))

                # It happens that p_e_ predict padded words ...
                p_c_s, p_c_e = spans[p_s_][0], spans[p_e_][-1]
                test_preds[qid] = ctx[p_c_s:p_c_e]

            with open(args.test_path, 'r') as test_f:
                from ujson import load
                em, f1 = evaluate(load(test_f)['data'], test_preds).values()

            # Accumulate
            for idx, value in enumerate((loss, loss_start, loss_end, em, f1)):
                if isinstance(value, float):
                    metrics[idx] += [value]
                elif isinstance(value, Tensor):
                    metrics[idx] += [value.tolist()]
                else:
                    metrics[idx] += value

    return tuple(metric for metric in metrics)


if __name__ == '__main__':
    parser = ArgumentParser('PyTorch RNet')

    # Global program variables
    parser.add_argument('--cuda', action='store_true', help='Enable GPU training if available')
    parser.add_argument('--device', type=int, default=0, help='CUDA device index to use')
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

    parser.add_argument('--attention-size', type=int, default=64, help='Size of the attention projectors')

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
    args.device = setup_torch(args.device)

    # Create train/test dataset along with embeddings
    squad_train, squad_test, words_v, chars_v = setup_squad(args)
    squad_train_it = BucketIterator(squad_train, args.batch, device=args.device, train=True, sort_key=sort_by_lengths)
    squad_test_it = BucketIterator(squad_test, args.batch, device=args.device, train=False, sort_key=sort_by_lengths)

    args.epochs = trange(args.epochs)

    # Setup model
    rnet = RNet(
        words_v.vectors, args.word_encoder_size, args.word_encoder_layers,
        args.attention_size, args.dropout
    ).to(args.device)

    # Setup optimizer
    optimizer = Adam(rnet.parameters(), args.lr, betas=(args.beta1, args.beta2), weight_decay=args.l2)

    # Setup Mixed-Precision training if needed, otherwise this op is a pass-through
    rnet, optimizer = amp.initialize(rnet, optimizer, enabled=args.fp16, opt_level='O1')

    # Train for the specified number of epoch
    for epoch in args.epochs:

        # Train
        training_metrics = train(squad_train_it, rnet, optimizer)

        # Test
        testing_metrics = test(squad_test_it, rnet)

        args.epochs.set_description(
            f'Loss: { round(sum(testing_metrics[0]) / len(testing_metrics[0]), 2) } | '
            f'F1: { round(sum(testing_metrics[-1]) / len(testing_metrics[-1]), 2) }'
        )

        # Disable benchmarking after first epoch
        torch.backends.cudnn.benchmark = False
