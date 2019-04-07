from torchtext.data import get_tokenizer
from torchtext.vocab import pretrained_aliases

from .squad import SquadDataset, SquadDebugDataset


def setup_squad(args):
    # Load spacy tokenizer
    args.tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

    # Retrieve the embeddings vocabs
    words_v = pretrained_aliases[args.word_embedding]()
    chars_v = None

    # Debug allow to load only a subset of the SQuAD dataset
    if args.debug:
        print('/!\\ Debug mode enabled, loading a subset of SQuAD /!\\')
        Squad = SquadDebugDataset
    else:
        Squad = SquadDataset

    # Load squad
    squad_train, squad_fields = Squad(args.train_path, args.tokenizer).to_torchtext(args.tokenizer, return_fields=True)
    squad_test = Squad(args.test_path, args.tokenizer).to_torchtext(args.tokenizer, fields=squad_fields)

    # Create vocab (content_w and question_w are the same field, so building on one of them will share the vocab)
    # same for content_c and question_c
    squad_train.fields['context_w'].build_vocab(squad_train, squad_test, vectors=words_v)
    squad_train.fields['context_c'].build_vocab(squad_train, squad_test, vectors=chars_v)

    return squad_train, squad_test, words_v, chars_v


def sort_by_lengths(x):
    return -len(x.context_w)