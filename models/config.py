from transformers import AutoTokenizer, AutoModel

# special tokens indices in different models available in transformers
TOKEN_IDX = {
    'bert': {
        'START_SEQ': 101,
        'PAD': 0,
        'END_SEQ': 102,
        'UNK': 100
    },
    'xlm': {
        'START_SEQ': 0,
        'PAD': 2,
        'END_SEQ': 1,
        'UNK': 3
    },
    'roberta': {
        'START_SEQ': 0,
        'PAD': 1,
        'END_SEQ': 2,
        'UNK': 3
    },
    'albert': {
        'START_SEQ': 2,
        'PAD': 0,
        'END_SEQ': 3,
        'UNK': 1
    },
}


TOKENS = {
    'bert': {
        'START_SEQ': '[CLS]',
        'PAD': '[PAD]',
        'END_SEQ': '[SEP]',
        'UNK': '[UNK]'
    },
    'roberta': {
        'START_SEQ': '<s>',
        'PAD': '<pad>',
        'END_SEQ': '</s>',
        'UNK': '<unk>'
    },
}

# pretrained model name: (model class, model tokenizer, output dimension, token style)
MODELS = {
    'bert-base-multilingual-cased': (AutoModel, AutoTokenizer, 768, 'bert'),
    'bert-base-multilingual-uncased': (AutoModel, AutoTokenizer, 768, 'bert'),
    'xlm-mlm-en-2048': (AutoModel, AutoTokenizer, 2048, 'xlm'),
    'xlm-mlm-100-1280': (AutoModel, AutoTokenizer, 1280, 'xlm'),
    'distilbert-base-multilingual-cased': (AutoModel, AutoTokenizer, 768, 'bert'),
    'xlm-roberta-base': (AutoModel, AutoTokenizer, 768, 'roberta'),
    'xlm-roberta-large': (AutoModel, AutoTokenizer, 1024, 'roberta'),
    'monsoon-nlp/bangla-electra': (AutoModel, AutoTokenizer, 256, 'bert')
}
