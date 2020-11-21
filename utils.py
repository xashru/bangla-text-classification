import argparse
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description='Aspect based sentiment classification')
    parser.add_argument('--task', default='sentiment', type=str, help='Task name')
    parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'), help='use cuda if available')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
    parser.add_argument('--decay', default=0., type=float, help='weight decay')
    parser.add_argument('--model', default="bert-base-multilingual-cased", type=str, help='pretrained BERT model name')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size (default: 64)')
    parser.add_argument('--epoch', default=10, type=int, help='total epochs (default: 200)')
    parser.add_argument('--fine-tune', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to fine-tune embedding or not')
    parser.add_argument('--save-path', default='out', type=str, help='output log/result directory')
    args = parser.parse_args()
    return args


def load_pickle(file_name):
    with open(file_name, 'rb') as handle:
        return pickle.load(handle)


class TaskConfig:
    num_class = None
    train_file = None
    val_file = None
    test_file = None
    sequence_len = None
    eval_interval = None
    patience = None
    balance = None


def get_task_config(task_name):
    config = TaskConfig()
    if task_name == 'sentiment':
        config.num_class = 2
        config.train_file = 'data/sentiment/train.pickle'
        config.val_file = 'data/sentiment/validation.pickle'
        config.test_file = 'data/sentiment/test.pickle'
        config.sequence_len = 100
        config.eval_interval = 100
        config.patience = 10
        config.balance = False
    elif task_name == 'news':
        config.num_class = 6
        config.train_file = 'data/news/train.pickle'
        config.val_file = 'data/news/validation.pickle'
        config.test_file = 'data/news/test.pickle'
        config.sequence_len = 300
        config.eval_interval = 200
        config.patience = 10
        config.balance = False
    elif task_name == 'authorship':
        config.num_class = 14
        config.train_file = 'data/authorship/train.pickle'
        config.val_file = 'data/authorship/validation.pickle'
        config.test_file = 'data/authorship/test.pickle'
        config.sequence_len = 300
        config.eval_interval = 200
        config.patience = 10
        config.balance = True
    elif task_name == 'yt-sent-3':
        config.num_class = 3
        config.train_file = 'data/youtube-sentiment-3/train.pickle'
        config.val_file = 'data/youtube-sentiment-3/validation.pickle'
        config.test_file = 'data/youtube-sentiment-3/test.pickle'
        config.sequence_len = 50
        config.eval_interval = 100
        config.patience = 30
        config.balance = False
    elif task_name == 'yt-sent-5':
        config.num_class = 5
        config.train_file = 'data/youtube-sentiment-5/train.pickle'
        config.val_file = 'data/youtube-sentiment-5/validation.pickle'
        config.test_file = 'data/youtube-sentiment-5/test.pickle'
        config.sequence_len = 50
        config.eval_interval = 30
        config.patience = 30
        config.balance = False
    elif task_name == 'yt-emotion':
        config.num_class = 5
        config.train_file = 'data/youtube-emotion/train.pickle'
        config.val_file = 'data/youtube-emotion/validation.pickle'
        config.test_file = 'data/youtube-emotion/test.pickle'
        config.sequence_len = 50
        config.eval_interval = 30
        config.patience = 30
        config.balance = False
    else:
        raise ValueError('Task not supported')
    return config
