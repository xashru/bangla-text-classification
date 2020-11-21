import train
from utils import parse_args
import numpy as np
import sklearn


args = parse_args()

task_type_sentiment = False
if 'sentiment' in args.task:
    task_type_sentiment = True

y_true = None
y_pred = None

NUM_RUNS = 10

for _ in range(NUM_RUNS):
    cls = train.Classification(args)
    acc, f1, _y_true, _y_pred = cls.run()

    if y_true is None:
        y_true = _y_true
        y_pred = _y_pred
    else:
        y_true = np.concatenate([y_true, _y_true])
        y_pred = np.concatenate([y_pred, _y_pred])

print('\n')
print(str(args))

pr, re, f1, _ = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average='micro')
print('precision: {}, recall: {}, f1: {}'.format(pr, re, f1))

cr = sklearn.metrics.classification_report(y_true, y_pred)

print(cr)


with open('result.txt', 'a') as f:
    f.write(str(args)+'\n')
    f.write(str(pr) + ' ' + str(re) + ' ' + str(f1) + '\n')
    f.write(str(cr) + '\n\n')
