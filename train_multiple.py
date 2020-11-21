import train
from utils import parse_args
import numpy as np


args = parse_args()

task_type_sentiment = False
if 'sentiment' in args.task:
    task_type_sentiment = True


NUM_RUNS = 10

acc_list = []
f1_list = []

for i in range(NUM_RUNS):
    args.seed = i + 1
    cls = train.Classification(args)
    acc, f1, _y_true, _y_pred = cls.run()

    acc_list.append(acc)
    f1_list.append(f1)

avg_acc = np.mean(acc_list)
std_acc = np.std(acc_list, ddof=1)
avg_f1 = np.mean(f1_list)
std_f1 = np.std(f1_list, ddof=1)

print('avg acc: {}, std acc: {}'.format(avg_acc, std_acc))
print('avg f1: {}, std f1: {}'.format(avg_f1, std_f1))

with open('result.txt', 'a') as f:
    f.write(str(args)+'\n')
    f.write('avg acc: {}, std acc: {}\n'.format(avg_acc, std_acc))
    f.write('avg f1: {}, std f1: {}\n'.format(avg_f1, std_f1))
