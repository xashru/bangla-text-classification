# bangla-text-classification
This repository contains implementation for [Bangla Text Classification using Transformers
](https://arxiv.org/abs/2011.04446)

### Training
Currently we have support for reading data from `pickle` format. The data is expected to contain a list of list (or tuples), where the first element is the text and the second element is the label. Check `utils.py` for task specific configurations. We have provided data for the `youtube-emotion` and `sentiment` task. Other will be provided later. 

Example command for training on `youtube-emotion` task
`python train_multiple.py --model=xlm-roberta-base --task=yt-emotion --save-path='logs/yt-emotion/xlm-base'`

If you want to train on a new dataset, prepare train/val/test `pickle` files in the required format and update the utils file to add other task-specific configurations.

### Cite this work
If you find this repository helpful in your work please cite the following
```
@article{alam2020bangla,
  title={Bangla Text Classification using Transformers},
  author={Alam, Tanvirul and Khan, Akib and Alam, Firoj},
  journal={arXiv preprint arXiv:2011.04446},
  year={2020}
}
```
