# ENRL

This is the implementation for
*Explainable Neural Rule Learning.  In Proceedings of The ACM Web Conference 2022 (TheWebConf ’22).*



## Environments

[`./requirements.txt`] - The codes can be successfully run with following packages in an Anaconda environment:

```
tqdm
treelib
scipy
torchmetrics==0.3.2
py3nvml
pytorch-lightning==1.3.1
numpy
pytorch==1.7.1
pandas
jsonlines
matplotlib
PyYAML
scikit-learn
```

Other settings with `pytorch>=1.3.1` may also work.



## Datasets

The processed datasets can be downloaded from [].

You should place the datasets in the `./dataset/`. The tree structure of directories should look like:

```
.
├── dataset
│   ├── Adult
│   ├── Credit
│   ├── RSC2017
│   └── Synthetic
├── preprocess
├── log
├── enrl
├── main.py
└── predict.py
```

-   **Adult**: The origin dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/Adult).
-   **Credit**: The origin dataset can be found [here](https://www.kaggle.com/c/GiveMeSomeCredit).
-   **RSC2017**: The origin dataset can be found [here](http://www.recsyschallenge.com/2017/).
-   The codes preprocessing are in [`./src/preprocess`] for reference.



## Examples to run the code

-   Some running commands can be found in [`./command/command.py`]
-   For example:

```
# ENRL on Synthetic dataset
> cd ENRL/enrl/
> python main.py --model_name ENRL --dataset Synthetic --rule_len 5 --rule_n 40 --es_patience 200 --op_loss 1 --cuda 0
```


## Cite
If you find this repository useful for your research or development, please cite the following paper:
```
@inproceedings{shi2021explainable,
    title = "Explainable Neural Rule Learning",
    author = "Shi, Shaoyun and Xie, Yuexiang  and Wang, Zhen and Ding, Bolin and Li, Yaliang and Zhang, Min",
    booktitle = "Proceedings of The Web Conference 2022",
    year = "2022"
}
