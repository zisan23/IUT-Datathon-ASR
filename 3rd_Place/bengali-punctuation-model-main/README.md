# Bengali Punctuation Model
The model we will train will infer punctuation marks between each word for Bengali sentences. Complete the following three steps to train the model. Also, please check each notebook for details on each step.
## 1. Preparing Dataset
In this notebook, we will clean Bengali sentences. Please prepare train.csv (given data) with fold column in advance. In this notebook, "train_with_fold.csv" is the file.

[bengaliai-punctuation-model-prepare.ipynb](https://github.com/espritmirai/bengali-punctuation-model/blob/main/bengaliai-punctuation-model-prepare.ipynb)
## 2. Splitting Dataset
This notebook formats Bengali sentences into training data.

[bengaliai-punctuation-model-split.ipynb](https://github.com/espritmirai/bengali-punctuation-model/blob/main/bengaliai-punctuation-model-split.ipynb)
## 3. Training
This notebook trains the punctuation model. The training dataset after steps 1 and 2 can also be found at [here](https://www.kaggle.com/datasets/takuji/punctuation-model-dataset).

[bengaliai-punctuation-model-train.ipynb](https://github.com/espritmirai/bengali-punctuation-model/blob/main/bengaliai-punctuation-model-train.ipynb)
## Remarks
* GPU I used: single RTX 4090
* Training time: 46 hours/epoch
