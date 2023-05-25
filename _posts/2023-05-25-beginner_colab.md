##### Copyright 2022 The TensorFlow Authors.


```python
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Build, train and evaluate models with TensorFlow Decision Forests

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/decision_forests/tutorials/beginner_colab"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/decision-forests/blob/main/documentation/tutorials/beginner_colab.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/decision-forests/blob/main/documentation/tutorials/beginner_colab.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View on GitHub</a>
  </td>
  <td>
    <a href="https://storage.googleapis.com/tensorflow_docs/decision-forests/documentation/tutorials/beginner_colab.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>
</table>


## Introduction

Decision Forests (DF) are a family of Machine Learning algorithms for
supervised classification, regression and ranking. As the name suggests, DFs use
decision trees as a building block. Today, the two most popular DF training
algorithms are [Random Forests](https://en.wikipedia.org/wiki/Random_forest) and
[Gradient Boosted Decision Trees](https://en.wikipedia.org/wiki/Gradient_boosting).

TensorFlow Decision Forests (TF-DF) is a library for the training,
evaluation, interpretation and inference of Decision Forest models.

In this tutorial, you will learn how to:

1.  Train a multi-class classification Random Forest on a dataset containing numerical, categorical and missing features.
1.  Evaluate the model on a test dataset.
1.  Prepare the model for
    [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving).
1.  Examine the overall structure of the model and the importance of each feature.
1.  Re-train the model with a different learning algorithm (Gradient Boosted Decision Trees).
1.  Use a different set of input features.
1.  Change the hyperparameters of the model.
1.  Preprocess the features.
1.  Train a model for regression.

Detailed documentation is available in the [user manual](https://github.com/tensorflow/decision-forests/tree/main/documentation).
The [example directory](https://github.com/tensorflow/decision-forests/tree/main/examples) contains other end-to-end examples.

## Installing TensorFlow Decision Forests

Install TF-DF by running the following cell.


```python
!pip install tensorflow_decision_forests
```

    Collecting tensorflow_decision_forests
      Using cached tensorflow_decision_forests-1.3.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.4 MB)
    Requirement already satisfied: numpy in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from tensorflow_decision_forests) (1.24.3)
    Requirement already satisfied: pandas in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from tensorflow_decision_forests) (2.0.1)
    Collecting tensorflow~=2.12.0 (from tensorflow_decision_forests)
      Using cached tensorflow-2.12.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (585.9 MB)
    Requirement already satisfied: six in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from tensorflow_decision_forests) (1.16.0)
    Requirement already satisfied: absl-py in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from tensorflow_decision_forests) (1.4.0)
    Requirement already satisfied: wheel in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from tensorflow_decision_forests) (0.38.4)
    Collecting wurlitzer (from tensorflow_decision_forests)
      Using cached wurlitzer-3.0.3-py3-none-any.whl (7.3 kB)
    Requirement already satisfied: astunparse>=1.6.0 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from tensorflow~=2.12.0->tensorflow_decision_forests) (1.6.3)
    Requirement already satisfied: flatbuffers>=2.0 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from tensorflow~=2.12.0->tensorflow_decision_forests) (23.5.9)
    Requirement already satisfied: gast<=0.4.0,>=0.2.1 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from tensorflow~=2.12.0->tensorflow_decision_forests) (0.4.0)
    Requirement already satisfied: google-pasta>=0.1.1 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from tensorflow~=2.12.0->tensorflow_decision_forests) (0.2.0)
    Requirement already satisfied: grpcio<2.0,>=1.24.3 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from tensorflow~=2.12.0->tensorflow_decision_forests) (1.55.0)
    Requirement already satisfied: h5py>=2.9.0 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from tensorflow~=2.12.0->tensorflow_decision_forests) (3.8.0)
    Collecting jax>=0.3.15 (from tensorflow~=2.12.0->tensorflow_decision_forests)
      Using cached jax-0.4.10.tar.gz (1.3 MB)
      Installing build dependencies ... [?25ldone
    [?25h  Getting requirements to build wheel ... [?25ldone
    [?25h  Preparing metadata (pyproject.toml) ... [?25ldone
    [?25hCollecting keras<2.13,>=2.12.0 (from tensorflow~=2.12.0->tensorflow_decision_forests)
      Using cached keras-2.12.0-py2.py3-none-any.whl (1.7 MB)
    Requirement already satisfied: libclang>=13.0.0 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from tensorflow~=2.12.0->tensorflow_decision_forests) (16.0.0)
    Collecting numpy (from tensorflow_decision_forests)
      Using cached numpy-1.23.5-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (17.1 MB)
    Requirement already satisfied: opt-einsum>=2.3.2 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from tensorflow~=2.12.0->tensorflow_decision_forests) (3.3.0)
    Requirement already satisfied: packaging in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from tensorflow~=2.12.0->tensorflow_decision_forests) (23.1)
    Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from tensorflow~=2.12.0->tensorflow_decision_forests) (4.23.1)
    Requirement already satisfied: setuptools in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from tensorflow~=2.12.0->tensorflow_decision_forests) (67.8.0)
    Collecting tensorboard<2.13,>=2.12 (from tensorflow~=2.12.0->tensorflow_decision_forests)
      Using cached tensorboard-2.12.3-py3-none-any.whl (5.6 MB)
    Collecting tensorflow-estimator<2.13,>=2.12.0 (from tensorflow~=2.12.0->tensorflow_decision_forests)
      Using cached tensorflow_estimator-2.12.0-py2.py3-none-any.whl (440 kB)
    Requirement already satisfied: termcolor>=1.1.0 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from tensorflow~=2.12.0->tensorflow_decision_forests) (2.3.0)
    Requirement already satisfied: typing-extensions>=3.6.6 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from tensorflow~=2.12.0->tensorflow_decision_forests) (4.6.0)
    Requirement already satisfied: wrapt<1.15,>=1.11.0 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from tensorflow~=2.12.0->tensorflow_decision_forests) (1.14.1)
    Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from tensorflow~=2.12.0->tensorflow_decision_forests) (0.32.0)
    Requirement already satisfied: python-dateutil>=2.8.2 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from pandas->tensorflow_decision_forests) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from pandas->tensorflow_decision_forests) (2023.3)
    Requirement already satisfied: tzdata>=2022.1 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from pandas->tensorflow_decision_forests) (2023.3)
    Collecting ml-dtypes>=0.1.0 (from jax>=0.3.15->tensorflow~=2.12.0->tensorflow_decision_forests)
      Using cached ml_dtypes-0.1.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (191 kB)
    Requirement already satisfied: scipy>=1.7 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from jax>=0.3.15->tensorflow~=2.12.0->tensorflow_decision_forests) (1.9.1)
    Requirement already satisfied: google-auth<3,>=1.6.3 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from tensorboard<2.13,>=2.12->tensorflow~=2.12.0->tensorflow_decision_forests) (2.18.1)
    Requirement already satisfied: google-auth-oauthlib<1.1,>=0.5 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from tensorboard<2.13,>=2.12->tensorflow~=2.12.0->tensorflow_decision_forests) (1.0.0)
    Requirement already satisfied: markdown>=2.6.8 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from tensorboard<2.13,>=2.12->tensorflow~=2.12.0->tensorflow_decision_forests) (3.4.3)
    Requirement already satisfied: requests<3,>=2.21.0 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from tensorboard<2.13,>=2.12->tensorflow~=2.12.0->tensorflow_decision_forests) (2.31.0)
    Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from tensorboard<2.13,>=2.12->tensorflow~=2.12.0->tensorflow_decision_forests) (0.7.0)
    Requirement already satisfied: werkzeug>=1.0.1 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from tensorboard<2.13,>=2.12->tensorflow~=2.12.0->tensorflow_decision_forests) (2.3.4)
    Requirement already satisfied: cachetools<6.0,>=2.0.0 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.13,>=2.12->tensorflow~=2.12.0->tensorflow_decision_forests) (5.3.0)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.13,>=2.12->tensorflow~=2.12.0->tensorflow_decision_forests) (0.3.0)
    Requirement already satisfied: urllib3<2.0 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.13,>=2.12->tensorflow~=2.12.0->tensorflow_decision_forests) (1.26.15)
    Requirement already satisfied: rsa<5,>=3.1.4 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.13,>=2.12->tensorflow~=2.12.0->tensorflow_decision_forests) (4.9)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from google-auth-oauthlib<1.1,>=0.5->tensorboard<2.13,>=2.12->tensorflow~=2.12.0->tensorflow_decision_forests) (1.3.1)
    Requirement already satisfied: importlib-metadata>=4.4 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from markdown>=2.6.8->tensorboard<2.13,>=2.12->tensorflow~=2.12.0->tensorflow_decision_forests) (6.6.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from requests<3,>=2.21.0->tensorboard<2.13,>=2.12->tensorflow~=2.12.0->tensorflow_decision_forests) (3.1.0)
    Requirement already satisfied: idna<4,>=2.5 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from requests<3,>=2.21.0->tensorboard<2.13,>=2.12->tensorflow~=2.12.0->tensorflow_decision_forests) (3.4)
    Requirement already satisfied: certifi>=2017.4.17 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from requests<3,>=2.21.0->tensorboard<2.13,>=2.12->tensorflow~=2.12.0->tensorflow_decision_forests) (2023.5.7)
    Requirement already satisfied: MarkupSafe>=2.1.1 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from werkzeug>=1.0.1->tensorboard<2.13,>=2.12->tensorflow~=2.12.0->tensorflow_decision_forests) (2.1.2)
    Requirement already satisfied: zipp>=0.5 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<2.13,>=2.12->tensorflow~=2.12.0->tensorflow_decision_forests) (3.15.0)
    Requirement already satisfied: pyasn1<0.6.0,>=0.4.6 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.13,>=2.12->tensorflow~=2.12.0->tensorflow_decision_forests) (0.5.0)
    Requirement already satisfied: oauthlib>=3.0.0 in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<1.1,>=0.5->tensorboard<2.13,>=2.12->tensorflow~=2.12.0->tensorflow_decision_forests) (3.2.2)
    Building wheels for collected packages: jax
      Building wheel for jax (pyproject.toml) ... [?25ldone
    [?25h  Created wheel for jax: filename=jax-0.4.10-py3-none-any.whl size=1480503 sha256=6add56eac934e8c1b9b84463d34202a97ddadb0f833261318eadd04fd46f0dc6
      Stored in directory: /home/kbuilder/.cache/pip/wheels/e5/6c/70/7c6be85fa56f05480fe043bdf0d4f6ec316b122be21e098066
    Successfully built jax
    Installing collected packages: wurlitzer, tensorflow-estimator, numpy, keras, ml-dtypes, jax, tensorboard, tensorflow, tensorflow_decision_forests
      Attempting uninstall: tensorflow-estimator
        Found existing installation: tensorflow-estimator 2.13.0rc0
        Uninstalling tensorflow-estimator-2.13.0rc0:
          Successfully uninstalled tensorflow-estimator-2.13.0rc0
      Attempting uninstall: numpy
        Found existing installation: numpy 1.24.3
        Uninstalling numpy-1.24.3:
          Successfully uninstalled numpy-1.24.3
      Attempting uninstall: keras
        Found existing installation: keras 2.13.1rc0
        Uninstalling keras-2.13.1rc0:
          Successfully uninstalled keras-2.13.1rc0
      Attempting uninstall: tensorboard
        Found existing installation: tensorboard 2.13.0
        Uninstalling tensorboard-2.13.0:
          Successfully uninstalled tensorboard-2.13.0
      Attempting uninstall: tensorflow
        Found existing installation: tensorflow 2.13.0rc0
        Uninstalling tensorflow-2.13.0rc0:
          Successfully uninstalled tensorflow-2.13.0rc0
    Successfully installed jax-0.4.10 keras-2.12.0 ml-dtypes-0.1.0 numpy-1.23.5 tensorboard-2.12.3 tensorflow-2.12.0 tensorflow-estimator-2.12.0 tensorflow_decision_forests-1.3.0 wurlitzer-3.0.3
    

[Wurlitzer](https://pypi.org/project/wurlitzer/) is needed to display the detailed training logs in Colabs (when using `verbose=2` in the model constructor).


```python
!pip install wurlitzer
```

    Requirement already satisfied: wurlitzer in /tmpfs/src/tf_docs_env/lib/python3.9/site-packages (3.0.3)


## Importing libraries


```python
import tensorflow_decision_forests as tfdf

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import math
```

The hidden code cell limits the output height in colab.



```python
#@title

from IPython.core.magic import register_line_magic
from IPython.display import Javascript
from IPython.display import display as ipy_display

# Some of the model training logs can cover the full
# screen if not compressed to a smaller viewport.
# This magic allows setting a max height for a cell.
@register_line_magic
def set_cell_height(size):
  ipy_display(
      Javascript("google.colab.output.setIframeHeight(0, true, {maxHeight: " +
                 str(size) + "})"))
```


```python
# Check the version of TensorFlow Decision Forests
print("Found TensorFlow Decision Forests v" + tfdf.__version__)
```

    Found TensorFlow Decision Forests v1.3.0
    

## Training a Random Forest model

In this section, we train, evaluate, analyse and export a multi-class classification Random Forest trained on the [Palmer's Penguins](https://allisonhorst.github.io/palmerpenguins/articles/intro.html) dataset.

<center>
<img src="https://allisonhorst.github.io/palmerpenguins/reference/figures/lter_penguins.png" width="150"/></center>

**Note:** The dataset was exported to a csv file without pre-processing: `library(palmerpenguins); write.csv(penguins, file="penguins.csv", quote=F, row.names=F)`. 

### Load the dataset and convert it in a tf.Dataset

This dataset is very small (300 examples) and stored as a .csv-like file. Therefore, use Pandas to load it.

**Note:** Pandas is practical as you don't have to type in name of the input features to load them. For larger datasets (>1M examples), using the
[TensorFlow Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) to read the files may be better suited.

Let's assemble the dataset into a csv file (i.e. add the header), and load it:


```python
# Download the dataset
!wget -q https://storage.googleapis.com/download.tensorflow.org/data/palmer_penguins/penguins.csv -O /tmp/penguins.csv

# Load a dataset into a Pandas Dataframe.
dataset_df = pd.read_csv("/tmp/penguins.csv")

# Display the first 3 examples.
dataset_df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>species</th>
      <th>island</th>
      <th>bill_length_mm</th>
      <th>bill_depth_mm</th>
      <th>flipper_length_mm</th>
      <th>body_mass_g</th>
      <th>sex</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.1</td>
      <td>18.7</td>
      <td>181.0</td>
      <td>3750.0</td>
      <td>male</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.5</td>
      <td>17.4</td>
      <td>186.0</td>
      <td>3800.0</td>
      <td>female</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>40.3</td>
      <td>18.0</td>
      <td>195.0</td>
      <td>3250.0</td>
      <td>female</td>
      <td>2007</td>
    </tr>
  </tbody>
</table>
</div>



The dataset contains a mix of numerical (e.g. `bill_depth_mm`), categorical
(e.g. `island`) and missing features. TF-DF supports all these feature types natively (differently than NN based models), therefore there is no need for preprocessing in the form of one-hot encoding, normalization or extra `is_present` feature.

Labels are a bit different: Keras metrics expect integers. The label (`species`) is stored as a string, so let's convert it into an integer.


```python
# Encode the categorical labels as integers.
#
# Details:
# This stage is necessary if your classification label is represented as a
# string since Keras expects integer classification labels.
# When using `pd_dataframe_to_tf_dataset` (see below), this step can be skipped.

# Name of the label column.
label = "species"

classes = dataset_df[label].unique().tolist()
print(f"Label classes: {classes}")

dataset_df[label] = dataset_df[label].map(classes.index)
```

    Label classes: ['Adelie', 'Gentoo', 'Chinstrap']
    

Next split the dataset into training and testing:


```python
# Split the dataset into a training and a testing dataset.

def split_dataset(dataset, test_ratio=0.30):
  """Splits a panda dataframe in two."""
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]


train_ds_pd, test_ds_pd = split_dataset(dataset_df)
print("{} examples in training, {} examples for testing.".format(
    len(train_ds_pd), len(test_ds_pd)))
```

    241 examples in training, 103 examples for testing.
    

And finally, convert the pandas dataframe (`pd.Dataframe`) into tensorflow datasets (`tf.data.Dataset`):


```python
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label=label)
```

**Notes:** Recall that `pd_dataframe_to_tf_dataset` converts string labels to integers if necessary.

If you want to create the `tf.data.Dataset` yourself, there are a couple of things to remember:

- The learning algorithms work with a one-epoch dataset and without shuffling.
- The batch size does not impact the training algorithm, but a small value might slow down reading the dataset.


### Train the model


```python
%set_cell_height 300

# Specify the model.
model_1 = tfdf.keras.RandomForestModel(verbose=2)

# Train the model.
model_1.fit(train_ds)
```


    <IPython.core.display.Javascript object>


    Warning: The `num_threads` constructor argument is not set and the number of CPU is os.cpu_count()=32 > 32. Setting num_threads to 32. Set num_threads manually to use more than 32 cpus.
    

    WARNING:absl:The `num_threads` constructor argument is not set and the number of CPU is os.cpu_count()=32 > 32. Setting num_threads to 32. Set num_threads manually to use more than 32 cpus.
    

    Use /tmpfs/tmp/tmpp2p594hd as temporary training directory
    Reading training dataset...
    Training tensor examples:
    Features: {'island': <tf.Tensor 'data:0' shape=(None,) dtype=string>, 'bill_length_mm': <tf.Tensor 'data_1:0' shape=(None,) dtype=float64>, 'bill_depth_mm': <tf.Tensor 'data_2:0' shape=(None,) dtype=float64>, 'flipper_length_mm': <tf.Tensor 'data_3:0' shape=(None,) dtype=float64>, 'body_mass_g': <tf.Tensor 'data_4:0' shape=(None,) dtype=float64>, 'sex': <tf.Tensor 'data_5:0' shape=(None,) dtype=string>, 'year': <tf.Tensor 'data_6:0' shape=(None,) dtype=int64>}
    Label: Tensor("data_7:0", shape=(None,), dtype=int64)
    Weights: None
    Normalized tensor features:
     {'island': SemanticTensor(semantic=<Semantic.CATEGORICAL: 2>, tensor=<tf.Tensor 'data:0' shape=(None,) dtype=string>), 'bill_length_mm': SemanticTensor(semantic=<Semantic.NUMERICAL: 1>, tensor=<tf.Tensor 'Cast:0' shape=(None,) dtype=float32>), 'bill_depth_mm': SemanticTensor(semantic=<Semantic.NUMERICAL: 1>, tensor=<tf.Tensor 'Cast_1:0' shape=(None,) dtype=float32>), 'flipper_length_mm': SemanticTensor(semantic=<Semantic.NUMERICAL: 1>, tensor=<tf.Tensor 'Cast_2:0' shape=(None,) dtype=float32>), 'body_mass_g': SemanticTensor(semantic=<Semantic.NUMERICAL: 1>, tensor=<tf.Tensor 'Cast_3:0' shape=(None,) dtype=float32>), 'sex': SemanticTensor(semantic=<Semantic.CATEGORICAL: 2>, tensor=<tf.Tensor 'data_5:0' shape=(None,) dtype=string>), 'year': SemanticTensor(semantic=<Semantic.NUMERICAL: 1>, tensor=<tf.Tensor 'Cast_4:0' shape=(None,) dtype=float32>)}
    Training dataset read in 0:00:03.440338. Found 241 examples.
    Training model...
    Standard output detected as not visible to the user e.g. running in a notebook. Creating a training log redirection. If training gets stuck, try calling tfdf.keras.set_training_logs_redirection(False).
    

    [INFO 23-05-23 11:13:09.6024 UTC kernel.cc:773] Start Yggdrasil model training
    [INFO 23-05-23 11:13:09.6024 UTC kernel.cc:774] Collect training examples
    [INFO 23-05-23 11:13:09.6025 UTC kernel.cc:787] Dataspec guide:
    column_guides {
      column_name_pattern: "^__LABEL$"
      type: CATEGORICAL
      categorial {
        min_vocab_frequency: 0
        max_vocab_count: -1
      }
    }
    default_column_guide {
      categorial {
        max_vocab_count: 2000
      }
      discretized_numerical {
        maximum_num_bins: 255
      }
    }
    ignore_columns_without_guides: false
    detect_numerical_as_discretized_numerical: false
    
    [INFO 23-05-23 11:13:09.6029 UTC kernel.cc:393] Number of batches: 1
    [INFO 23-05-23 11:13:09.6029 UTC kernel.cc:394] Number of examples: 241
    [INFO 23-05-23 11:13:09.6030 UTC kernel.cc:794] Training dataset:
    Number of records: 241
    Number of columns: 8
    
    Number of columns by type:
    	NUMERICAL: 5 (62.5%)
    	CATEGORICAL: 3 (37.5%)
    
    Columns:
    
    NUMERICAL: 5 (62.5%)
    	1: "bill_depth_mm" NUMERICAL num-nas:2 (0.829876%) mean:17.0849 min:13.1 max:21.5 sd:1.9903
    	2: "bill_length_mm" NUMERICAL num-nas:2 (0.829876%) mean:43.5561 min:32.1 max:59.6 sd:5.41022
    	3: "body_mass_g" NUMERICAL num-nas:2 (0.829876%) mean:4195.4 min:2850 max:6050 sd:768.594
    	4: "flipper_length_mm" NUMERICAL num-nas:2 (0.829876%) mean:200.439 min:176 max:230 sd:13.9195
    	7: "year" NUMERICAL mean:2008.06 min:2007 max:2009 sd:0.802451
    
    CATEGORICAL: 3 (37.5%)
    	0: "__LABEL" CATEGORICAL integerized vocab-size:4 no-ood-item
    	5: "island" CATEGORICAL has-dict vocab-size:4 zero-ood-items most-frequent:"Biscoe" 122 (50.6224%)
    	6: "sex" CATEGORICAL num-nas:7 (2.90456%) has-dict vocab-size:3 zero-ood-items most-frequent:"female" 123 (52.5641%)
    
    Terminology:
    	nas: Number of non-available (i.e. missing) values.
    	ood: Out of dictionary.
    	manually-defined: Attribute which type is manually defined by the user i.e. the type was not automatically inferred.
    	tokenized: The attribute value is obtained through tokenization.
    	has-dict: The attribute is attached to a string dictionary e.g. a categorical attribute stored as a string.
    	vocab-size: Number of unique values.
    
    [INFO 23-05-23 11:13:09.6030 UTC kernel.cc:810] Configure learner
    [INFO 23-05-23 11:13:09.6032 UTC kernel.cc:824] Training config:
    learner: "RANDOM_FOREST"
    features: "^bill_depth_mm$"
    features: "^bill_length_mm$"
    features: "^body_mass_g$"
    features: "^flipper_length_mm$"
    features: "^island$"
    features: "^sex$"
    features: "^year$"
    label: "^__LABEL$"
    task: CLASSIFICATION
    random_seed: 123456
    metadata {
      framework: "TF Keras"
    }
    pure_serving_model: false
    [yggdrasil_decision_forests.model.random_forest.proto.random_forest_config] {
      num_trees: 300
      decision_tree {
        max_depth: 16
        min_examples: 5
        in_split_min_examples_check: true
        keep_non_leaf_label_distribution: true
        num_candidate_attributes: 0
        missing_value_policy: GLOBAL_IMPUTATION
        allow_na_conditions: false
        categorical_set_greedy_forward {
          sampling: 0.1
          max_num_items: -1
          min_item_frequency: 1
        }
        growing_strategy_local {
        }
        categorical {
          cart {
          }
        }
        axis_aligned_split {
        }
        internal {
          sorting_strategy: PRESORTED
        }
        uplift {
          min_examples_in_treatment: 5
          split_score: KULLBACK_LEIBLER
        }
      }
      winner_take_all_inference: true
      compute_oob_performances: true
      compute_oob_variable_importances: false
      num_oob_variable_importances_permutations: 1
      bootstrap_training_dataset: true
      bootstrap_size_ratio: 1
      adapt_bootstrap_size_ratio_for_maximum_training_duration: false
      sampling_with_replacement: true
    }
    
    [INFO 23-05-23 11:13:09.6035 UTC kernel.cc:827] Deployment config:
    cache_path: "/tmpfs/tmp/tmpp2p594hd/working_cache"
    num_threads: 32
    try_resume_training: true
    
    [INFO 23-05-23 11:13:09.6037 UTC kernel.cc:889] Train model
    [INFO 23-05-23 11:13:09.6038 UTC random_forest.cc:416] Training random forest on 241 example(s) and 7 feature(s).
    [INFO 23-05-23 11:13:09.6093 UTC random_forest.cc:805] Training of tree  1/300 (tree index:0) done accuracy:0.965116 logloss:1.25734
    [INFO 23-05-23 11:13:09.6102 UTC random_forest.cc:805] Training of tree  11/300 (tree index:37) done accuracy:0.966805 logloss:0.229339
    [INFO 23-05-23 11:13:09.6108 UTC random_forest.cc:805] Training of tree  21/300 (tree index:41) done accuracy:0.975104 logloss:0.0891776
    [INFO 23-05-23 11:13:09.6109 UTC random_forest.cc:805] Training of tree  35/300 (tree index:17) done accuracy:0.975104 logloss:0.0891645
    [INFO 23-05-23 11:13:09.6110 UTC random_forest.cc:805] Training of tree  50/300 (tree index:50) done accuracy:0.975104 logloss:0.0943436
    [INFO 23-05-23 11:13:09.6114 UTC random_forest.cc:805] Training of tree  60/300 (tree index:62) done accuracy:0.983402 logloss:0.0839909
    [INFO 23-05-23 11:13:09.6117 UTC random_forest.cc:805] Training of tree  70/300 (tree index:69) done accuracy:0.983402 logloss:0.088746
    [INFO 23-05-23 11:13:09.6119 UTC random_forest.cc:805] Training of tree  80/300 (tree index:80) done accuracy:0.966805 logloss:0.0902204
    [INFO 23-05-23 11:13:09.6122 UTC random_forest.cc:805] Training of tree  91/300 (tree index:90) done accuracy:0.966805 logloss:0.0906052
    [INFO 23-05-23 11:13:09.6127 UTC random_forest.cc:805] Training of tree  103/300 (tree index:103) done accuracy:0.966805 logloss:0.0935805
    [INFO 23-05-23 11:13:09.6131 UTC random_forest.cc:805] Training of tree  114/300 (tree index:115) done accuracy:0.966805 logloss:0.0916048
    [INFO 23-05-23 11:13:09.6135 UTC random_forest.cc:805] Training of tree  126/300 (tree index:124) done accuracy:0.966805 logloss:0.0946431
    [INFO 23-05-23 11:13:09.6137 UTC random_forest.cc:805] Training of tree  136/300 (tree index:134) done accuracy:0.966805 logloss:0.0929181
    [INFO 23-05-23 11:13:09.6141 UTC random_forest.cc:805] Training of tree  146/300 (tree index:147) done accuracy:0.970954 logloss:0.093388
    [INFO 23-05-23 11:13:09.6144 UTC random_forest.cc:805] Training of tree  157/300 (tree index:156) done accuracy:0.970954 logloss:0.095734
    [INFO 23-05-23 11:13:09.6147 UTC random_forest.cc:805] Training of tree  167/300 (tree index:165) done accuracy:0.975104 logloss:0.0956275
    [INFO 23-05-23 11:13:09.6150 UTC random_forest.cc:805] Training of tree  177/300 (tree index:178) done accuracy:0.975104 logloss:0.0953425
    [INFO 23-05-23 11:13:09.6153 UTC random_forest.cc:805] Training of tree  188/300 (tree index:189) done accuracy:0.975104 logloss:0.0962487
    [INFO 23-05-23 11:13:09.6155 UTC random_forest.cc:805] Training of tree  199/300 (tree index:198) done accuracy:0.975104 logloss:0.0965289
    [INFO 23-05-23 11:13:09.6158 UTC random_forest.cc:805] Training of tree  209/300 (tree index:208) done accuracy:0.975104 logloss:0.0953626
    [INFO 23-05-23 11:13:09.6161 UTC random_forest.cc:805] Training of tree  220/300 (tree index:219) done accuracy:0.975104 logloss:0.0954487
    [INFO 23-05-23 11:13:09.6164 UTC random_forest.cc:805] Training of tree  230/300 (tree index:229) done accuracy:0.975104 logloss:0.0959775
    [INFO 23-05-23 11:13:09.6168 UTC random_forest.cc:805] Training of tree  240/300 (tree index:241) done accuracy:0.975104 logloss:0.095632
    [INFO 23-05-23 11:13:09.6171 UTC random_forest.cc:805] Training of tree  250/300 (tree index:251) done accuracy:0.975104 logloss:0.0949993
    [INFO 23-05-23 11:13:09.6174 UTC random_forest.cc:805] Training of tree  260/300 (tree index:260) done accuracy:0.979253 logloss:0.0950653
    [INFO 23-05-23 11:13:09.6177 UTC random_forest.cc:805] Training of tree  272/300 (tree index:272) done accuracy:0.979253 logloss:0.0948893
    [INFO 23-05-23 11:13:09.6179 UTC random_forest.cc:805] Training of tree  282/300 (tree index:281) done accuracy:0.979253 logloss:0.0945184
    [INFO 23-05-23 11:13:09.6182 UTC random_forest.cc:805] Training of tree  293/300 (tree index:293) done accuracy:0.975104 logloss:0.0947919
    [INFO 23-05-23 11:13:09.6195 UTC random_forest.cc:805] Training of tree  300/300 (tree index:299) done accuracy:0.979253 logloss:0.0936497
    [INFO 23-05-23 11:13:09.6199 UTC random_forest.cc:885] Final OOB metrics: accuracy:0.979253 logloss:0.0936497
    [INFO 23-05-23 11:13:09.6211 UTC kernel.cc:926] Export model in log directory: /tmpfs/tmp/tmpp2p594hd with prefix 1f061298b1a0444a
    [INFO 23-05-23 11:13:09.6267 UTC kernel.cc:944] Save model in resources
    [INFO 23-05-23 11:13:09.6297 UTC abstract_model.cc:849] Model self evaluation:
    Number of predictions (without weights): 241
    Number of predictions (with weights): 241
    Task: CLASSIFICATION
    Label: __LABEL
    
    Accuracy: 0.979253  CI95[W][0.956875 0.99179]
    LogLoss: : 0.0936497
    ErrorRate: : 0.0207469
    
    Default Accuracy: : 0.46888
    Default LogLoss: : 1.02428
    Default ErrorRate: : 0.53112
    
    Confusion Table:
    truth\prediction
       0    1   2   3
    0  0    0   0   0
    1  0  111   1   1
    2  0    1  86   0
    3  0    1   1  39
    Total: 241
    
    One vs other classes:
    
    [INFO 23-05-23 11:13:09.6403 UTC kernel.cc:1242] Loading model from path /tmpfs/tmp/tmpp2p594hd/model/ with prefix 1f061298b1a0444a
    [INFO 23-05-23 11:13:09.6548 UTC decision_forest.cc:660] Model loaded with 300 root(s), 4560 node(s), and 7 input feature(s).
    [INFO 23-05-23 11:13:09.6548 UTC abstract_model.cc:1311] Engine "RandomForestGeneric" built
    [INFO 23-05-23 11:13:09.6549 UTC kernel.cc:1074] Use fast generic engine
    

    Model trained in 0:00:00.059506
    Compiling model...
    WARNING:tensorflow:AutoGraph could not transform <function simple_ml_inference_op_with_handle at 0x7ff38e9ec700> and will run it as-is.
    Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
    Cause: could not get source code
    To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
    

    WARNING:tensorflow:AutoGraph could not transform <function simple_ml_inference_op_with_handle at 0x7ff38e9ec700> and will run it as-is.
    Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
    Cause: could not get source code
    To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
    

    WARNING: AutoGraph could not transform <function simple_ml_inference_op_with_handle at 0x7ff38e9ec700> and will run it as-is.
    Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
    Cause: could not get source code
    To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
    Model compiled.
    




    <keras.callbacks.History at 0x7ff38e0b8b80>



### Remarks

-   No input features are specified. Therefore, all the columns will be used as
    input features except for the label. The feature used by the model are shown
    in the training logs and in the `model.summary()`.
-   DFs consume natively numerical, categorical, categorical-set features and
    missing-values. Numerical features do not need to be normalized. Categorical
    string values do not need to be encoded in a dictionary.
-   No training hyper-parameters are specified. Therefore the default
    hyper-parameters will be used. Default hyper-parameters provide
    reasonable results in most situations.
-   Calling `compile` on the model before the `fit` is optional. Compile can be
    used to provide extra evaluation metrics.
-   Training algorithms do not need validation datasets. If a validation dataset
    is provided, it will only be used to show metrics.
-   Tweak the `verbose` argument to `RandomForestModel` to control the amount of
    displayed training logs. Set `verbose=0` to hide most of the logs. Set
    `verbose=2` to show all the logs.

**Note:** A *Categorical-Set* feature is composed of a set of categorical values (while a *Categorical* is only one value). More details and examples are given later.

## Evaluate the model

Let's evaluate our model on the test dataset.


```python
model_1.compile(metrics=["accuracy"])
evaluation = model_1.evaluate(test_ds, return_dict=True)
print()

for name, value in evaluation.items():
  print(f"{name}: {value:.4f}")
```

    1/1 [==============================] - 0s 284ms/step - loss: 0.0000e+00 - accuracy: 0.9806
    
    loss: 0.0000
    accuracy: 0.9806
    

**Remark:** The test accuracy is close to the Out-of-bag accuracy
shown in the training logs.

See the **Model Self Evaluation** section below for more evaluation methods.

## Prepare this model for TensorFlow Serving.

Export the model to the SavedModel format for later re-use e.g.
[TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving).



```python
model_1.save("/tmp/my_saved_model")
```

    WARNING:absl:Found untraced functions such as call_get_leaves, _update_step_xla while saving (showing 2 of 2). These functions will not be directly callable after loading.
    

    INFO:tensorflow:Assets written to: /tmp/my_saved_model/assets
    

    INFO:tensorflow:Assets written to: /tmp/my_saved_model/assets
    

## Plot the model

Plotting a decision tree and following the first branches helps learning about decision forests. In some cases, plotting a model can even be used for debugging.

Because of the difference in the way they are trained, some models are more interesting to plan than others. Because of the noise injected during training and the depth of the trees, plotting Random Forest is less informative than plotting a CART or the first tree of a Gradient Boosted Tree.

Never the less, let's plot the first tree of our Random Forest model:


```python
tfdf.model_plotter.plot_model_in_colab(model_1, tree_idx=0, max_depth=3)
```





<script src="https://d3js.org/d3.v6.min.js"></script>
<div id="tree_plot_5b8200139d5d4e1ab3c2a5346ad918cf"></div>
<script>
/*
 * Copyright 2021 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 *  Plotting of decision trees generated by TF-DF.
 *
 *  A tree is a recursive structure of node objects.
 *  A node contains one or more of the following components:
 *
 *    - A value: Representing the output of the node. If the node is not a leaf,
 *      the value is only present for analysis i.e. it is not used for
 *      predictions.
 *
 *    - A condition : For non-leaf nodes, the condition (also known as split)
 *      defines a binary test to branch to the positive or negative child.
 *
 *    - An explanation: Generally a plot showing the relation between the label
 *      and the condition to give insights about the effect of the condition.
 *
 *    - Two children : For non-leaf nodes, the children nodes. The first
 *      children (i.e. "node.children[0]") is the negative children (drawn in
 *      red). The second children is the positive one (drawn in green).
 *
 */

/**
 * Plots a single decision tree into a DOM element.
 * @param {!options} options Dictionary of configurations.
 * @param {!tree} raw_tree Recursive tree structure.
 * @param {string} canvas_id Id of the output dom element.
 */
function display_tree(options, raw_tree, canvas_id) {
  console.log(options);

  // Determine the node placement.
  const tree_struct = d3.tree().nodeSize(
      [options.node_y_offset, options.node_x_offset])(d3.hierarchy(raw_tree));

  // Boundaries of the node placement.
  let x_min = Infinity;
  let x_max = -x_min;
  let y_min = Infinity;
  let y_max = -x_min;

  tree_struct.each(d => {
    if (d.x > x_max) x_max = d.x;
    if (d.x < x_min) x_min = d.x;
    if (d.y > y_max) y_max = d.y;
    if (d.y < y_min) y_min = d.y;
  });

  // Size of the plot.
  const width = y_max - y_min + options.node_x_size + options.margin * 2;
  const height = x_max - x_min + options.node_y_size + options.margin * 2 +
      options.node_y_offset - options.node_y_size;

  const plot = d3.select(canvas_id);

  // Tool tip
  options.tooltip = plot.append('div')
                        .attr('width', 100)
                        .attr('height', 100)
                        .style('padding', '4px')
                        .style('background', '#fff')
                        .style('box-shadow', '4px 4px 0px rgba(0,0,0,0.1)')
                        .style('border', '1px solid black')
                        .style('font-family', 'sans-serif')
                        .style('font-size', options.font_size)
                        .style('position', 'absolute')
                        .style('z-index', '10')
                        .attr('pointer-events', 'none')
                        .style('display', 'none');

  // Create canvas
  const svg = plot.append('svg').attr('width', width).attr('height', height);
  const graph =
      svg.style('overflow', 'visible')
          .append('g')
          .attr('font-family', 'sans-serif')
          .attr('font-size', options.font_size)
          .attr(
              'transform',
              () => `translate(${options.margin},${
                  - x_min + options.node_y_offset / 2 + options.margin})`);

  // Plot bounding box.
  if (options.show_plot_bounding_box) {
    svg.append('rect')
        .attr('width', width)
        .attr('height', height)
        .attr('fill', 'none')
        .attr('stroke-width', 1.0)
        .attr('stroke', 'black');
  }

  // Draw the edges.
  display_edges(options, graph, tree_struct);

  // Draw the nodes.
  display_nodes(options, graph, tree_struct);
}

/**
 * Draw the nodes of the tree.
 * @param {!options} options Dictionary of configurations.
 * @param {!graph} graph D3 search handle containing the graph.
 * @param {!tree_struct} tree_struct Structure of the tree (node placement,
 *     data, etc.).
 */
function display_nodes(options, graph, tree_struct) {
  const nodes = graph.append('g')
                    .selectAll('g')
                    .data(tree_struct.descendants())
                    .join('g')
                    .attr('transform', d => `translate(${d.y},${d.x})`);

  nodes.append('rect')
      .attr('x', 0.5)
      .attr('y', 0.5)
      .attr('width', options.node_x_size)
      .attr('height', options.node_y_size)
      .attr('stroke', 'lightgrey')
      .attr('stroke-width', 1)
      .attr('fill', 'white')
      .attr('y', -options.node_y_size / 2);

  // Brackets on the right of condition nodes without children.
  non_leaf_node_without_children =
      nodes.filter(node => node.data.condition != null && node.children == null)
          .append('g')
          .attr('transform', `translate(${options.node_x_size},0)`);

  non_leaf_node_without_children.append('path')
      .attr('d', 'M0,0 C 10,0 0,10 10,10')
      .attr('fill', 'none')
      .attr('stroke-width', 1.0)
      .attr('stroke', '#F00');

  non_leaf_node_without_children.append('path')
      .attr('d', 'M0,0 C 10,0 0,-10 10,-10')
      .attr('fill', 'none')
      .attr('stroke-width', 1.0)
      .attr('stroke', '#0F0');

  const node_content = nodes.append('g').attr(
      'transform',
      `translate(0,${options.node_padding - options.node_y_size / 2})`);

  node_content.append(node => create_node_element(options, node));
}

/**
 * Creates the D3 content for a single node.
 * @param {!options} options Dictionary of configurations.
 * @param {!node} node Node to draw.
 * @return {!d3} D3 content.
 */
function create_node_element(options, node) {
  // Output accumulator.
  let output = {
    // Content to draw.
    content: d3.create('svg:g'),
    // Vertical offset to the next element to draw.
    vertical_offset: 0
  };

  // Conditions.
  if (node.data.condition != null) {
    display_condition(options, node.data.condition, output);
  }

  // Values.
  if (node.data.value != null) {
    display_value(options, node.data.value, output);
  }

  // Explanations.
  if (node.data.explanation != null) {
    display_explanation(options, node.data.explanation, output);
  }

  return output.content.node();
}


/**
 * Adds a single line of text inside of a node.
 * @param {!options} options Dictionary of configurations.
 * @param {string} text Text to display.
 * @param {!output} output Output display accumulator.
 */
function display_node_text(options, text, output) {
  output.content.append('text')
      .attr('x', options.node_padding)
      .attr('y', output.vertical_offset)
      .attr('alignment-baseline', 'hanging')
      .text(text);
  output.vertical_offset += 10;
}

/**
 * Adds a single line of text inside of a node with a tooltip.
 * @param {!options} options Dictionary of configurations.
 * @param {string} text Text to display.
 * @param {string} tooltip Text in the Tooltip.
 * @param {!output} output Output display accumulator.
 */
function display_node_text_with_tooltip(options, text, tooltip, output) {
  const item = output.content.append('text')
                   .attr('x', options.node_padding)
                   .attr('alignment-baseline', 'hanging')
                   .text(text);

  add_tooltip(options, item, () => tooltip);
  output.vertical_offset += 10;
}

/**
 * Adds a tooltip to a dom element.
 * @param {!options} options Dictionary of configurations.
 * @param {!dom} target Dom element to equip with a tooltip.
 * @param {!func} get_content Generates the html content of the tooltip.
 */
function add_tooltip(options, target, get_content) {
  function show(d) {
    options.tooltip.style('display', 'block');
    options.tooltip.html(get_content());
  }

  function hide(d) {
    options.tooltip.style('display', 'none');
  }

  function move(d) {
    options.tooltip.style('display', 'block');
    options.tooltip.style('left', (d.pageX + 5) + 'px');
    options.tooltip.style('top', d.pageY + 'px');
  }

  target.on('mouseover', show);
  target.on('mouseout', hide);
  target.on('mousemove', move);
}

/**
 * Adds a condition inside of a node.
 * @param {!options} options Dictionary of configurations.
 * @param {!condition} condition Condition to display.
 * @param {!output} output Output display accumulator.
 */
function display_condition(options, condition, output) {
  threshold_format = d3.format('r');

  if (condition.type === 'IS_MISSING') {
    display_node_text(options, `${condition.attribute} is missing`, output);
    return;
  }

  if (condition.type === 'IS_TRUE') {
    display_node_text(options, `${condition.attribute} is true`, output);
    return;
  }

  if (condition.type === 'NUMERICAL_IS_HIGHER_THAN') {
    format = d3.format('r');
    display_node_text(
        options,
        `${condition.attribute} >= ${threshold_format(condition.threshold)}`,
        output);
    return;
  }

  if (condition.type === 'CATEGORICAL_IS_IN') {
    display_node_text_with_tooltip(
        options, `${condition.attribute} in [...]`,
        `${condition.attribute} in [${condition.mask}]`, output);
    return;
  }

  if (condition.type === 'CATEGORICAL_SET_CONTAINS') {
    display_node_text_with_tooltip(
        options, `${condition.attribute} intersect [...]`,
        `${condition.attribute} intersect [${condition.mask}]`, output);
    return;
  }

  if (condition.type === 'NUMERICAL_SPARSE_OBLIQUE') {
    display_node_text_with_tooltip(
        options, `Sparse oblique split...`,
        `[${condition.attributes}]*[${condition.weights}]>=${
            threshold_format(condition.threshold)}`,
        output);
    return;
  }

  display_node_text(
      options, `Non supported condition ${condition.type}`, output);
}

/**
 * Adds a value inside of a node.
 * @param {!options} options Dictionary of configurations.
 * @param {!value} value Value to display.
 * @param {!output} output Output display accumulator.
 */
function display_value(options, value, output) {
  if (value.type === 'PROBABILITY') {
    const left_margin = 0;
    const right_margin = 50;
    const plot_width = options.node_x_size - options.node_padding * 2 -
        left_margin - right_margin;

    let cusum = Array.from(d3.cumsum(value.distribution));
    cusum.unshift(0);
    const distribution_plot = output.content.append('g').attr(
        'transform', `translate(0,${output.vertical_offset + 0.5})`);

    distribution_plot.selectAll('rect')
        .data(value.distribution)
        .join('rect')
        .attr('height', 10)
        .attr(
            'x',
            (d, i) =>
                (cusum[i] * plot_width + left_margin + options.node_padding))
        .attr('width', (d, i) => d * plot_width)
        .style('fill', (d, i) => d3.schemeSet1[i]);

    const num_examples =
        output.content.append('g')
            .attr('transform', `translate(0,${output.vertical_offset})`)
            .append('text')
            .attr('x', options.node_x_size - options.node_padding)
            .attr('alignment-baseline', 'hanging')
            .attr('text-anchor', 'end')
            .text(`(${value.num_examples})`);

    const distribution_details = d3.create('ul');
    distribution_details.selectAll('li')
        .data(value.distribution)
        .join('li')
        .append('span')
        .text(
            (d, i) =>
                'class ' + i + ': ' + d3.format('.3%')(value.distribution[i]));

    add_tooltip(options, distribution_plot, () => distribution_details.html());
    add_tooltip(options, num_examples, () => 'Number of examples');

    output.vertical_offset += 10;
    return;
  }

  if (value.type === 'REGRESSION') {
    display_node_text(
        options,
        'value: ' + d3.format('r')(value.value) + ` (` +
            d3.format('.6')(value.num_examples) + `)`,
        output);
    return;
  }

  display_node_text(options, `Non supported value ${value.type}`, output);
}

/**
 * Adds an explanation inside of a node.
 * @param {!options} options Dictionary of configurations.
 * @param {!explanation} explanation Explanation to display.
 * @param {!output} output Output display accumulator.
 */
function display_explanation(options, explanation, output) {
  // Margin before the explanation.
  output.vertical_offset += 10;

  display_node_text(
      options, `Non supported explanation ${explanation.type}`, output);
}


/**
 * Draw the edges of the tree.
 * @param {!options} options Dictionary of configurations.
 * @param {!graph} graph D3 search handle containing the graph.
 * @param {!tree_struct} tree_struct Structure of the tree (node placement,
 *     data, etc.).
 */
function display_edges(options, graph, tree_struct) {
  // Draw an edge between a parent and a child node with a bezier.
  function draw_single_edge(d) {
    return 'M' + (d.source.y + options.node_x_size) + ',' + d.source.x + ' C' +
        (d.source.y + options.node_x_size + options.edge_rounding) + ',' +
        d.source.x + ' ' + (d.target.y - options.edge_rounding) + ',' +
        d.target.x + ' ' + d.target.y + ',' + d.target.x;
  }

  graph.append('g')
      .attr('fill', 'none')
      .attr('stroke-width', 1.2)
      .selectAll('path')
      .data(tree_struct.links())
      .join('path')
      .attr('d', draw_single_edge)
      .attr(
          'stroke', d => (d.target === d.source.children[0]) ? '#0F0' : '#F00');
}

display_tree({"margin": 10, "node_x_size": 160, "node_y_size": 28, "node_x_offset": 180, "node_y_offset": 33, "font_size": 10, "edge_rounding": 20, "node_padding": 2, "show_plot_bounding_box": false}, {"value": {"type": "PROBABILITY", "distribution": [0.49377593360995853, 0.3153526970954357, 0.1908713692946058], "num_examples": 241.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "flipper_length_mm", "threshold": 207.5}, "children": [{"value": {"type": "PROBABILITY", "distribution": [0.012658227848101266, 0.9367088607594937, 0.05063291139240506], "num_examples": 79.0}, "condition": {"type": "CATEGORICAL_IS_IN", "attribute": "island", "mask": ["Dream"]}, "children": [{"value": {"type": "PROBABILITY", "distribution": [0.2, 0.0, 0.8], "num_examples": 5.0}}, {"value": {"type": "PROBABILITY", "distribution": [0.0, 1.0, 0.0], "num_examples": 74.0}}]}, {"value": {"type": "PROBABILITY", "distribution": [0.7283950617283951, 0.012345679012345678, 0.25925925925925924], "num_examples": 162.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "bill_length_mm", "threshold": 43.349998474121094}, "children": [{"value": {"type": "PROBABILITY", "distribution": [0.1111111111111111, 0.044444444444444446, 0.8444444444444444], "num_examples": 45.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "body_mass_g", "threshold": 4125.0}, "children": [{"value": {"type": "PROBABILITY", "distribution": [0.45454545454545453, 0.18181818181818182, 0.36363636363636365], "num_examples": 11.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "bill_length_mm", "threshold": 47.20000076293945}}, {"value": {"type": "PROBABILITY", "distribution": [0.0, 0.0, 1.0], "num_examples": 34.0}}]}, {"value": {"type": "PROBABILITY", "distribution": [0.9658119658119658, 0.0, 0.03418803418803419], "num_examples": 117.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "bill_length_mm", "threshold": 42.349998474121094}, "children": [{"value": {"type": "PROBABILITY", "distribution": [0.7, 0.0, 0.3], "num_examples": 10.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "bill_length_mm", "threshold": 42.75}}, {"value": {"type": "PROBABILITY", "distribution": [0.9906542056074766, 0.0, 0.009345794392523364], "num_examples": 107.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "bill_length_mm", "threshold": 40.849998474121094}}]}]}]}, "#tree_plot_5b8200139d5d4e1ab3c2a5346ad918cf")
</script>




The root node on the left contains the first condition (`bill_depth_mm >= 16.55`), number of examples (240) and label distribution (the red-blue-green bar).

Examples that evaluates true to `bill_depth_mm >= 16.55` are branched to the green path. The other ones are branched to the red path.

The deeper the node, the more `pure` they become i.e. the label distribution is biased toward a subset of classes. 

**Note:** Over the mouse on top of the plot for details.

## Model structure and feature importance

The overall structure of the model is show with `.summary()`. You will see:

-   **Type**: The learning algorithm used to train the model (`Random Forest` in
    our case).
-   **Task**: The problem solved by the model (`Classification` in our case).
-   **Input Features**: The input features of the model.
-   **Variable Importance**: Different measures of the importance of each
    feature for the model.
-   **Out-of-bag evaluation**: The out-of-bag evaluation of the model. This is a
    cheap and efficient alternative to cross-validation.
-   **Number of {trees, nodes} and other metrics**: Statistics about the
    structure of the decisions forests.

**Remark:** The summary's content depends on the learning algorithm (e.g.
Out-of-bag is only available for Random Forest) and the hyper-parameters (e.g.
the *mean-decrease-in-accuracy* variable importance can be disabled in the
hyper-parameters).


```python
%set_cell_height 300
model_1.summary()
```


    <IPython.core.display.Javascript object>


    Model: "random_forest_model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
    =================================================================
    Total params: 1
    Trainable params: 0
    Non-trainable params: 1
    _________________________________________________________________
    Type: "RANDOM_FOREST"
    Task: CLASSIFICATION
    Label: "__LABEL"
    
    Input Features (7):
    	bill_depth_mm
    	bill_length_mm
    	body_mass_g
    	flipper_length_mm
    	island
    	sex
    	year
    
    No weights
    
    Variable Importance: INV_MEAN_MIN_DEPTH:
        1. "flipper_length_mm"  0.446777 ################
        2.    "bill_length_mm"  0.414541 #############
        3.     "bill_depth_mm"  0.314568 ######
        4.            "island"  0.305414 #####
        5.       "body_mass_g"  0.270869 ##
        6.               "sex"  0.234735 
        7.              "year"  0.233312 
    
    Variable Importance: NUM_AS_ROOT:
        1. "flipper_length_mm" 159.000000 ################
        2.    "bill_length_mm" 69.000000 ######
        3.     "bill_depth_mm" 60.000000 #####
        4.       "body_mass_g"  9.000000 
        5.            "island"  3.000000 
    
    Variable Importance: NUM_NODES:
        1.    "bill_length_mm" 666.000000 ################
        2. "flipper_length_mm" 403.000000 #########
        3.     "bill_depth_mm" 400.000000 #########
        4.       "body_mass_g" 330.000000 #######
        5.            "island" 284.000000 ######
        6.               "sex" 30.000000 
        7.              "year" 17.000000 
    
    Variable Importance: SUM_SCORE:
        1. "flipper_length_mm" 24616.837831 ################
        2.    "bill_length_mm" 22270.054683 ##############
        3.     "bill_depth_mm" 10999.832416 #######
        4.            "island" 8892.439133 #####
        5.       "body_mass_g" 3670.308069 ##
        6.               "sex" 209.510333 
        7.              "year" 27.375907 
    
    
    
    Winner takes all: true
    Out-of-bag evaluation: accuracy:0.979253 logloss:0.0936497
    Number of trees: 300
    Total number of nodes: 4560
    
    Number of nodes by tree:
    Count: 300 Average: 15.2 StdDev: 3.10054
    Min: 7 Max: 25 Ignored: 0
    ----------------------------------------------
    [  7,  8)  1   0.33%   0.33%
    [  8,  9)  0   0.00%   0.33%
    [  9, 10)  5   1.67%   2.00% #
    [ 10, 11)  0   0.00%   2.00%
    [ 11, 12) 31  10.33%  12.33% ####
    [ 12, 13)  0   0.00%  12.33%
    [ 13, 14) 76  25.33%  37.67% #########
    [ 14, 15)  0   0.00%  37.67%
    [ 15, 16) 87  29.00%  66.67% ##########
    [ 16, 17)  0   0.00%  66.67%
    [ 17, 18) 50  16.67%  83.33% ######
    [ 18, 19)  0   0.00%  83.33%
    [ 19, 20) 25   8.33%  91.67% ###
    [ 20, 21)  0   0.00%  91.67%
    [ 21, 22) 15   5.00%  96.67% ##
    [ 22, 23)  0   0.00%  96.67%
    [ 23, 24)  8   2.67%  99.33% #
    [ 24, 25)  0   0.00%  99.33%
    [ 25, 25]  2   0.67% 100.00%
    
    Depth by leafs:
    Count: 2430 Average: 3.37325 StdDev: 1.06467
    Min: 1 Max: 7 Ignored: 0
    ----------------------------------------------
    [ 1, 2)  11   0.45%   0.45%
    [ 2, 3) 583  23.99%  24.44% ########
    [ 3, 4) 731  30.08%  54.53% ##########
    [ 4, 5) 760  31.28%  85.80% ##########
    [ 5, 6) 291  11.98%  97.78% ####
    [ 6, 7)  46   1.89%  99.67% #
    [ 7, 7]   8   0.33% 100.00%
    
    Number of training obs by leaf:
    Count: 2430 Average: 29.7531 StdDev: 32.3951
    Min: 5 Max: 121 Ignored: 0
    ----------------------------------------------
    [   5,  10) 1239  50.99%  50.99% ##########
    [  10,  16)  135   5.56%  56.54% #
    [  16,  22)   96   3.95%  60.49% #
    [  22,  28)  100   4.12%  64.61% #
    [  28,  34)  109   4.49%  69.09% #
    [  34,  40)   95   3.91%  73.00% #
    [  40,  45)   41   1.69%  74.69%
    [  45,  51)   21   0.86%  75.56%
    [  51,  57)    7   0.29%  75.84%
    [  57,  63)   17   0.70%  76.54%
    [  63,  69)   39   1.60%  78.15%
    [  69,  75)   65   2.67%  80.82% #
    [  75,  81)  119   4.90%  85.72% #
    [  81,  86)  102   4.20%  89.92% #
    [  86,  92)  107   4.40%  94.32% #
    [  92,  98)   67   2.76%  97.08% #
    [  98, 104)   44   1.81%  98.89%
    [ 104, 110)   18   0.74%  99.63%
    [ 110, 116)    8   0.33%  99.96%
    [ 116, 121]    1   0.04% 100.00%
    
    Attribute in nodes:
    	666 : bill_length_mm [NUMERICAL]
    	403 : flipper_length_mm [NUMERICAL]
    	400 : bill_depth_mm [NUMERICAL]
    	330 : body_mass_g [NUMERICAL]
    	284 : island [CATEGORICAL]
    	30 : sex [CATEGORICAL]
    	17 : year [NUMERICAL]
    
    Attribute in nodes with depth <= 0:
    	159 : flipper_length_mm [NUMERICAL]
    	69 : bill_length_mm [NUMERICAL]
    	60 : bill_depth_mm [NUMERICAL]
    	9 : body_mass_g [NUMERICAL]
    	3 : island [CATEGORICAL]
    
    Attribute in nodes with depth <= 1:
    	239 : bill_length_mm [NUMERICAL]
    	233 : flipper_length_mm [NUMERICAL]
    	178 : bill_depth_mm [NUMERICAL]
    	140 : island [CATEGORICAL]
    	99 : body_mass_g [NUMERICAL]
    
    Attribute in nodes with depth <= 2:
    	427 : bill_length_mm [NUMERICAL]
    	312 : flipper_length_mm [NUMERICAL]
    	280 : bill_depth_mm [NUMERICAL]
    	244 : island [CATEGORICAL]
    	212 : body_mass_g [NUMERICAL]
    	6 : sex [CATEGORICAL]
    	3 : year [NUMERICAL]
    
    Attribute in nodes with depth <= 3:
    	580 : bill_length_mm [NUMERICAL]
    	380 : flipper_length_mm [NUMERICAL]
    	368 : bill_depth_mm [NUMERICAL]
    	302 : body_mass_g [NUMERICAL]
    	278 : island [CATEGORICAL]
    	25 : sex [CATEGORICAL]
    	10 : year [NUMERICAL]
    
    Attribute in nodes with depth <= 5:
    	663 : bill_length_mm [NUMERICAL]
    	403 : flipper_length_mm [NUMERICAL]
    	400 : bill_depth_mm [NUMERICAL]
    	330 : body_mass_g [NUMERICAL]
    	284 : island [CATEGORICAL]
    	30 : sex [CATEGORICAL]
    	16 : year [NUMERICAL]
    
    Condition type in nodes:
    	1816 : HigherCondition
    	314 : ContainsBitmapCondition
    Condition type in nodes with depth <= 0:
    	297 : HigherCondition
    	3 : ContainsBitmapCondition
    Condition type in nodes with depth <= 1:
    	749 : HigherCondition
    	140 : ContainsBitmapCondition
    Condition type in nodes with depth <= 2:
    	1234 : HigherCondition
    	250 : ContainsBitmapCondition
    Condition type in nodes with depth <= 3:
    	1640 : HigherCondition
    	303 : ContainsBitmapCondition
    Condition type in nodes with depth <= 5:
    	1812 : HigherCondition
    	314 : ContainsBitmapCondition
    Node format: NOT_SET
    
    Training OOB:
    	trees: 1, Out-of-bag evaluation: accuracy:0.965116 logloss:1.25734
    	trees: 11, Out-of-bag evaluation: accuracy:0.966805 logloss:0.229339
    	trees: 21, Out-of-bag evaluation: accuracy:0.975104 logloss:0.0891776
    	trees: 35, Out-of-bag evaluation: accuracy:0.975104 logloss:0.0891645
    	trees: 50, Out-of-bag evaluation: accuracy:0.975104 logloss:0.0943436
    	trees: 60, Out-of-bag evaluation: accuracy:0.983402 logloss:0.0839909
    	trees: 70, Out-of-bag evaluation: accuracy:0.983402 logloss:0.088746
    	trees: 80, Out-of-bag evaluation: accuracy:0.966805 logloss:0.0902204
    	trees: 91, Out-of-bag evaluation: accuracy:0.966805 logloss:0.0906052
    	trees: 103, Out-of-bag evaluation: accuracy:0.966805 logloss:0.0935805
    	trees: 114, Out-of-bag evaluation: accuracy:0.966805 logloss:0.0916048
    	trees: 126, Out-of-bag evaluation: accuracy:0.966805 logloss:0.0946431
    	trees: 136, Out-of-bag evaluation: accuracy:0.966805 logloss:0.0929181
    	trees: 146, Out-of-bag evaluation: accuracy:0.970954 logloss:0.093388
    	trees: 157, Out-of-bag evaluation: accuracy:0.970954 logloss:0.095734
    	trees: 167, Out-of-bag evaluation: accuracy:0.975104 logloss:0.0956275
    	trees: 177, Out-of-bag evaluation: accuracy:0.975104 logloss:0.0953425
    	trees: 188, Out-of-bag evaluation: accuracy:0.975104 logloss:0.0962487
    	trees: 199, Out-of-bag evaluation: accuracy:0.975104 logloss:0.0965289
    	trees: 209, Out-of-bag evaluation: accuracy:0.975104 logloss:0.0953626
    	trees: 220, Out-of-bag evaluation: accuracy:0.975104 logloss:0.0954487
    	trees: 230, Out-of-bag evaluation: accuracy:0.975104 logloss:0.0959775
    	trees: 240, Out-of-bag evaluation: accuracy:0.975104 logloss:0.095632
    	trees: 250, Out-of-bag evaluation: accuracy:0.975104 logloss:0.0949993
    	trees: 260, Out-of-bag evaluation: accuracy:0.979253 logloss:0.0950653
    	trees: 272, Out-of-bag evaluation: accuracy:0.979253 logloss:0.0948893
    	trees: 282, Out-of-bag evaluation: accuracy:0.979253 logloss:0.0945184
    	trees: 293, Out-of-bag evaluation: accuracy:0.975104 logloss:0.0947919
    	trees: 300, Out-of-bag evaluation: accuracy:0.979253 logloss:0.0936497
    
    

The information in ``summary`` are all available programmatically using the model inspector:


```python
# The input features
model_1.make_inspector().features()
```




    ["bill_depth_mm" (1; #1),
     "bill_length_mm" (1; #2),
     "body_mass_g" (1; #3),
     "flipper_length_mm" (1; #4),
     "island" (4; #5),
     "sex" (4; #6),
     "year" (1; #7)]




```python
# The feature importances
model_1.make_inspector().variable_importances()
```




    {'INV_MEAN_MIN_DEPTH': [("flipper_length_mm" (1; #4), 0.44677653630204406),
      ("bill_length_mm" (1; #2), 0.4145407654606358),
      ("bill_depth_mm" (1; #1), 0.31456768665853035),
      ("island" (4; #5), 0.30541448366912655),
      ("body_mass_g" (1; #3), 0.2708693342923087),
      ("sex" (4; #6), 0.23473453363984986),
      ("year" (1; #7), 0.2333118557714748)],
     'SUM_SCORE': [("flipper_length_mm" (1; #4), 24616.837831266224),
      ("bill_length_mm" (1; #2), 22270.0546827456),
      ("bill_depth_mm" (1; #1), 10999.832415618002),
      ("island" (4; #5), 8892.439132906497),
      ("body_mass_g" (1; #3), 3670.3080693744123),
      ("sex" (4; #6), 209.51033288240433),
      ("year" (1; #7), 27.375907368957996)],
     'NUM_AS_ROOT': [("flipper_length_mm" (1; #4), 159.0),
      ("bill_length_mm" (1; #2), 69.0),
      ("bill_depth_mm" (1; #1), 60.0),
      ("body_mass_g" (1; #3), 9.0),
      ("island" (4; #5), 3.0)],
     'NUM_NODES': [("bill_length_mm" (1; #2), 666.0),
      ("flipper_length_mm" (1; #4), 403.0),
      ("bill_depth_mm" (1; #1), 400.0),
      ("body_mass_g" (1; #3), 330.0),
      ("island" (4; #5), 284.0),
      ("sex" (4; #6), 30.0),
      ("year" (1; #7), 17.0)]}



The content of the summary and the inspector depends on the learning algorithm (`tfdf.keras.RandomForestModel` in this case) and its hyper-parameters (e.g. `compute_oob_variable_importances=True` will trigger the computation of Out-of-bag variable importances for the Random Forest learner).

## Model Self Evaluation

During training TFDF models can self evaluate even if no validation dataset is provided to the `fit()` method. The exact logic depends on the model. For example, Random Forest will use Out-of-bag evaluation while Gradient Boosted Trees will use internal train-validation.

**Note:** While this evaluation is  computed during training, it is NOT computed on the training dataset and can be used as a low quality evaluation.

The model self evaluation is available with the inspector's `evaluation()`:


```python
model_1.make_inspector().evaluation()
```




    Evaluation(num_examples=241, accuracy=0.979253112033195, loss=0.09364969250108444, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)



## Plotting the training logs

The training logs show the quality of the model (e.g. accuracy evaluated on the out-of-bag or validation dataset) according to the number of trees in the model. These logs are helpful to study the balance between model size and model quality.

The logs are available in multiple ways:

1. Displayed in during training if `fit()` is wrapped in `with sys_pipes():` (see example above).
1. At the end of the model summary i.e. `model.summary()` (see example above).
1. Programmatically, using the model inspector i.e. `model.make_inspector().training_logs()`.
1. Using [TensorBoard](https://www.tensorflow.org/tensorboard)

Let's try the options 2 and 3:



```python
%set_cell_height 150
model_1.make_inspector().training_logs()
```


    <IPython.core.display.Javascript object>





    [TrainLog(num_trees=1, evaluation=Evaluation(num_examples=86, accuracy=0.9651162790697675, loss=1.2573366830515307, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)),
     TrainLog(num_trees=11, evaluation=Evaluation(num_examples=241, accuracy=0.966804979253112, loss=0.22933945183437396, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)),
     TrainLog(num_trees=21, evaluation=Evaluation(num_examples=241, accuracy=0.975103734439834, loss=0.08917763452064942, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)),
     TrainLog(num_trees=35, evaluation=Evaluation(num_examples=241, accuracy=0.975103734439834, loss=0.08916448127802971, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)),
     TrainLog(num_trees=50, evaluation=Evaluation(num_examples=241, accuracy=0.975103734439834, loss=0.09434359505216115, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)),
     TrainLog(num_trees=60, evaluation=Evaluation(num_examples=241, accuracy=0.983402489626556, loss=0.08399087594072353, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)),
     TrainLog(num_trees=70, evaluation=Evaluation(num_examples=241, accuracy=0.983402489626556, loss=0.0887460284615206, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)),
     TrainLog(num_trees=80, evaluation=Evaluation(num_examples=241, accuracy=0.966804979253112, loss=0.0902203571097732, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)),
     TrainLog(num_trees=91, evaluation=Evaluation(num_examples=241, accuracy=0.966804979253112, loss=0.09060522603234315, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)),
     TrainLog(num_trees=103, evaluation=Evaluation(num_examples=241, accuracy=0.966804979253112, loss=0.0935804553608182, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)),
     TrainLog(num_trees=114, evaluation=Evaluation(num_examples=241, accuracy=0.966804979253112, loss=0.09160484072918466, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)),
     TrainLog(num_trees=126, evaluation=Evaluation(num_examples=241, accuracy=0.966804979253112, loss=0.09464313228193408, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)),
     TrainLog(num_trees=136, evaluation=Evaluation(num_examples=241, accuracy=0.966804979253112, loss=0.09291811574142256, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)),
     TrainLog(num_trees=146, evaluation=Evaluation(num_examples=241, accuracy=0.970954356846473, loss=0.09338802008027852, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)),
     TrainLog(num_trees=157, evaluation=Evaluation(num_examples=241, accuracy=0.970954356846473, loss=0.09573401113454237, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)),
     TrainLog(num_trees=167, evaluation=Evaluation(num_examples=241, accuracy=0.975103734439834, loss=0.09562747313783995, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)),
     TrainLog(num_trees=177, evaluation=Evaluation(num_examples=241, accuracy=0.975103734439834, loss=0.09534245911073636, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)),
     TrainLog(num_trees=188, evaluation=Evaluation(num_examples=241, accuracy=0.975103734439834, loss=0.0962487407245082, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)),
     TrainLog(num_trees=199, evaluation=Evaluation(num_examples=241, accuracy=0.975103734439834, loss=0.09652885899365071, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)),
     TrainLog(num_trees=209, evaluation=Evaluation(num_examples=241, accuracy=0.975103734439834, loss=0.0953625899225597, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)),
     TrainLog(num_trees=220, evaluation=Evaluation(num_examples=241, accuracy=0.975103734439834, loss=0.09544872451589068, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)),
     TrainLog(num_trees=230, evaluation=Evaluation(num_examples=241, accuracy=0.975103734439834, loss=0.09597752142805777, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)),
     TrainLog(num_trees=240, evaluation=Evaluation(num_examples=241, accuracy=0.975103734439834, loss=0.09563197056411213, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)),
     TrainLog(num_trees=250, evaluation=Evaluation(num_examples=241, accuracy=0.975103734439834, loss=0.0949992554965603, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)),
     TrainLog(num_trees=260, evaluation=Evaluation(num_examples=241, accuracy=0.979253112033195, loss=0.09506525110484282, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)),
     TrainLog(num_trees=272, evaluation=Evaluation(num_examples=241, accuracy=0.979253112033195, loss=0.09488926957751706, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)),
     TrainLog(num_trees=282, evaluation=Evaluation(num_examples=241, accuracy=0.979253112033195, loss=0.09451841080708127, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)),
     TrainLog(num_trees=293, evaluation=Evaluation(num_examples=241, accuracy=0.975103734439834, loss=0.09479190538641205, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)),
     TrainLog(num_trees=300, evaluation=Evaluation(num_examples=241, accuracy=0.979253112033195, loss=0.09364969250108444, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None))]



Let's plot it:


```python
import matplotlib.pyplot as plt

logs = model_1.make_inspector().training_logs()

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Accuracy (out-of-bag)")

plt.subplot(1, 2, 2)
plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Logloss (out-of-bag)")

plt.show()
```


    
![png](../images
/beginner_colab_img.png)
    


This dataset is small. You can see the model converging almost immediately.

Let's use TensorBoard:


```python
# This cell start TensorBoard that can be slow.
# Load the TensorBoard notebook extension
%load_ext tensorboard
# Google internal version
# %load_ext google3.learning.brain.tensorboard.notebook.extension
```


```python
# Clear existing results (if any)
!rm -fr "/tmp/tensorboard_logs"
```


```python
# Export the meta-data to tensorboard.
model_1.make_inspector().export_to_tensorboard("/tmp/tensorboard_logs")
```


```python
# docs_infra: no_execute
# Start a tensorboard instance.
%tensorboard --logdir "/tmp/tensorboard_logs"
```

<!-- <img class="tfo-display-only-on-site" src="images/beginner_tensorboard.png"/> -->


## Re-train the model with a different learning algorithm

The learning algorithm is defined by the model class. For
example, `tfdf.keras.RandomForestModel()` trains a Random Forest, while
`tfdf.keras.GradientBoostedTreesModel()` trains a Gradient Boosted Decision
Trees.

The learning algorithms are listed by calling `tfdf.keras.get_all_models()` or in the
[learner list](https://github.com/google/yggdrasil-decision-forests/blob/main/documentation/learners.md).


```python
tfdf.keras.get_all_models()
```




    [tensorflow_decision_forests.keras.RandomForestModel,
     tensorflow_decision_forests.keras.GradientBoostedTreesModel,
     tensorflow_decision_forests.keras.CartModel,
     tensorflow_decision_forests.keras.DistributedGradientBoostedTreesModel]



The description of the learning algorithms and their hyper-parameters are also available in the [API reference](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf) and builtin help:


```python
# help works anywhere.
help(tfdf.keras.RandomForestModel)

# ? only works in ipython or notebooks, it usually opens on a separate panel.
tfdf.keras.RandomForestModel?
```

    Help on class RandomForestModel in module tensorflow_decision_forests.keras:
    
    class RandomForestModel(tensorflow_decision_forests.keras.wrappers.RandomForestModel)
     |  RandomForestModel(*args, **kwargs)
     |  
     |  Method resolution order:
     |      RandomForestModel
     |      tensorflow_decision_forests.keras.wrappers.RandomForestModel
     |      tensorflow_decision_forests.keras.core.CoreModel
     |      tensorflow_decision_forests.keras.core_inference.InferenceCoreModel
     |      keras.engine.training.Model
     |      keras.engine.base_layer.Layer
     |      tensorflow.python.module.module.Module
     |      tensorflow.python.trackable.autotrackable.AutoTrackable
     |      tensorflow.python.trackable.base.Trackable
     |      keras.utils.version_utils.LayerVersionSelector
     |      keras.utils.version_utils.ModelVersionSelector
     |      builtins.object
     |  
     |  Methods inherited from tensorflow_decision_forests.keras.wrappers.RandomForestModel:
     |  
     |  __init__(self, task: Optional[ForwardRef('abstract_model_pb2.Task')] = 1, features: Optional[List[tensorflow_decision_forests.keras.core.FeatureUsage]] = None, exclude_non_specified_features: Optional[bool] = False, preprocessing: Optional[ForwardRef('tf.keras.models.Functional')] = None, postprocessing: Optional[ForwardRef('tf.keras.models.Functional')] = None, ranking_group: Optional[str] = None, uplift_treatment: Optional[str] = None, temp_directory: Optional[str] = None, verbose: int = 1, hyperparameter_template: Optional[str] = None, advanced_arguments: Optional[tensorflow_decision_forests.keras.core_inference.AdvancedArguments] = None, num_threads: Optional[int] = None, name: Optional[str] = None, max_vocab_count: Optional[int] = 2000, try_resume_training: Optional[bool] = True, check_dataset: Optional[bool] = True, tuner: Optional[tensorflow_decision_forests.component.tuner.tuner.Tuner] = None, discretize_numerical_features: bool = False, num_discretized_numerical_bins: int = 255, multitask: Optional[List[tensorflow_decision_forests.keras.core_inference.MultiTaskItem]] = None, adapt_bootstrap_size_ratio_for_maximum_training_duration: Optional[bool] = False, allow_na_conditions: Optional[bool] = False, bootstrap_size_ratio: Optional[float] = 1.0, bootstrap_training_dataset: Optional[bool] = True, categorical_algorithm: Optional[str] = 'CART', categorical_set_split_greedy_sampling: Optional[float] = 0.1, categorical_set_split_max_num_items: Optional[int] = -1, categorical_set_split_min_item_frequency: Optional[int] = 1, compute_oob_performances: Optional[bool] = True, compute_oob_variable_importances: Optional[bool] = False, growing_strategy: Optional[str] = 'LOCAL', honest: Optional[bool] = False, honest_fixed_separation: Optional[bool] = False, honest_ratio_leaf_examples: Optional[float] = 0.5, in_split_min_examples_check: Optional[bool] = True, keep_non_leaf_label_distribution: Optional[bool] = True, max_depth: Optional[int] = 16, max_num_nodes: Optional[int] = None, maximum_model_size_in_memory_in_bytes: Optional[float] = -1.0, maximum_training_duration_seconds: Optional[float] = -1.0, min_examples: Optional[int] = 5, missing_value_policy: Optional[str] = 'GLOBAL_IMPUTATION', num_candidate_attributes: Optional[int] = 0, num_candidate_attributes_ratio: Optional[float] = -1.0, num_oob_variable_importances_permutations: Optional[int] = 1, num_trees: Optional[int] = 300, pure_serving_model: Optional[bool] = False, random_seed: Optional[int] = 123456, sampling_with_replacement: Optional[bool] = True, sorting_strategy: Optional[str] = 'PRESORT', sparse_oblique_normalization: Optional[str] = None, sparse_oblique_num_projections_exponent: Optional[float] = None, sparse_oblique_projection_density_factor: Optional[float] = None, sparse_oblique_weights: Optional[str] = None, split_axis: Optional[str] = 'AXIS_ALIGNED', uplift_min_examples_in_treatment: Optional[int] = 5, uplift_split_score: Optional[str] = 'KULLBACK_LEIBLER', winner_take_all: Optional[bool] = True, explicit_args: Optional[Set[str]] = None)
     |  
     |  ----------------------------------------------------------------------
     |  Static methods inherited from tensorflow_decision_forests.keras.wrappers.RandomForestModel:
     |  
     |  capabilities() -> yggdrasil_decision_forests.learner.abstract_learner_pb2.LearnerCapabilities
     |      Lists the capabilities of the learning algorithm.
     |  
     |  predefined_hyperparameters() -> List[tensorflow_decision_forests.keras.core.HyperParameterTemplate]
     |      Returns a better than default set of hyper-parameters.
     |      
     |      They can be used directly with the `hyperparameter_template` argument of the
     |      model constructor.
     |      
     |      These hyper-parameters outperform the default hyper-parameters (either
     |      generally or in specific scenarios). Like default hyper-parameters, existing
     |      pre-defined hyper-parameters cannot change.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from tensorflow_decision_forests.keras.core.CoreModel:
     |  
     |  collect_data_step(self, data, is_training_example)
     |      Collect examples e.g. training or validation.
     |  
     |  fit(self, x=None, y=None, callbacks=None, verbose: Optional[Any] = None, validation_steps: Optional[int] = None, validation_data: Optional[Any] = None, sample_weight: Optional[Any] = None, steps_per_epoch: Optional[Any] = None, class_weight: Optional[Any] = None, **kwargs) -> keras.callbacks.History
     |      Trains the model.
     |      
     |      Local training
     |      ==============
     |      
     |      It is recommended to use a Pandas Dataframe dataset and to convert it to
     |      a TensorFlow dataset with `pd_dataframe_to_tf_dataset()`:
     |        ```python
     |        pd_dataset = pandas.Dataframe(...)
     |        tf_dataset = pd_dataframe_to_tf_dataset(dataset, label="my_label")
     |        model.fit(pd_dataset)
     |        ```
     |      
     |      The following dataset formats are supported:
     |      
     |        1. "x" is a `tf.data.Dataset` containing a tuple "(features, labels)".
     |           "features" can be a dictionary a tensor, a list of tensors or a
     |           dictionary of tensors (recommended). "labels" is a tensor.
     |      
     |        2. "x" is a tensor, list of tensors or dictionary of tensors containing
     |           the input features. "y" is a tensor.
     |      
     |        3. "x" is a numpy-array, list of numpy-arrays or dictionary of
     |           numpy-arrays containing the input features. "y" is a numpy-array.
     |      
     |      IMPORTANT: This model trains on the entire dataset at once. This has the
     |      following consequences:
     |      
     |        1. The dataset need to be read exactly once. If you use a TensorFlow
     |           dataset, make sure NOT to add a "repeat" operation.
     |        2. The algorithm does not benefit from shuffling the dataset. If you use a
     |           TensorFlow dataset, make sure NOT to add a "shuffle" operation.
     |        3. The dataset needs to be batched (i.e. with a "batch" operation).
     |           However, the number of elements per batch has not impact on the model.
     |           Generally, it is recommended to use batches as large as possible as its
     |           speeds-up reading the dataset in TensorFlow.
     |      
     |      Input features do not need to be normalized (e.g. dividing numerical values
     |      by the variance) or indexed (e.g. replacing categorical string values by
     |      an integer). Additionally, missing values can be consumed natively.
     |      
     |      Distributed training
     |      ====================
     |      
     |      Some of the learning algorithms will support distributed training with the
     |      ParameterServerStrategy.
     |      
     |      In this case, the dataset is read asynchronously in between the workers. The
     |      distribution of the training depends on the learning algorithm.
     |      
     |      Like for non-distributed training, the dataset should be read exactly once.
     |      The simplest solution is to divide the dataset into different files (i.e.
     |      shards) and have each of the worker read a non overlapping subset of shards.
     |      
     |      IMPORTANT: The training dataset should not be infinite i.e. the training
     |      dataset should not contain any repeat operation.
     |      
     |      Currently (to be changed), the validation dataset (if provided) is simply
     |      feed to the `model.evaluate()` method. Therefore, it should satisfy Keras'
     |      evaluate API. Notably, for distributed training, the validation dataset
     |      should be infinite (i.e. have a repeat operation).
     |      
     |      See https://www.tensorflow.org/decision_forests/distributed_training for
     |      more details and examples.
     |      
     |      Here is a single example of distributed training using PSS for both dataset
     |      reading and training distribution.
     |      
     |        ```python
     |        def dataset_fn(context, paths, training=True):
     |          ds_path = tf.data.Dataset.from_tensor_slices(paths)
     |      
     |      
     |          if context is not None:
     |            # Train on at least 2 workers.
     |            current_worker = tfdf.keras.get_worker_idx_and_num_workers(context)
     |            assert current_worker.num_workers > 2
     |      
     |            # Split the dataset's examples among the workers.
     |            ds_path = ds_path.shard(
     |                num_shards=current_worker.num_workers,
     |                index=current_worker.worker_idx)
     |      
     |          def read_csv_file(path):
     |            numerical = tf.constant([math.nan], dtype=tf.float32)
     |            categorical_string = tf.constant([""], dtype=tf.string)
     |            csv_columns = [
     |                numerical,  # age
     |                categorical_string,  # workclass
     |                numerical,  # fnlwgt
     |                ...
     |            ]
     |            column_names = [
     |              "age", "workclass", "fnlwgt", ...
     |            ]
     |            label_name = "label"
     |            return tf.data.experimental.CsvDataset(path, csv_columns, header=True)
     |      
     |          ds_columns = ds_path.interleave(read_csv_file)
     |      
     |          def map_features(*columns):
     |            assert len(column_names) == len(columns)
     |            features = {column_names[i]: col for i, col in enumerate(columns)}
     |            label = label_table.lookup(features.pop(label_name))
     |            return features, label
     |      
     |          ds_dataset = ds_columns.map(map_features)
     |          if not training:
     |            dataset = dataset.repeat(None)
     |          ds_dataset = ds_dataset.batch(batch_size)
     |          return ds_dataset
     |      
     |        strategy = tf.distribute.experimental.ParameterServerStrategy(...)
     |        sharded_train_paths = [list of dataset files]
     |        with strategy.scope():
     |          model = DistributedGradientBoostedTreesModel()
     |          train_dataset = strategy.distribute_datasets_from_function(
     |            lambda context: dataset_fn(context, sharded_train_paths))
     |      
     |          test_dataset = strategy.distribute_datasets_from_function(
     |            lambda context: dataset_fn(context, sharded_test_paths))
     |      
     |        model.fit(sharded_train_paths)
     |        evaluation = model.evaluate(test_dataset, steps=num_test_examples //
     |          batch_size)
     |        ```
     |      
     |      Args:
     |        x: Training dataset (See details above for the supported formats).
     |        y: Label of the training dataset. Only used if "x" does not contains the
     |          labels.
     |        callbacks: Callbacks triggered during the training. The training runs in a
     |          single epoch, itself run in a single step. Therefore, callback logic can
     |          be called equivalently before/after the fit function.
     |        verbose: Verbosity mode. 0 = silent, 1 = small details, 2 = full details.
     |        validation_steps: Number of steps in the evaluation dataset when
     |          evaluating the trained model with `model.evaluate()`. If not specified,
     |          evaluates the model on the entire dataset (generally recommended; not
     |          yet supported for distributed datasets).
     |        validation_data: Validation dataset. If specified, the learner might use
     |          this dataset to help training e.g. early stopping.
     |        sample_weight: Training weights. Note: training weights can also be
     |          provided as the third output in a `tf.data.Dataset` e.g. (features,
     |          label, weights).
     |        steps_per_epoch: [Parameter will be removed] Number of training batch to
     |          load before training the model. Currently, only supported for
     |          distributed training.
     |        class_weight: For binary classification only. Mapping class indices
     |          (integers) to a weight (float) value. Only available for non-Distributed
     |          training. For maximum compatibility, feed example weights through the
     |          tf.data.Dataset or using the `weight` argument of
     |          `pd_dataframe_to_tf_dataset`.
     |        **kwargs: Extra arguments passed to the core keras model's fit. Note that
     |          not all keras' model fit arguments are supported.
     |      
     |      Returns:
     |        A `History` object. Its `History.history` attribute is not yet
     |        implemented for decision forests algorithms, and will return empty.
     |        All other fields are filled as usual for `Keras.Mode.fit()`.
     |  
     |  fit_on_dataset_path(self, train_path: str, label_key: Optional[str] = None, weight_key: Optional[str] = None, valid_path: Optional[str] = None, dataset_format: Optional[str] = 'csv', max_num_scanned_rows_to_accumulate_statistics: Optional[int] = 100000, try_resume_training: Optional[bool] = True, input_model_signature_fn: Optional[Callable[[tensorflow_decision_forests.component.inspector.inspector.AbstractInspector], Any]] = <function build_default_input_model_signature at 0x7ff38ea010d0>, num_io_threads: int = 10)
     |      Trains the model on a dataset stored on disk.
     |      
     |      This solution is generally more efficient and easier than loading the
     |      dataset with a `tf.Dataset` both for local and distributed training.
     |      
     |      Usage example:
     |      
     |        # Local training
     |        ```python
     |        model = keras.GradientBoostedTreesModel()
     |        model.fit_on_dataset_path(
     |          train_path="/path/to/dataset.csv",
     |          label_key="label",
     |          dataset_format="csv")
     |        model.save("/model/path")
     |        ```
     |      
     |        # Distributed training
     |        ```python
     |        with tf.distribute.experimental.ParameterServerStrategy(...).scope():
     |          model = model = keras.DistributedGradientBoostedTreesModel()
     |        model.fit_on_dataset_path(
     |          train_path="/path/to/dataset@10",
     |          label_key="label",
     |          dataset_format="tfrecord+tfe")
     |        model.save("/model/path")
     |        ```
     |      
     |      Args:
     |        train_path: Path to the training dataset. Supports comma separated files,
     |          shard and glob notation.
     |        label_key: Name of the label column.
     |        weight_key: Name of the weighing column.
     |        valid_path: Path to the validation dataset. If not provided, or if the
     |          learning algorithm does not supports/needs a validation dataset,
     |          `valid_path` is ignored.
     |        dataset_format: Format of the dataset. Should be one of the registered
     |          dataset format (see [User
     |          Manual](https://github.com/google/yggdrasil-decision-forests/blob/main/documentation/user_manual.md#dataset-path-and-format)
     |          for more details). The format "csv" is always available but it is
     |          generally only suited for small datasets.
     |        max_num_scanned_rows_to_accumulate_statistics: Maximum number of examples
     |          to scan to determine the statistics of the features (i.e. the dataspec,
     |          e.g. mean value, dictionaries). (Currently) the "first" examples of the
     |          dataset are scanned (e.g. the first examples of the dataset is a single
     |          file). Therefore, it is important that the sampled dataset is relatively
     |          uniformly sampled, notably the scanned examples should contains all the
     |          possible categorical values (otherwise the not seen value will be
     |          treated as out-of-vocabulary). If set to None, the entire dataset is
     |          scanned. This parameter has no effect if the dataset is stored in a
     |          format that already contains those values.
     |        try_resume_training: If true, tries to resume training from the model
     |          checkpoint stored in the `temp_directory` directory. If `temp_directory`
     |          does not contain any model checkpoint, start the training from the
     |          start. Works in the following three situations: (1) The training was
     |          interrupted by the user (e.g. ctrl+c). (2) the training job was
     |          interrupted (e.g. rescheduling), ond (3) the hyper-parameter of the
     |          model were changed such that an initially completed training is now
     |          incomplete (e.g. increasing the number of trees).
     |        input_model_signature_fn: A lambda that returns the
     |          (Dense,Sparse,Ragged)TensorSpec (or structure of TensorSpec e.g.
     |          dictionary, list) corresponding to input signature of the model. If not
     |          specified, the input model signature is created by
     |          `build_default_input_model_signature`. For example, specify
     |          `input_model_signature_fn` if an numerical input feature (which is
     |          consumed as DenseTensorSpec(float32) by default) will be feed
     |          differently (e.g. RaggedTensor(int64)).
     |        num_io_threads: Number of threads to use for IO operations e.g. reading a
     |          dataset from disk. Increasing this value can speed-up IO operations when
     |          IO operations are either latency or cpu bounded.
     |      
     |      Returns:
     |        A `History` object. Its `History.history` attribute is not yet
     |        implemented for decision forests algorithms, and will return empty.
     |        All other fields are filled as usual for `Keras.Mode.fit()`.
     |  
     |  load_weights(self, *args, **kwargs)
     |      No-op for TensorFlow Decision Forests models.
     |      
     |      `load_weights` is not supported by TensorFlow Decision Forests models.
     |      To save and restore a model, use the SavedModel API i.e.
     |      `model.save(...)` and `tf.keras.models.load_model(...)`. To resume the
     |      training of an existing model, create the model with
     |      `try_resume_training=True` (default value) and with a similar
     |      `temp_directory` argument. See documentation of `try_resume_training`
     |      for more details.
     |      
     |      Args:
     |        *args: Passed through to base `keras.Model` implemenation.
     |        **kwargs: Passed through to base `keras.Model` implemenation.
     |  
     |  save(self, filepath: str, overwrite: Optional[bool] = True, **kwargs)
     |      Saves the model as a TensorFlow SavedModel.
     |      
     |      The exported SavedModel contains a standalone Yggdrasil Decision Forests
     |      model in the "assets" sub-directory. The Yggdrasil model can be used
     |      directly using the Yggdrasil API. However, this model does not contain the
     |      "preprocessing" layer (if any).
     |      
     |      Args:
     |        filepath: Path to the output model.
     |        overwrite: If true, override an already existing model. If false, raise an
     |          error if a model already exist.
     |        **kwargs: Arguments passed to the core keras model's save.
     |  
     |  support_distributed_training(self)
     |  
     |  train_on_batch(self, *args, **kwargs)
     |      No supported for Tensorflow Decision Forests models.
     |      
     |      Decision forests are not trained in batches the same way neural networks
     |      are. To avoid confusion, train_on_batch is disabled.
     |      
     |      Args:
     |        *args: Ignored
     |        **kwargs: Ignored.
     |  
     |  train_step(self, data)
     |      Collects training examples.
     |  
     |  valid_step(self, data)
     |      Collects validation examples.
     |  
     |  ----------------------------------------------------------------------
     |  Readonly properties inherited from tensorflow_decision_forests.keras.core.CoreModel:
     |  
     |  exclude_non_specified_features
     |      If true, only use the features specified in "features".
     |  
     |  learner
     |      Name of the learning algorithm used to train the model.
     |  
     |  learner_params
     |      Gets the dictionary of hyper-parameters passed in the model constructor.
     |      
     |      Changing this dictionary will impact the training.
     |  
     |  num_threads
     |      Number of threads used to train the model.
     |  
     |  num_training_examples
     |      Number of training examples.
     |  
     |  num_validation_examples
     |      Number of validation examples.
     |  
     |  training_model_id
     |      Identifier of the model.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from tensorflow_decision_forests.keras.core_inference.InferenceCoreModel:
     |  
     |  call(self, inputs, training=False)
     |      Inference of the model.
     |      
     |      This method is used for prediction and evaluation of a trained model.
     |      
     |      Args:
     |        inputs: Input tensors.
     |        training: Is the model being trained. Always False.
     |      
     |      Returns:
     |        Model predictions.
     |  
     |  call_get_leaves(self, inputs)
     |      Computes the index of the active leaf in each tree.
     |      
     |      The active leaf is the leave that that receive the example during inference.
     |      
     |      The returned value "leaves[i,j]" is the index of the active leave for the
     |      i-th example and the j-th tree. Leaves are indexed by depth first
     |      exploration with the negative child visited before the positive one
     |      (similarly as "iterate_on_nodes()" iteration). Leaf indices are also
     |      available with LeafNode.leaf_idx.
     |      
     |      Args:
     |        inputs: Input tensors. Same signature as the model's "call(inputs)".
     |      
     |      Returns:
     |        Index of the active leaf for each tree in the model.
     |  
     |  compile(self, metrics=None, weighted_metrics=None)
     |      Configure the model for training.
     |      
     |      Unlike for most Keras model, calling "compile" is optional before calling
     |      "fit".
     |      
     |      Args:
     |        metrics: List of metrics to be evaluated by the model during training and
     |          testing.
     |        weighted_metrics: List of metrics to be evaluated and weighted by
     |          `sample_weight` or `class_weight` during training and testing.
     |      
     |      Raises:
     |        ValueError: Invalid arguments.
     |  
     |  make_inspector(self, index: int = 0) -> tensorflow_decision_forests.component.inspector.inspector.AbstractInspector
     |      Creates an inspector to access the internal model structure.
     |      
     |      Usage example:
     |      
     |      ```python
     |      inspector = model.make_inspector()
     |      print(inspector.num_trees())
     |      print(inspector.variable_importances())
     |      ```
     |      
     |      Args:
     |        index: Index of the sub-model. Only used for multitask models.
     |      
     |      Returns:
     |        A model inspector.
     |  
     |  make_predict_function(self)
     |      Prediction of the model (!= evaluation).
     |  
     |  make_test_function(self)
     |      Predictions for evaluation.
     |  
     |  predict_get_leaves(self, x)
     |      Gets the index of the active leaf of each tree.
     |      
     |      The active leaf is the leave that that receive the example during inference.
     |      
     |      The returned value "leaves[i,j]" is the index of the active leave for the
     |      i-th example and the j-th tree. Leaves are indexed by depth first
     |      exploration with the negative child visited before the positive one
     |      (similarly as "iterate_on_nodes()" iteration). Leaf indices are also
     |      available with LeafNode.leaf_idx.
     |      
     |      Args:
     |        x: Input samples as a tf.data.Dataset.
     |      
     |      Returns:
     |        Index of the active leaf for each tree in the model.
     |  
     |  ranking_group(self) -> Optional[str]
     |  
     |  summary(self, line_length=None, positions=None, print_fn=None)
     |      Shows information about the model.
     |  
     |  uplift_treatment(self) -> Optional[str]
     |  
     |  yggdrasil_model_path_tensor(self, multitask_model_index: int = 0) -> Optional[tensorflow.python.framework.ops.Tensor]
     |      Gets the path to yggdrasil model, if available.
     |      
     |      The effective path can be obtained with:
     |      
     |      ```python
     |      yggdrasil_model_path_tensor().numpy().decode("utf-8")
     |      ```
     |      
     |      Args:
     |        multitask_model_index: Index of the sub-model. Only used for multitask
     |          models.
     |      
     |      Returns:
     |        Path to the Yggdrasil model.
     |  
     |  yggdrasil_model_prefix(self, index: int = 0) -> str
     |      Gets the prefix of the internal yggdrasil model.
     |  
     |  ----------------------------------------------------------------------
     |  Readonly properties inherited from tensorflow_decision_forests.keras.core_inference.InferenceCoreModel:
     |  
     |  multitask
     |      Tasks to solve.
     |  
     |  task
     |      Task to solve (e.g. CLASSIFICATION, REGRESSION, RANKING).
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from keras.engine.training.Model:
     |  
     |  __call__(self, *args, **kwargs)
     |  
     |  __copy__(self)
     |  
     |  __deepcopy__(self, memo)
     |  
     |  __reduce__(self)
     |      Helper for pickle.
     |  
     |  __setattr__(self, name, value)
     |      Support self.foo = trackable syntax.
     |  
     |  build(self, input_shape)
     |      Builds the model based on input shapes received.
     |      
     |      This is to be used for subclassed models, which do not know at
     |      instantiation time what their inputs look like.
     |      
     |      This method only exists for users who want to call `model.build()` in a
     |      standalone way (as a substitute for calling the model on real data to
     |      build it). It will never be called by the framework (and thus it will
     |      never throw unexpected errors in an unrelated workflow).
     |      
     |      Args:
     |       input_shape: Single tuple, `TensorShape` instance, or list/dict of
     |         shapes, where shapes are tuples, integers, or `TensorShape`
     |         instances.
     |      
     |      Raises:
     |        ValueError:
     |          1. In case of invalid user-provided data (not of type tuple,
     |             list, `TensorShape`, or dict).
     |          2. If the model requires call arguments that are agnostic
     |             to the input shapes (positional or keyword arg in call
     |             signature).
     |          3. If not all layers were properly built.
     |          4. If float type inputs are not supported within the layers.
     |      
     |        In each of these cases, the user should build their model by calling
     |        it on real tensor data.
     |  
     |  compile_from_config(self, config)
     |  
     |  compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None)
     |      Compute the total loss, validate it, and return it.
     |      
     |      Subclasses can optionally override this method to provide custom loss
     |      computation logic.
     |      
     |      Example:
     |      ```python
     |      class MyModel(tf.keras.Model):
     |      
     |        def __init__(self, *args, **kwargs):
     |          super(MyModel, self).__init__(*args, **kwargs)
     |          self.loss_tracker = tf.keras.metrics.Mean(name='loss')
     |      
     |        def compute_loss(self, x, y, y_pred, sample_weight):
     |          loss = tf.reduce_mean(tf.math.squared_difference(y_pred, y))
     |          loss += tf.add_n(self.losses)
     |          self.loss_tracker.update_state(loss)
     |          return loss
     |      
     |        def reset_metrics(self):
     |          self.loss_tracker.reset_states()
     |      
     |        @property
     |        def metrics(self):
     |          return [self.loss_tracker]
     |      
     |      tensors = tf.random.uniform((10, 10)), tf.random.uniform((10,))
     |      dataset = tf.data.Dataset.from_tensor_slices(tensors).repeat().batch(1)
     |      
     |      inputs = tf.keras.layers.Input(shape=(10,), name='my_input')
     |      outputs = tf.keras.layers.Dense(10)(inputs)
     |      model = MyModel(inputs, outputs)
     |      model.add_loss(tf.reduce_sum(outputs))
     |      
     |      optimizer = tf.keras.optimizers.SGD()
     |      model.compile(optimizer, loss='mse', steps_per_execution=10)
     |      model.fit(dataset, epochs=2, steps_per_epoch=10)
     |      print('My custom loss: ', model.loss_tracker.result().numpy())
     |      ```
     |      
     |      Args:
     |        x: Input data.
     |        y: Target data.
     |        y_pred: Predictions returned by the model (output of `model(x)`)
     |        sample_weight: Sample weights for weighting the loss function.
     |      
     |      Returns:
     |        The total loss as a `tf.Tensor`, or `None` if no loss results (which
     |        is the case when called by `Model.test_step`).
     |  
     |  compute_metrics(self, x, y, y_pred, sample_weight)
     |      Update metric states and collect all metrics to be returned.
     |      
     |      Subclasses can optionally override this method to provide custom metric
     |      updating and collection logic.
     |      
     |      Example:
     |      ```python
     |      class MyModel(tf.keras.Sequential):
     |      
     |        def compute_metrics(self, x, y, y_pred, sample_weight):
     |      
     |          # This super call updates `self.compiled_metrics` and returns
     |          # results for all metrics listed in `self.metrics`.
     |          metric_results = super(MyModel, self).compute_metrics(
     |              x, y, y_pred, sample_weight)
     |      
     |          # Note that `self.custom_metric` is not listed in `self.metrics`.
     |          self.custom_metric.update_state(x, y, y_pred, sample_weight)
     |          metric_results['custom_metric_name'] = self.custom_metric.result()
     |          return metric_results
     |      ```
     |      
     |      Args:
     |        x: Input data.
     |        y: Target data.
     |        y_pred: Predictions returned by the model (output of `model.call(x)`)
     |        sample_weight: Sample weights for weighting the loss function.
     |      
     |      Returns:
     |        A `dict` containing values that will be passed to
     |        `tf.keras.callbacks.CallbackList.on_train_batch_end()`. Typically, the
     |        values of the metrics listed in `self.metrics` are returned. Example:
     |        `{'loss': 0.2, 'accuracy': 0.7}`.
     |  
     |  evaluate(self, x=None, y=None, batch_size=None, verbose='auto', sample_weight=None, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, return_dict=False, **kwargs)
     |      Returns the loss value & metrics values for the model in test mode.
     |      
     |      Computation is done in batches (see the `batch_size` arg.)
     |      
     |      Args:
     |          x: Input data. It could be:
     |            - A Numpy array (or array-like), or a list of arrays
     |              (in case the model has multiple inputs).
     |            - A TensorFlow tensor, or a list of tensors
     |              (in case the model has multiple inputs).
     |            - A dict mapping input names to the corresponding array/tensors,
     |              if the model has named inputs.
     |            - A `tf.data` dataset. Should return a tuple
     |              of either `(inputs, targets)` or
     |              `(inputs, targets, sample_weights)`.
     |            - A generator or `keras.utils.Sequence` returning `(inputs,
     |              targets)` or `(inputs, targets, sample_weights)`.
     |            A more detailed description of unpacking behavior for iterator
     |            types (Dataset, generator, Sequence) is given in the `Unpacking
     |            behavior for iterator-like inputs` section of `Model.fit`.
     |          y: Target data. Like the input data `x`, it could be either Numpy
     |            array(s) or TensorFlow tensor(s). It should be consistent with `x`
     |            (you cannot have Numpy inputs and tensor targets, or inversely).
     |            If `x` is a dataset, generator or `keras.utils.Sequence` instance,
     |            `y` should not be specified (since targets will be obtained from
     |            the iterator/dataset).
     |          batch_size: Integer or `None`. Number of samples per batch of
     |            computation. If unspecified, `batch_size` will default to 32. Do
     |            not specify the `batch_size` if your data is in the form of a
     |            dataset, generators, or `keras.utils.Sequence` instances (since
     |            they generate batches).
     |          verbose: `"auto"`, 0, 1, or 2. Verbosity mode.
     |              0 = silent, 1 = progress bar, 2 = single line.
     |              `"auto"` defaults to 1 for most cases, and to 2 when used with
     |              `ParameterServerStrategy`. Note that the progress bar is not
     |              particularly useful when logged to a file, so `verbose=2` is
     |              recommended when not running interactively (e.g. in a production
     |              environment).
     |          sample_weight: Optional Numpy array of weights for the test samples,
     |            used for weighting the loss function. You can either pass a flat
     |            (1D) Numpy array with the same length as the input samples
     |              (1:1 mapping between weights and samples), or in the case of
     |                temporal data, you can pass a 2D array with shape `(samples,
     |                sequence_length)`, to apply a different weight to every
     |                timestep of every sample. This argument is not supported when
     |                `x` is a dataset, instead pass sample weights as the third
     |                element of `x`.
     |          steps: Integer or `None`. Total number of steps (batches of samples)
     |            before declaring the evaluation round finished. Ignored with the
     |            default value of `None`. If x is a `tf.data` dataset and `steps`
     |            is None, 'evaluate' will run until the dataset is exhausted. This
     |            argument is not supported with array inputs.
     |          callbacks: List of `keras.callbacks.Callback` instances. List of
     |            callbacks to apply during evaluation. See
     |            [callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks).
     |          max_queue_size: Integer. Used for generator or
     |            `keras.utils.Sequence` input only. Maximum size for the generator
     |            queue. If unspecified, `max_queue_size` will default to 10.
     |          workers: Integer. Used for generator or `keras.utils.Sequence` input
     |            only. Maximum number of processes to spin up when using
     |            process-based threading. If unspecified, `workers` will default to
     |            1.
     |          use_multiprocessing: Boolean. Used for generator or
     |            `keras.utils.Sequence` input only. If `True`, use process-based
     |            threading. If unspecified, `use_multiprocessing` will default to
     |            `False`. Note that because this implementation relies on
     |            multiprocessing, you should not pass non-picklable arguments to
     |            the generator as they can't be passed easily to children
     |            processes.
     |          return_dict: If `True`, loss and metric results are returned as a
     |            dict, with each key being the name of the metric. If `False`, they
     |            are returned as a list.
     |          **kwargs: Unused at this time.
     |      
     |      See the discussion of `Unpacking behavior for iterator-like inputs` for
     |      `Model.fit`.
     |      
     |      Returns:
     |          Scalar test loss (if the model has a single output and no metrics)
     |          or list of scalars (if the model has multiple outputs
     |          and/or metrics). The attribute `model.metrics_names` will give you
     |          the display labels for the scalar outputs.
     |      
     |      Raises:
     |          RuntimeError: If `model.evaluate` is wrapped in a `tf.function`.
     |  
     |  evaluate_generator(self, generator, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
     |      Evaluates the model on a data generator.
     |      
     |      DEPRECATED:
     |        `Model.evaluate` now supports generators, so there is no longer any
     |        need to use this endpoint.
     |  
     |  export(self, filepath)
     |      Create a SavedModel artifact for inference (e.g. via TF-Serving).
     |      
     |      This method lets you export a model to a lightweight SavedModel artifact
     |      that contains the model's forward pass only (its `call()` method)
     |      and can be served via e.g. TF-Serving. The forward pass is registered
     |      under the name `serve()` (see example below).
     |      
     |      The original code of the model (including any custom layers you may
     |      have used) is *no longer* necessary to reload the artifact -- it is
     |      entirely standalone.
     |      
     |      Args:
     |          filepath: `str` or `pathlib.Path` object. Path where to save
     |              the artifact.
     |      
     |      Example:
     |      
     |      ```python
     |      # Create the artifact
     |      model.export("path/to/location")
     |      
     |      # Later, in a different process / environment...
     |      reloaded_artifact = tf.saved_model.load("path/to/location")
     |      predictions = reloaded_artifact.serve(input_data)
     |      ```
     |      
     |      If you would like to customize your serving endpoints, you can
     |      use the lower-level `keras.export.ExportArchive` class. The `export()`
     |      method relies on `ExportArchive` internally.
     |  
     |  fit_generator(self, generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, validation_freq=1, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
     |      Fits the model on data yielded batch-by-batch by a Python generator.
     |      
     |      DEPRECATED:
     |        `Model.fit` now supports generators, so there is no longer any need to
     |        use this endpoint.
     |  
     |  get_compile_config(self)
     |  
     |  get_config(self)
     |      Returns the config of the `Model`.
     |      
     |      Config is a Python dictionary (serializable) containing the
     |      configuration of an object, which in this case is a `Model`. This allows
     |      the `Model` to be be reinstantiated later (without its trained weights)
     |      from this configuration.
     |      
     |      Note that `get_config()` does not guarantee to return a fresh copy of
     |      dict every time it is called. The callers should make a copy of the
     |      returned dict if they want to modify it.
     |      
     |      Developers of subclassed `Model` are advised to override this method,
     |      and continue to update the dict from `super(MyModel, self).get_config()`
     |      to provide the proper configuration of this `Model`. The default config
     |      will return config dict for init parameters if they are basic types.
     |      Raises `NotImplementedError` when in cases where a custom
     |      `get_config()` implementation is required for the subclassed model.
     |      
     |      Returns:
     |          Python dictionary containing the configuration of this `Model`.
     |  
     |  get_layer(self, name=None, index=None)
     |      Retrieves a layer based on either its name (unique) or index.
     |      
     |      If `name` and `index` are both provided, `index` will take precedence.
     |      Indices are based on order of horizontal graph traversal (bottom-up).
     |      
     |      Args:
     |          name: String, name of layer.
     |          index: Integer, index of layer.
     |      
     |      Returns:
     |          A layer instance.
     |  
     |  get_metrics_result(self)
     |      Returns the model's metrics values as a dict.
     |      
     |      If any of the metric result is a dict (containing multiple metrics),
     |      each of them gets added to the top level returned dict of this method.
     |      
     |      Returns:
     |        A `dict` containing values of the metrics listed in `self.metrics`.
     |        Example:
     |        `{'loss': 0.2, 'accuracy': 0.7}`.
     |  
     |  get_weight_paths(self)
     |      Retrieve all the variables and their paths for the model.
     |      
     |      The variable path (string) is a stable key to identify a `tf.Variable`
     |      instance owned by the model. It can be used to specify variable-specific
     |      configurations (e.g. DTensor, quantization) from a global view.
     |      
     |      This method returns a dict with weight object paths as keys
     |      and the corresponding `tf.Variable` instances as values.
     |      
     |      Note that if the model is a subclassed model and the weights haven't
     |      been initialized, an empty dict will be returned.
     |      
     |      Returns:
     |          A dict where keys are variable paths and values are `tf.Variable`
     |           instances.
     |      
     |      Example:
     |      
     |      ```python
     |      class SubclassModel(tf.keras.Model):
     |      
     |        def __init__(self, name=None):
     |          super().__init__(name=name)
     |          self.d1 = tf.keras.layers.Dense(10)
     |          self.d2 = tf.keras.layers.Dense(20)
     |      
     |        def call(self, inputs):
     |          x = self.d1(inputs)
     |          return self.d2(x)
     |      
     |      model = SubclassModel()
     |      model(tf.zeros((10, 10)))
     |      weight_paths = model.get_weight_paths()
     |      # weight_paths:
     |      # {
     |      #    'd1.kernel': model.d1.kernel,
     |      #    'd1.bias': model.d1.bias,
     |      #    'd2.kernel': model.d2.kernel,
     |      #    'd2.bias': model.d2.bias,
     |      # }
     |      
     |      # Functional model
     |      inputs = tf.keras.Input((10,), batch_size=10)
     |      x = tf.keras.layers.Dense(20, name='d1')(inputs)
     |      output = tf.keras.layers.Dense(30, name='d2')(x)
     |      model = tf.keras.Model(inputs, output)
     |      d1 = model.layers[1]
     |      d2 = model.layers[2]
     |      weight_paths = model.get_weight_paths()
     |      # weight_paths:
     |      # {
     |      #    'd1.kernel': d1.kernel,
     |      #    'd1.bias': d1.bias,
     |      #    'd2.kernel': d2.kernel,
     |      #    'd2.bias': d2.bias,
     |      # }
     |      ```
     |  
     |  get_weights(self)
     |      Retrieves the weights of the model.
     |      
     |      Returns:
     |          A flat list of Numpy arrays.
     |  
     |  make_train_function(self, force=False)
     |      Creates a function that executes one step of training.
     |      
     |      This method can be overridden to support custom training logic.
     |      This method is called by `Model.fit` and `Model.train_on_batch`.
     |      
     |      Typically, this method directly controls `tf.function` and
     |      `tf.distribute.Strategy` settings, and delegates the actual training
     |      logic to `Model.train_step`.
     |      
     |      This function is cached the first time `Model.fit` or
     |      `Model.train_on_batch` is called. The cache is cleared whenever
     |      `Model.compile` is called. You can skip the cache and generate again the
     |      function with `force=True`.
     |      
     |      Args:
     |        force: Whether to regenerate the train function and skip the cached
     |          function if available.
     |      
     |      Returns:
     |        Function. The function created by this method should accept a
     |        `tf.data.Iterator`, and return a `dict` containing values that will
     |        be passed to `tf.keras.Callbacks.on_train_batch_end`, such as
     |        `{'loss': 0.2, 'accuracy': 0.7}`.
     |  
     |  predict(self, x, batch_size=None, verbose='auto', steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)
     |      Generates output predictions for the input samples.
     |      
     |      Computation is done in batches. This method is designed for batch
     |      processing of large numbers of inputs. It is not intended for use inside
     |      of loops that iterate over your data and process small numbers of inputs
     |      at a time.
     |      
     |      For small numbers of inputs that fit in one batch,
     |      directly use `__call__()` for faster execution, e.g.,
     |      `model(x)`, or `model(x, training=False)` if you have layers such as
     |      `tf.keras.layers.BatchNormalization` that behave differently during
     |      inference. You may pair the individual model call with a `tf.function`
     |      for additional performance inside your inner loop.
     |      If you need access to numpy array values instead of tensors after your
     |      model call, you can use `tensor.numpy()` to get the numpy array value of
     |      an eager tensor.
     |      
     |      Also, note the fact that test loss is not affected by
     |      regularization layers like noise and dropout.
     |      
     |      Note: See [this FAQ entry](
     |      https://keras.io/getting_started/faq/#whats-the-difference-between-model-methods-predict-and-call)
     |      for more details about the difference between `Model` methods
     |      `predict()` and `__call__()`.
     |      
     |      Args:
     |          x: Input samples. It could be:
     |            - A Numpy array (or array-like), or a list of arrays
     |              (in case the model has multiple inputs).
     |            - A TensorFlow tensor, or a list of tensors
     |              (in case the model has multiple inputs).
     |            - A `tf.data` dataset.
     |            - A generator or `keras.utils.Sequence` instance.
     |            A more detailed description of unpacking behavior for iterator
     |            types (Dataset, generator, Sequence) is given in the `Unpacking
     |            behavior for iterator-like inputs` section of `Model.fit`.
     |          batch_size: Integer or `None`.
     |              Number of samples per batch.
     |              If unspecified, `batch_size` will default to 32.
     |              Do not specify the `batch_size` if your data is in the
     |              form of dataset, generators, or `keras.utils.Sequence` instances
     |              (since they generate batches).
     |          verbose: `"auto"`, 0, 1, or 2. Verbosity mode.
     |              0 = silent, 1 = progress bar, 2 = single line.
     |              `"auto"` defaults to 1 for most cases, and to 2 when used with
     |              `ParameterServerStrategy`. Note that the progress bar is not
     |              particularly useful when logged to a file, so `verbose=2` is
     |              recommended when not running interactively (e.g. in a production
     |              environment).
     |          steps: Total number of steps (batches of samples)
     |              before declaring the prediction round finished.
     |              Ignored with the default value of `None`. If x is a `tf.data`
     |              dataset and `steps` is None, `predict()` will
     |              run until the input dataset is exhausted.
     |          callbacks: List of `keras.callbacks.Callback` instances.
     |              List of callbacks to apply during prediction.
     |              See [callbacks](
     |              https://www.tensorflow.org/api_docs/python/tf/keras/callbacks).
     |          max_queue_size: Integer. Used for generator or
     |              `keras.utils.Sequence` input only. Maximum size for the
     |              generator queue. If unspecified, `max_queue_size` will default
     |              to 10.
     |          workers: Integer. Used for generator or `keras.utils.Sequence` input
     |              only. Maximum number of processes to spin up when using
     |              process-based threading. If unspecified, `workers` will default
     |              to 1.
     |          use_multiprocessing: Boolean. Used for generator or
     |              `keras.utils.Sequence` input only. If `True`, use process-based
     |              threading. If unspecified, `use_multiprocessing` will default to
     |              `False`. Note that because this implementation relies on
     |              multiprocessing, you should not pass non-picklable arguments to
     |              the generator as they can't be passed easily to children
     |              processes.
     |      
     |      See the discussion of `Unpacking behavior for iterator-like inputs` for
     |      `Model.fit`. Note that Model.predict uses the same interpretation rules
     |      as `Model.fit` and `Model.evaluate`, so inputs must be unambiguous for
     |      all three methods.
     |      
     |      Returns:
     |          Numpy array(s) of predictions.
     |      
     |      Raises:
     |          RuntimeError: If `model.predict` is wrapped in a `tf.function`.
     |          ValueError: In case of mismatch between the provided
     |              input data and the model's expectations,
     |              or in case a stateful model receives a number of samples
     |              that is not a multiple of the batch size.
     |  
     |  predict_generator(self, generator, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
     |      Generates predictions for the input samples from a data generator.
     |      
     |      DEPRECATED:
     |        `Model.predict` now supports generators, so there is no longer any
     |        need to use this endpoint.
     |  
     |  predict_on_batch(self, x)
     |      Returns predictions for a single batch of samples.
     |      
     |      Args:
     |          x: Input data. It could be:
     |            - A Numpy array (or array-like), or a list of arrays (in case the
     |                model has multiple inputs).
     |            - A TensorFlow tensor, or a list of tensors (in case the model has
     |                multiple inputs).
     |      
     |      Returns:
     |          Numpy array(s) of predictions.
     |      
     |      Raises:
     |          RuntimeError: If `model.predict_on_batch` is wrapped in a
     |            `tf.function`.
     |  
     |  predict_step(self, data)
     |      The logic for one inference step.
     |      
     |      This method can be overridden to support custom inference logic.
     |      This method is called by `Model.make_predict_function`.
     |      
     |      This method should contain the mathematical logic for one step of
     |      inference.  This typically includes the forward pass.
     |      
     |      Configuration details for *how* this logic is run (e.g. `tf.function`
     |      and `tf.distribute.Strategy` settings), should be left to
     |      `Model.make_predict_function`, which can also be overridden.
     |      
     |      Args:
     |        data: A nested structure of `Tensor`s.
     |      
     |      Returns:
     |        The result of one inference step, typically the output of calling the
     |        `Model` on data.
     |  
     |  reset_metrics(self)
     |      Resets the state of all the metrics in the model.
     |      
     |      Examples:
     |      
     |      >>> inputs = tf.keras.layers.Input(shape=(3,))
     |      >>> outputs = tf.keras.layers.Dense(2)(inputs)
     |      >>> model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
     |      >>> model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
     |      
     |      >>> x = np.random.random((2, 3))
     |      >>> y = np.random.randint(0, 2, (2, 2))
     |      >>> _ = model.fit(x, y, verbose=0)
     |      >>> assert all(float(m.result()) for m in model.metrics)
     |      
     |      >>> model.reset_metrics()
     |      >>> assert all(float(m.result()) == 0 for m in model.metrics)
     |  
     |  reset_states(self)
     |  
     |  save_spec(self, dynamic_batch=True)
     |      Returns the `tf.TensorSpec` of call args as a tuple `(args, kwargs)`.
     |      
     |      This value is automatically defined after calling the model for the
     |      first time. Afterwards, you can use it when exporting the model for
     |      serving:
     |      
     |      ```python
     |      model = tf.keras.Model(...)
     |      
     |      @tf.function
     |      def serve(*args, **kwargs):
     |        outputs = model(*args, **kwargs)
     |        # Apply postprocessing steps, or add additional outputs.
     |        ...
     |        return outputs
     |      
     |      # arg_specs is `[tf.TensorSpec(...), ...]`. kwarg_specs, in this
     |      # example, is an empty dict since functional models do not use keyword
     |      # arguments.
     |      arg_specs, kwarg_specs = model.save_spec()
     |      
     |      model.save(path, signatures={
     |        'serving_default': serve.get_concrete_function(*arg_specs,
     |                                                       **kwarg_specs)
     |      })
     |      ```
     |      
     |      Args:
     |        dynamic_batch: Whether to set the batch sizes of all the returned
     |          `tf.TensorSpec` to `None`. (Note that when defining functional or
     |          Sequential models with `tf.keras.Input([...], batch_size=X)`, the
     |          batch size will always be preserved). Defaults to `True`.
     |      Returns:
     |        If the model inputs are defined, returns a tuple `(args, kwargs)`. All
     |        elements in `args` and `kwargs` are `tf.TensorSpec`.
     |        If the model inputs are not defined, returns `None`.
     |        The model inputs are automatically set when calling the model,
     |        `model.fit`, `model.evaluate` or `model.predict`.
     |  
     |  save_weights(self, filepath, overwrite=True, save_format=None, options=None)
     |      Saves all layer weights.
     |      
     |      Either saves in HDF5 or in TensorFlow format based on the `save_format`
     |      argument.
     |      
     |      When saving in HDF5 format, the weight file has:
     |        - `layer_names` (attribute), a list of strings
     |            (ordered names of model layers).
     |        - For every layer, a `group` named `layer.name`
     |            - For every such layer group, a group attribute `weight_names`,
     |                a list of strings
     |                (ordered names of weights tensor of the layer).
     |            - For every weight in the layer, a dataset
     |                storing the weight value, named after the weight tensor.
     |      
     |      When saving in TensorFlow format, all objects referenced by the network
     |      are saved in the same format as `tf.train.Checkpoint`, including any
     |      `Layer` instances or `Optimizer` instances assigned to object
     |      attributes. For networks constructed from inputs and outputs using
     |      `tf.keras.Model(inputs, outputs)`, `Layer` instances used by the network
     |      are tracked/saved automatically. For user-defined classes which inherit
     |      from `tf.keras.Model`, `Layer` instances must be assigned to object
     |      attributes, typically in the constructor. See the documentation of
     |      `tf.train.Checkpoint` and `tf.keras.Model` for details.
     |      
     |      While the formats are the same, do not mix `save_weights` and
     |      `tf.train.Checkpoint`. Checkpoints saved by `Model.save_weights` should
     |      be loaded using `Model.load_weights`. Checkpoints saved using
     |      `tf.train.Checkpoint.save` should be restored using the corresponding
     |      `tf.train.Checkpoint.restore`. Prefer `tf.train.Checkpoint` over
     |      `save_weights` for training checkpoints.
     |      
     |      The TensorFlow format matches objects and variables by starting at a
     |      root object, `self` for `save_weights`, and greedily matching attribute
     |      names. For `Model.save` this is the `Model`, and for `Checkpoint.save`
     |      this is the `Checkpoint` even if the `Checkpoint` has a model attached.
     |      This means saving a `tf.keras.Model` using `save_weights` and loading
     |      into a `tf.train.Checkpoint` with a `Model` attached (or vice versa)
     |      will not match the `Model`'s variables. See the
     |      [guide to training checkpoints](
     |      https://www.tensorflow.org/guide/checkpoint) for details on
     |      the TensorFlow format.
     |      
     |      Args:
     |          filepath: String or PathLike, path to the file to save the weights
     |              to. When saving in TensorFlow format, this is the prefix used
     |              for checkpoint files (multiple files are generated). Note that
     |              the '.h5' suffix causes weights to be saved in HDF5 format.
     |          overwrite: Whether to silently overwrite any existing file at the
     |              target location, or provide the user with a manual prompt.
     |          save_format: Either 'tf' or 'h5'. A `filepath` ending in '.h5' or
     |              '.keras' will default to HDF5 if `save_format` is `None`.
     |              Otherwise `None` defaults to 'tf'.
     |          options: Optional `tf.train.CheckpointOptions` object that specifies
     |              options for saving weights.
     |      
     |      Raises:
     |          ImportError: If `h5py` is not available when attempting to save in
     |              HDF5 format.
     |  
     |  test_on_batch(self, x, y=None, sample_weight=None, reset_metrics=True, return_dict=False)
     |      Test the model on a single batch of samples.
     |      
     |      Args:
     |          x: Input data. It could be:
     |            - A Numpy array (or array-like), or a list of arrays (in case the
     |                model has multiple inputs).
     |            - A TensorFlow tensor, or a list of tensors (in case the model has
     |                multiple inputs).
     |            - A dict mapping input names to the corresponding array/tensors,
     |                if the model has named inputs.
     |          y: Target data. Like the input data `x`, it could be either Numpy
     |            array(s) or TensorFlow tensor(s). It should be consistent with `x`
     |            (you cannot have Numpy inputs and tensor targets, or inversely).
     |          sample_weight: Optional array of the same length as x, containing
     |            weights to apply to the model's loss for each sample. In the case
     |            of temporal data, you can pass a 2D array with shape (samples,
     |            sequence_length), to apply a different weight to every timestep of
     |            every sample.
     |          reset_metrics: If `True`, the metrics returned will be only for this
     |            batch. If `False`, the metrics will be statefully accumulated
     |            across batches.
     |          return_dict: If `True`, loss and metric results are returned as a
     |            dict, with each key being the name of the metric. If `False`, they
     |            are returned as a list.
     |      
     |      Returns:
     |          Scalar test loss (if the model has a single output and no metrics)
     |          or list of scalars (if the model has multiple outputs
     |          and/or metrics). The attribute `model.metrics_names` will give you
     |          the display labels for the scalar outputs.
     |      
     |      Raises:
     |          RuntimeError: If `model.test_on_batch` is wrapped in a
     |            `tf.function`.
     |  
     |  test_step(self, data)
     |      The logic for one evaluation step.
     |      
     |      This method can be overridden to support custom evaluation logic.
     |      This method is called by `Model.make_test_function`.
     |      
     |      This function should contain the mathematical logic for one step of
     |      evaluation.
     |      This typically includes the forward pass, loss calculation, and metrics
     |      updates.
     |      
     |      Configuration details for *how* this logic is run (e.g. `tf.function`
     |      and `tf.distribute.Strategy` settings), should be left to
     |      `Model.make_test_function`, which can also be overridden.
     |      
     |      Args:
     |        data: A nested structure of `Tensor`s.
     |      
     |      Returns:
     |        A `dict` containing values that will be passed to
     |        `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
     |        values of the `Model`'s metrics are returned.
     |  
     |  to_json(self, **kwargs)
     |      Returns a JSON string containing the network configuration.
     |      
     |      To load a network from a JSON save file, use
     |      `keras.models.model_from_json(json_string, custom_objects={})`.
     |      
     |      Args:
     |          **kwargs: Additional keyword arguments to be passed to
     |              *`json.dumps()`.
     |      
     |      Returns:
     |          A JSON string.
     |  
     |  to_yaml(self, **kwargs)
     |      Returns a yaml string containing the network configuration.
     |      
     |      Note: Since TF 2.6, this method is no longer supported and will raise a
     |      RuntimeError.
     |      
     |      To load a network from a yaml save file, use
     |      `keras.models.model_from_yaml(yaml_string, custom_objects={})`.
     |      
     |      `custom_objects` should be a dictionary mapping
     |      the names of custom losses / layers / etc to the corresponding
     |      functions / classes.
     |      
     |      Args:
     |          **kwargs: Additional keyword arguments
     |              to be passed to `yaml.dump()`.
     |      
     |      Returns:
     |          A YAML string.
     |      
     |      Raises:
     |          RuntimeError: announces that the method poses a security risk
     |  
     |  ----------------------------------------------------------------------
     |  Class methods inherited from keras.engine.training.Model:
     |  
     |  from_config(config, custom_objects=None) from builtins.type
     |      Creates a layer from its config.
     |      
     |      This method is the reverse of `get_config`,
     |      capable of instantiating the same layer from the config
     |      dictionary. It does not handle layer connectivity
     |      (handled by Network), nor weights (handled by `set_weights`).
     |      
     |      Args:
     |          config: A Python dictionary, typically the
     |              output of get_config.
     |      
     |      Returns:
     |          A layer instance.
     |  
     |  ----------------------------------------------------------------------
     |  Static methods inherited from keras.engine.training.Model:
     |  
     |  __new__(cls, *args, **kwargs)
     |      Create and return a new object.  See help(type) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Readonly properties inherited from keras.engine.training.Model:
     |  
     |  distribute_strategy
     |      The `tf.distribute.Strategy` this model was created under.
     |  
     |  metrics
     |      Return metrics added using `compile()` or `add_metric()`.
     |      
     |      Note: Metrics passed to `compile()` are available only after a
     |      `keras.Model` has been trained/evaluated on actual data.
     |      
     |      Examples:
     |      
     |      >>> inputs = tf.keras.layers.Input(shape=(3,))
     |      >>> outputs = tf.keras.layers.Dense(2)(inputs)
     |      >>> model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
     |      >>> model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
     |      >>> [m.name for m in model.metrics]
     |      []
     |      
     |      >>> x = np.random.random((2, 3))
     |      >>> y = np.random.randint(0, 2, (2, 2))
     |      >>> model.fit(x, y)
     |      >>> [m.name for m in model.metrics]
     |      ['loss', 'mae']
     |      
     |      >>> inputs = tf.keras.layers.Input(shape=(3,))
     |      >>> d = tf.keras.layers.Dense(2, name='out')
     |      >>> output_1 = d(inputs)
     |      >>> output_2 = d(inputs)
     |      >>> model = tf.keras.models.Model(
     |      ...    inputs=inputs, outputs=[output_1, output_2])
     |      >>> model.add_metric(
     |      ...    tf.reduce_sum(output_2), name='mean', aggregation='mean')
     |      >>> model.compile(optimizer="Adam", loss="mse", metrics=["mae", "acc"])
     |      >>> model.fit(x, (y, y))
     |      >>> [m.name for m in model.metrics]
     |      ['loss', 'out_loss', 'out_1_loss', 'out_mae', 'out_acc', 'out_1_mae',
     |      'out_1_acc', 'mean']
     |  
     |  metrics_names
     |      Returns the model's display labels for all outputs.
     |      
     |      Note: `metrics_names` are available only after a `keras.Model` has been
     |      trained/evaluated on actual data.
     |      
     |      Examples:
     |      
     |      >>> inputs = tf.keras.layers.Input(shape=(3,))
     |      >>> outputs = tf.keras.layers.Dense(2)(inputs)
     |      >>> model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
     |      >>> model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
     |      >>> model.metrics_names
     |      []
     |      
     |      >>> x = np.random.random((2, 3))
     |      >>> y = np.random.randint(0, 2, (2, 2))
     |      >>> model.fit(x, y)
     |      >>> model.metrics_names
     |      ['loss', 'mae']
     |      
     |      >>> inputs = tf.keras.layers.Input(shape=(3,))
     |      >>> d = tf.keras.layers.Dense(2, name='out')
     |      >>> output_1 = d(inputs)
     |      >>> output_2 = d(inputs)
     |      >>> model = tf.keras.models.Model(
     |      ...    inputs=inputs, outputs=[output_1, output_2])
     |      >>> model.compile(optimizer="Adam", loss="mse", metrics=["mae", "acc"])
     |      >>> model.fit(x, (y, y))
     |      >>> model.metrics_names
     |      ['loss', 'out_loss', 'out_1_loss', 'out_mae', 'out_acc', 'out_1_mae',
     |      'out_1_acc']
     |  
     |  non_trainable_weights
     |      List of all non-trainable weights tracked by this layer.
     |      
     |      Non-trainable weights are *not* updated during training. They are
     |      expected to be updated manually in `call()`.
     |      
     |      Returns:
     |        A list of non-trainable variables.
     |  
     |  state_updates
     |      Deprecated, do NOT use!
     |      
     |      Returns the `updates` from all layers that are stateful.
     |      
     |      This is useful for separating training updates and
     |      state updates, e.g. when we need to update a layer's internal state
     |      during prediction.
     |      
     |      Returns:
     |          A list of update ops.
     |  
     |  trainable_weights
     |      List of all trainable weights tracked by this layer.
     |      
     |      Trainable weights are updated via gradient descent during training.
     |      
     |      Returns:
     |        A list of trainable variables.
     |  
     |  weights
     |      Returns the list of all layer variables/weights.
     |      
     |      Note: This will not track the weights of nested `tf.Modules` that are
     |      not themselves Keras layers.
     |      
     |      Returns:
     |        A list of variables.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from keras.engine.training.Model:
     |  
     |  distribute_reduction_method
     |      The method employed to reduce per-replica values during training.
     |      
     |      Unless specified, the value "auto" will be assumed, indicating that
     |      the reduction strategy should be chosen based on the current
     |      running environment.
     |      See `reduce_per_replica` function for more details.
     |  
     |  jit_compile
     |      Specify whether to compile the model with XLA.
     |      
     |      [XLA](https://www.tensorflow.org/xla) is an optimizing compiler
     |      for machine learning. `jit_compile` is not enabled by default.
     |      Note that `jit_compile=True` may not necessarily work for all models.
     |      
     |      For more information on supported operations please refer to the
     |      [XLA documentation](https://www.tensorflow.org/xla). Also refer to
     |      [known XLA issues](https://www.tensorflow.org/xla/known_issues)
     |      for more details.
     |  
     |  layers
     |  
     |  run_eagerly
     |      Settable attribute indicating whether the model should run eagerly.
     |      
     |      Running eagerly means that your model will be run step by step,
     |      like Python code. Your model might run slower, but it should become
     |      easier for you to debug it by stepping into individual layer calls.
     |      
     |      By default, we will attempt to compile your model to a static graph to
     |      deliver the best execution performance.
     |      
     |      Returns:
     |        Boolean, whether the model should run eagerly.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from keras.engine.base_layer.Layer:
     |  
     |  __delattr__(self, name)
     |      Implement delattr(self, name).
     |  
     |  __getstate__(self)
     |  
     |  __setstate__(self, state)
     |  
     |  add_loss(self, losses, **kwargs)
     |      Add loss tensor(s), potentially dependent on layer inputs.
     |      
     |      Some losses (for instance, activity regularization losses) may be
     |      dependent on the inputs passed when calling a layer. Hence, when reusing
     |      the same layer on different inputs `a` and `b`, some entries in
     |      `layer.losses` may be dependent on `a` and some on `b`. This method
     |      automatically keeps track of dependencies.
     |      
     |      This method can be used inside a subclassed layer or model's `call`
     |      function, in which case `losses` should be a Tensor or list of Tensors.
     |      
     |      Example:
     |      
     |      ```python
     |      class MyLayer(tf.keras.layers.Layer):
     |        def call(self, inputs):
     |          self.add_loss(tf.abs(tf.reduce_mean(inputs)))
     |          return inputs
     |      ```
     |      
     |      The same code works in distributed training: the input to `add_loss()`
     |      is treated like a regularization loss and averaged across replicas
     |      by the training loop (both built-in `Model.fit()` and compliant custom
     |      training loops).
     |      
     |      The `add_loss` method can also be called directly on a Functional Model
     |      during construction. In this case, any loss Tensors passed to this Model
     |      must be symbolic and be able to be traced back to the model's `Input`s.
     |      These losses become part of the model's topology and are tracked in
     |      `get_config`.
     |      
     |      Example:
     |      
     |      ```python
     |      inputs = tf.keras.Input(shape=(10,))
     |      x = tf.keras.layers.Dense(10)(inputs)
     |      outputs = tf.keras.layers.Dense(1)(x)
     |      model = tf.keras.Model(inputs, outputs)
     |      # Activity regularization.
     |      model.add_loss(tf.abs(tf.reduce_mean(x)))
     |      ```
     |      
     |      If this is not the case for your loss (if, for example, your loss
     |      references a `Variable` of one of the model's layers), you can wrap your
     |      loss in a zero-argument lambda. These losses are not tracked as part of
     |      the model's topology since they can't be serialized.
     |      
     |      Example:
     |      
     |      ```python
     |      inputs = tf.keras.Input(shape=(10,))
     |      d = tf.keras.layers.Dense(10)
     |      x = d(inputs)
     |      outputs = tf.keras.layers.Dense(1)(x)
     |      model = tf.keras.Model(inputs, outputs)
     |      # Weight regularization.
     |      model.add_loss(lambda: tf.reduce_mean(d.kernel))
     |      ```
     |      
     |      Args:
     |        losses: Loss tensor, or list/tuple of tensors. Rather than tensors,
     |          losses may also be zero-argument callables which create a loss
     |          tensor.
     |        **kwargs: Used for backwards compatibility only.
     |  
     |  add_metric(self, value, name=None, **kwargs)
     |      Adds metric tensor to the layer.
     |      
     |      This method can be used inside the `call()` method of a subclassed layer
     |      or model.
     |      
     |      ```python
     |      class MyMetricLayer(tf.keras.layers.Layer):
     |        def __init__(self):
     |          super(MyMetricLayer, self).__init__(name='my_metric_layer')
     |          self.mean = tf.keras.metrics.Mean(name='metric_1')
     |      
     |        def call(self, inputs):
     |          self.add_metric(self.mean(inputs))
     |          self.add_metric(tf.reduce_sum(inputs), name='metric_2')
     |          return inputs
     |      ```
     |      
     |      This method can also be called directly on a Functional Model during
     |      construction. In this case, any tensor passed to this Model must
     |      be symbolic and be able to be traced back to the model's `Input`s. These
     |      metrics become part of the model's topology and are tracked when you
     |      save the model via `save()`.
     |      
     |      ```python
     |      inputs = tf.keras.Input(shape=(10,))
     |      x = tf.keras.layers.Dense(10)(inputs)
     |      outputs = tf.keras.layers.Dense(1)(x)
     |      model = tf.keras.Model(inputs, outputs)
     |      model.add_metric(math_ops.reduce_sum(x), name='metric_1')
     |      ```
     |      
     |      Note: Calling `add_metric()` with the result of a metric object on a
     |      Functional Model, as shown in the example below, is not supported. This
     |      is because we cannot trace the metric result tensor back to the model's
     |      inputs.
     |      
     |      ```python
     |      inputs = tf.keras.Input(shape=(10,))
     |      x = tf.keras.layers.Dense(10)(inputs)
     |      outputs = tf.keras.layers.Dense(1)(x)
     |      model = tf.keras.Model(inputs, outputs)
     |      model.add_metric(tf.keras.metrics.Mean()(x), name='metric_1')
     |      ```
     |      
     |      Args:
     |        value: Metric tensor.
     |        name: String metric name.
     |        **kwargs: Additional keyword arguments for backward compatibility.
     |          Accepted values:
     |          `aggregation` - When the `value` tensor provided is not the result
     |          of calling a `keras.Metric` instance, it will be aggregated by
     |          default using a `keras.Metric.Mean`.
     |  
     |  add_update(self, updates)
     |      Add update op(s), potentially dependent on layer inputs.
     |      
     |      Weight updates (for instance, the updates of the moving mean and
     |      variance in a BatchNormalization layer) may be dependent on the inputs
     |      passed when calling a layer. Hence, when reusing the same layer on
     |      different inputs `a` and `b`, some entries in `layer.updates` may be
     |      dependent on `a` and some on `b`. This method automatically keeps track
     |      of dependencies.
     |      
     |      This call is ignored when eager execution is enabled (in that case,
     |      variable updates are run on the fly and thus do not need to be tracked
     |      for later execution).
     |      
     |      Args:
     |        updates: Update op, or list/tuple of update ops, or zero-arg callable
     |          that returns an update op. A zero-arg callable should be passed in
     |          order to disable running the updates by setting `trainable=False`
     |          on this Layer, when executing in Eager mode.
     |  
     |  add_variable(self, *args, **kwargs)
     |      Deprecated, do NOT use! Alias for `add_weight`.
     |  
     |  add_weight(self, name=None, shape=None, dtype=None, initializer=None, regularizer=None, trainable=None, constraint=None, use_resource=None, synchronization=<VariableSynchronization.AUTO: 0>, aggregation=<VariableAggregationV2.NONE: 0>, **kwargs)
     |      Adds a new variable to the layer.
     |      
     |      Args:
     |        name: Variable name.
     |        shape: Variable shape. Defaults to scalar if unspecified.
     |        dtype: The type of the variable. Defaults to `self.dtype`.
     |        initializer: Initializer instance (callable).
     |        regularizer: Regularizer instance (callable).
     |        trainable: Boolean, whether the variable should be part of the layer's
     |          "trainable_variables" (e.g. variables, biases)
     |          or "non_trainable_variables" (e.g. BatchNorm mean and variance).
     |          Note that `trainable` cannot be `True` if `synchronization`
     |          is set to `ON_READ`.
     |        constraint: Constraint instance (callable).
     |        use_resource: Whether to use a `ResourceVariable` or not.
     |          See [this guide](
     |          https://www.tensorflow.org/guide/migrate/tf1_vs_tf2#resourcevariables_instead_of_referencevariables)
     |           for more information.
     |        synchronization: Indicates when a distributed a variable will be
     |          aggregated. Accepted values are constants defined in the class
     |          `tf.VariableSynchronization`. By default the synchronization is set
     |          to `AUTO` and the current `DistributionStrategy` chooses when to
     |          synchronize. If `synchronization` is set to `ON_READ`, `trainable`
     |          must not be set to `True`.
     |        aggregation: Indicates how a distributed variable will be aggregated.
     |          Accepted values are constants defined in the class
     |          `tf.VariableAggregation`.
     |        **kwargs: Additional keyword arguments. Accepted values are `getter`,
     |          `collections`, `experimental_autocast` and `caching_device`.
     |      
     |      Returns:
     |        The variable created.
     |      
     |      Raises:
     |        ValueError: When giving unsupported dtype and no initializer or when
     |          trainable has been set to True with synchronization set as
     |          `ON_READ`.
     |  
     |  build_from_config(self, config)
     |  
     |  compute_mask(self, inputs, mask=None)
     |      Computes an output mask tensor.
     |      
     |      Args:
     |          inputs: Tensor or list of tensors.
     |          mask: Tensor or list of tensors.
     |      
     |      Returns:
     |          None or a tensor (or list of tensors,
     |              one per output tensor of the layer).
     |  
     |  compute_output_shape(self, input_shape)
     |      Computes the output shape of the layer.
     |      
     |      This method will cause the layer's state to be built, if that has not
     |      happened before. This requires that the layer will later be used with
     |      inputs that match the input shape provided here.
     |      
     |      Args:
     |          input_shape: Shape tuple (tuple of integers) or `tf.TensorShape`,
     |              or structure of shape tuples / `tf.TensorShape` instances
     |              (one per output tensor of the layer).
     |              Shape tuples can include None for free dimensions,
     |              instead of an integer.
     |      
     |      Returns:
     |          A `tf.TensorShape` instance
     |          or structure of `tf.TensorShape` instances.
     |  
     |  compute_output_signature(self, input_signature)
     |      Compute the output tensor signature of the layer based on the inputs.
     |      
     |      Unlike a TensorShape object, a TensorSpec object contains both shape
     |      and dtype information for a tensor. This method allows layers to provide
     |      output dtype information if it is different from the input dtype.
     |      For any layer that doesn't implement this function,
     |      the framework will fall back to use `compute_output_shape`, and will
     |      assume that the output dtype matches the input dtype.
     |      
     |      Args:
     |        input_signature: Single TensorSpec or nested structure of TensorSpec
     |          objects, describing a candidate input for the layer.
     |      
     |      Returns:
     |        Single TensorSpec or nested structure of TensorSpec objects,
     |          describing how the layer would transform the provided input.
     |      
     |      Raises:
     |        TypeError: If input_signature contains a non-TensorSpec object.
     |  
     |  count_params(self)
     |      Count the total number of scalars composing the weights.
     |      
     |      Returns:
     |          An integer count.
     |      
     |      Raises:
     |          ValueError: if the layer isn't yet built
     |            (in which case its weights aren't yet defined).
     |  
     |  finalize_state(self)
     |      Finalizes the layers state after updating layer weights.
     |      
     |      This function can be subclassed in a layer and will be called after
     |      updating a layer weights. It can be overridden to finalize any
     |      additional layer state after a weight update.
     |      
     |      This function will be called after weights of a layer have been restored
     |      from a loaded model.
     |  
     |  get_build_config(self)
     |  
     |  get_input_at(self, node_index)
     |      Retrieves the input tensor(s) of a layer at a given node.
     |      
     |      Args:
     |          node_index: Integer, index of the node
     |              from which to retrieve the attribute.
     |              E.g. `node_index=0` will correspond to the
     |              first input node of the layer.
     |      
     |      Returns:
     |          A tensor (or list of tensors if the layer has multiple inputs).
     |      
     |      Raises:
     |        RuntimeError: If called in Eager mode.
     |  
     |  get_input_mask_at(self, node_index)
     |      Retrieves the input mask tensor(s) of a layer at a given node.
     |      
     |      Args:
     |          node_index: Integer, index of the node
     |              from which to retrieve the attribute.
     |              E.g. `node_index=0` will correspond to the
     |              first time the layer was called.
     |      
     |      Returns:
     |          A mask tensor
     |          (or list of tensors if the layer has multiple inputs).
     |  
     |  get_input_shape_at(self, node_index)
     |      Retrieves the input shape(s) of a layer at a given node.
     |      
     |      Args:
     |          node_index: Integer, index of the node
     |              from which to retrieve the attribute.
     |              E.g. `node_index=0` will correspond to the
     |              first time the layer was called.
     |      
     |      Returns:
     |          A shape tuple
     |          (or list of shape tuples if the layer has multiple inputs).
     |      
     |      Raises:
     |        RuntimeError: If called in Eager mode.
     |  
     |  get_output_at(self, node_index)
     |      Retrieves the output tensor(s) of a layer at a given node.
     |      
     |      Args:
     |          node_index: Integer, index of the node
     |              from which to retrieve the attribute.
     |              E.g. `node_index=0` will correspond to the
     |              first output node of the layer.
     |      
     |      Returns:
     |          A tensor (or list of tensors if the layer has multiple outputs).
     |      
     |      Raises:
     |        RuntimeError: If called in Eager mode.
     |  
     |  get_output_mask_at(self, node_index)
     |      Retrieves the output mask tensor(s) of a layer at a given node.
     |      
     |      Args:
     |          node_index: Integer, index of the node
     |              from which to retrieve the attribute.
     |              E.g. `node_index=0` will correspond to the
     |              first time the layer was called.
     |      
     |      Returns:
     |          A mask tensor
     |          (or list of tensors if the layer has multiple outputs).
     |  
     |  get_output_shape_at(self, node_index)
     |      Retrieves the output shape(s) of a layer at a given node.
     |      
     |      Args:
     |          node_index: Integer, index of the node
     |              from which to retrieve the attribute.
     |              E.g. `node_index=0` will correspond to the
     |              first time the layer was called.
     |      
     |      Returns:
     |          A shape tuple
     |          (or list of shape tuples if the layer has multiple outputs).
     |      
     |      Raises:
     |        RuntimeError: If called in Eager mode.
     |  
     |  set_weights(self, weights)
     |      Sets the weights of the layer, from NumPy arrays.
     |      
     |      The weights of a layer represent the state of the layer. This function
     |      sets the weight values from numpy arrays. The weight values should be
     |      passed in the order they are created by the layer. Note that the layer's
     |      weights must be instantiated before calling this function, by calling
     |      the layer.
     |      
     |      For example, a `Dense` layer returns a list of two values: the kernel
     |      matrix and the bias vector. These can be used to set the weights of
     |      another `Dense` layer:
     |      
     |      >>> layer_a = tf.keras.layers.Dense(1,
     |      ...   kernel_initializer=tf.constant_initializer(1.))
     |      >>> a_out = layer_a(tf.convert_to_tensor([[1., 2., 3.]]))
     |      >>> layer_a.get_weights()
     |      [array([[1.],
     |             [1.],
     |             [1.]], dtype=float32), array([0.], dtype=float32)]
     |      >>> layer_b = tf.keras.layers.Dense(1,
     |      ...   kernel_initializer=tf.constant_initializer(2.))
     |      >>> b_out = layer_b(tf.convert_to_tensor([[10., 20., 30.]]))
     |      >>> layer_b.get_weights()
     |      [array([[2.],
     |             [2.],
     |             [2.]], dtype=float32), array([0.], dtype=float32)]
     |      >>> layer_b.set_weights(layer_a.get_weights())
     |      >>> layer_b.get_weights()
     |      [array([[1.],
     |             [1.],
     |             [1.]], dtype=float32), array([0.], dtype=float32)]
     |      
     |      Args:
     |        weights: a list of NumPy arrays. The number
     |          of arrays and their shape must match
     |          number of the dimensions of the weights
     |          of the layer (i.e. it should match the
     |          output of `get_weights`).
     |      
     |      Raises:
     |        ValueError: If the provided weights list does not match the
     |          layer's specifications.
     |  
     |  ----------------------------------------------------------------------
     |  Readonly properties inherited from keras.engine.base_layer.Layer:
     |  
     |  compute_dtype
     |      The dtype of the layer's computations.
     |      
     |      This is equivalent to `Layer.dtype_policy.compute_dtype`. Unless
     |      mixed precision is used, this is the same as `Layer.dtype`, the dtype of
     |      the weights.
     |      
     |      Layers automatically cast their inputs to the compute dtype, which
     |      causes computations and the output to be in the compute dtype as well.
     |      This is done by the base Layer class in `Layer.__call__`, so you do not
     |      have to insert these casts if implementing your own layer.
     |      
     |      Layers often perform certain internal computations in higher precision
     |      when `compute_dtype` is float16 or bfloat16 for numeric stability. The
     |      output will still typically be float16 or bfloat16 in such cases.
     |      
     |      Returns:
     |        The layer's compute dtype.
     |  
     |  dtype
     |      The dtype of the layer weights.
     |      
     |      This is equivalent to `Layer.dtype_policy.variable_dtype`. Unless
     |      mixed precision is used, this is the same as `Layer.compute_dtype`, the
     |      dtype of the layer's computations.
     |  
     |  dtype_policy
     |      The dtype policy associated with this layer.
     |      
     |      This is an instance of a `tf.keras.mixed_precision.Policy`.
     |  
     |  dynamic
     |      Whether the layer is dynamic (eager-only); set in the constructor.
     |  
     |  inbound_nodes
     |      Return Functional API nodes upstream of this layer.
     |  
     |  input
     |      Retrieves the input tensor(s) of a layer.
     |      
     |      Only applicable if the layer has exactly one input,
     |      i.e. if it is connected to one incoming layer.
     |      
     |      Returns:
     |          Input tensor or list of input tensors.
     |      
     |      Raises:
     |        RuntimeError: If called in Eager mode.
     |        AttributeError: If no inbound nodes are found.
     |  
     |  input_mask
     |      Retrieves the input mask tensor(s) of a layer.
     |      
     |      Only applicable if the layer has exactly one inbound node,
     |      i.e. if it is connected to one incoming layer.
     |      
     |      Returns:
     |          Input mask tensor (potentially None) or list of input
     |          mask tensors.
     |      
     |      Raises:
     |          AttributeError: if the layer is connected to
     |          more than one incoming layers.
     |  
     |  input_shape
     |      Retrieves the input shape(s) of a layer.
     |      
     |      Only applicable if the layer has exactly one input,
     |      i.e. if it is connected to one incoming layer, or if all inputs
     |      have the same shape.
     |      
     |      Returns:
     |          Input shape, as an integer shape tuple
     |          (or list of shape tuples, one tuple per input tensor).
     |      
     |      Raises:
     |          AttributeError: if the layer has no defined input_shape.
     |          RuntimeError: if called in Eager mode.
     |  
     |  losses
     |      List of losses added using the `add_loss()` API.
     |      
     |      Variable regularization tensors are created when this property is
     |      accessed, so it is eager safe: accessing `losses` under a
     |      `tf.GradientTape` will propagate gradients back to the corresponding
     |      variables.
     |      
     |      Examples:
     |      
     |      >>> class MyLayer(tf.keras.layers.Layer):
     |      ...   def call(self, inputs):
     |      ...     self.add_loss(tf.abs(tf.reduce_mean(inputs)))
     |      ...     return inputs
     |      >>> l = MyLayer()
     |      >>> l(np.ones((10, 1)))
     |      >>> l.losses
     |      [1.0]
     |      
     |      >>> inputs = tf.keras.Input(shape=(10,))
     |      >>> x = tf.keras.layers.Dense(10)(inputs)
     |      >>> outputs = tf.keras.layers.Dense(1)(x)
     |      >>> model = tf.keras.Model(inputs, outputs)
     |      >>> # Activity regularization.
     |      >>> len(model.losses)
     |      0
     |      >>> model.add_loss(tf.abs(tf.reduce_mean(x)))
     |      >>> len(model.losses)
     |      1
     |      
     |      >>> inputs = tf.keras.Input(shape=(10,))
     |      >>> d = tf.keras.layers.Dense(10, kernel_initializer='ones')
     |      >>> x = d(inputs)
     |      >>> outputs = tf.keras.layers.Dense(1)(x)
     |      >>> model = tf.keras.Model(inputs, outputs)
     |      >>> # Weight regularization.
     |      >>> model.add_loss(lambda: tf.reduce_mean(d.kernel))
     |      >>> model.losses
     |      [<tf.Tensor: shape=(), dtype=float32, numpy=1.0>]
     |      
     |      Returns:
     |        A list of tensors.
     |  
     |  name
     |      Name of the layer (string), set in the constructor.
     |  
     |  non_trainable_variables
     |      Sequence of non-trainable variables owned by this module and its submodules.
     |      
     |      Note: this method uses reflection to find variables on the current instance
     |      and submodules. For performance reasons you may wish to cache the result
     |      of calling this method if you don't expect the return value to change.
     |      
     |      Returns:
     |        A sequence of variables for the current module (sorted by attribute
     |        name) followed by variables from all submodules recursively (breadth
     |        first).
     |  
     |  outbound_nodes
     |      Return Functional API nodes downstream of this layer.
     |  
     |  output
     |      Retrieves the output tensor(s) of a layer.
     |      
     |      Only applicable if the layer has exactly one output,
     |      i.e. if it is connected to one incoming layer.
     |      
     |      Returns:
     |        Output tensor or list of output tensors.
     |      
     |      Raises:
     |        AttributeError: if the layer is connected to more than one incoming
     |          layers.
     |        RuntimeError: if called in Eager mode.
     |  
     |  output_mask
     |      Retrieves the output mask tensor(s) of a layer.
     |      
     |      Only applicable if the layer has exactly one inbound node,
     |      i.e. if it is connected to one incoming layer.
     |      
     |      Returns:
     |          Output mask tensor (potentially None) or list of output
     |          mask tensors.
     |      
     |      Raises:
     |          AttributeError: if the layer is connected to
     |          more than one incoming layers.
     |  
     |  output_shape
     |      Retrieves the output shape(s) of a layer.
     |      
     |      Only applicable if the layer has one output,
     |      or if all outputs have the same shape.
     |      
     |      Returns:
     |          Output shape, as an integer shape tuple
     |          (or list of shape tuples, one tuple per output tensor).
     |      
     |      Raises:
     |          AttributeError: if the layer has no defined output shape.
     |          RuntimeError: if called in Eager mode.
     |  
     |  trainable_variables
     |      Sequence of trainable variables owned by this module and its submodules.
     |      
     |      Note: this method uses reflection to find variables on the current instance
     |      and submodules. For performance reasons you may wish to cache the result
     |      of calling this method if you don't expect the return value to change.
     |      
     |      Returns:
     |        A sequence of variables for the current module (sorted by attribute
     |        name) followed by variables from all submodules recursively (breadth
     |        first).
     |  
     |  updates
     |  
     |  variable_dtype
     |      Alias of `Layer.dtype`, the dtype of the weights.
     |  
     |  variables
     |      Returns the list of all layer variables/weights.
     |      
     |      Alias of `self.weights`.
     |      
     |      Note: This will not track the weights of nested `tf.Modules` that are
     |      not themselves Keras layers.
     |      
     |      Returns:
     |        A list of variables.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from keras.engine.base_layer.Layer:
     |  
     |  activity_regularizer
     |      Optional regularizer function for the output of this layer.
     |  
     |  input_spec
     |      `InputSpec` instance(s) describing the input format for this layer.
     |      
     |      When you create a layer subclass, you can set `self.input_spec` to
     |      enable the layer to run input compatibility checks when it is called.
     |      Consider a `Conv2D` layer: it can only be called on a single input
     |      tensor of rank 4. As such, you can set, in `__init__()`:
     |      
     |      ```python
     |      self.input_spec = tf.keras.layers.InputSpec(ndim=4)
     |      ```
     |      
     |      Now, if you try to call the layer on an input that isn't rank 4
     |      (for instance, an input of shape `(2,)`, it will raise a
     |      nicely-formatted error:
     |      
     |      ```
     |      ValueError: Input 0 of layer conv2d is incompatible with the layer:
     |      expected ndim=4, found ndim=1. Full shape received: [2]
     |      ```
     |      
     |      Input checks that can be specified via `input_spec` include:
     |      - Structure (e.g. a single input, a list of 2 inputs, etc)
     |      - Shape
     |      - Rank (ndim)
     |      - Dtype
     |      
     |      For more information, see `tf.keras.layers.InputSpec`.
     |      
     |      Returns:
     |        A `tf.keras.layers.InputSpec` instance, or nested structure thereof.
     |  
     |  stateful
     |  
     |  supports_masking
     |      Whether this layer supports computing a mask using `compute_mask`.
     |  
     |  trainable
     |  
     |  ----------------------------------------------------------------------
     |  Class methods inherited from tensorflow.python.module.module.Module:
     |  
     |  with_name_scope(method) from builtins.type
     |      Decorator to automatically enter the module name scope.
     |      
     |      >>> class MyModule(tf.Module):
     |      ...   @tf.Module.with_name_scope
     |      ...   def __call__(self, x):
     |      ...     if not hasattr(self, 'w'):
     |      ...       self.w = tf.Variable(tf.random.normal([x.shape[1], 3]))
     |      ...     return tf.matmul(x, self.w)
     |      
     |      Using the above module would produce `tf.Variable`s and `tf.Tensor`s whose
     |      names included the module name:
     |      
     |      >>> mod = MyModule()
     |      >>> mod(tf.ones([1, 2]))
     |      <tf.Tensor: shape=(1, 3), dtype=float32, numpy=..., dtype=float32)>
     |      >>> mod.w
     |      <tf.Variable 'my_module/Variable:0' shape=(2, 3) dtype=float32,
     |      numpy=..., dtype=float32)>
     |      
     |      Args:
     |        method: The method to wrap.
     |      
     |      Returns:
     |        The original method wrapped such that it enters the module's name scope.
     |  
     |  ----------------------------------------------------------------------
     |  Readonly properties inherited from tensorflow.python.module.module.Module:
     |  
     |  name_scope
     |      Returns a `tf.name_scope` instance for this class.
     |  
     |  submodules
     |      Sequence of all sub-modules.
     |      
     |      Submodules are modules which are properties of this module, or found as
     |      properties of modules which are properties of this module (and so on).
     |      
     |      >>> a = tf.Module()
     |      >>> b = tf.Module()
     |      >>> c = tf.Module()
     |      >>> a.b = b
     |      >>> b.c = c
     |      >>> list(a.submodules) == [b, c]
     |      True
     |      >>> list(b.submodules) == [c]
     |      True
     |      >>> list(c.submodules) == []
     |      True
     |      
     |      Returns:
     |        A sequence of all submodules.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from tensorflow.python.trackable.base.Trackable:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    

## Using a subset of features

The previous example did not specify the features, so all the columns were used
as input feature (except for the label). The following example shows how to
specify input features.


```python
feature_1 = tfdf.keras.FeatureUsage(name="bill_length_mm")
feature_2 = tfdf.keras.FeatureUsage(name="island")

all_features = [feature_1, feature_2]

# Note: This model is only trained with two features. It will not be as good as
# the one trained on all features.

model_2 = tfdf.keras.GradientBoostedTreesModel(
    features=all_features, exclude_non_specified_features=True)

model_2.compile(metrics=["accuracy"])
model_2.fit(train_ds, validation_data=test_ds)

print(model_2.evaluate(test_ds, return_dict=True))
```

    Warning: The `num_threads` constructor argument is not set and the number of CPU is os.cpu_count()=32 > 32. Setting num_threads to 32. Set num_threads manually to use more than 32 cpus.
    

    WARNING:absl:The `num_threads` constructor argument is not set and the number of CPU is os.cpu_count()=32 > 32. Setting num_threads to 32. Set num_threads manually to use more than 32 cpus.
    

    Use /tmpfs/tmp/tmpsuqcd1gs as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.128138. Found 241 examples.
    Reading validation dataset...
    

    [WARNING 23-05-23 11:13:13.2288 UTC gradient_boosted_trees.cc:1797] "goss_alpha" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-23 11:13:13.2288 UTC gradient_boosted_trees.cc:1808] "goss_beta" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-23 11:13:13.2288 UTC gradient_boosted_trees.cc:1822] "selective_gradient_boosting_ratio" set but "sampling_method" not equal to "SELGB".
    

    Num validation examples: tf.Tensor(103, shape=(), dtype=int32)
    Validation dataset read in 0:00:00.187595. Found 103 examples.
    Training model...
    Model trained in 0:00:00.410462
    Compiling model...
    Model compiled.
    

    [INFO 23-05-23 11:13:13.9547 UTC kernel.cc:1242] Loading model from path /tmpfs/tmp/tmpsuqcd1gs/model/ with prefix 9f995efd4e9a4def
    [INFO 23-05-23 11:13:13.9639 UTC decision_forest.cc:660] Model loaded with 105 root(s), 3217 node(s), and 2 input feature(s).
    [INFO 23-05-23 11:13:13.9639 UTC kernel.cc:1074] Use fast generic engine
    

    1/1 [==============================] - 0s 76ms/step - loss: 0.0000e+00 - accuracy: 0.9806
    {'loss': 0.0, 'accuracy': 0.9805825352668762}
    

**Note:** As expected, the accuracy is lower than previously.

**TF-DF** attaches a **semantics** to each feature. This semantics controls how
the feature is used by the model. The following semantics are currently supported:

-   **Numerical**: Generally for quantities or counts with full ordering. For
    example, the age of a person, or the number of items in a bag. Can be a
    float or an integer. Missing values are represented with float(Nan) or with
    an empty sparse tensor.
-   **Categorical**: Generally for a type/class in finite set of possible values
    without ordering. For example, the color RED in the set {RED, BLUE, GREEN}.
    Can be a string or an integer. Missing values are represented as "" (empty
    sting), value -2 or with an empty sparse tensor.
-   **Categorical-Set**: A set of categorical values. Great to represent
    tokenized text. Can be a string or an integer in a sparse tensor or a
    ragged tensor (recommended). The order/index of each item doesn't matter.

If not specified, the semantics is inferred from the representation type and shown in the training logs:

- int, float (dense or sparse) → Numerical semantics.
- str (dense or sparse) → Categorical semantics
- int, str (ragged) → Categorical-Set semantics

In some cases, the inferred semantics is incorrect. For example: An Enum stored as an integer is semantically categorical, but it will be detected as numerical. In this case, you should specify the semantic argument in the input. The `education_num` field of the Adult dataset is classical example.

This dataset doesn't contain such a feature. However, for the demonstration, we will make the model treat the `year` as a categorical feature:


```python
%set_cell_height 300

feature_1 = tfdf.keras.FeatureUsage(name="year", semantic=tfdf.keras.FeatureSemantic.CATEGORICAL)
feature_2 = tfdf.keras.FeatureUsage(name="bill_length_mm")
feature_3 = tfdf.keras.FeatureUsage(name="sex")
all_features = [feature_1, feature_2, feature_3]

model_3 = tfdf.keras.GradientBoostedTreesModel(features=all_features, exclude_non_specified_features=True)
model_3.compile( metrics=["accuracy"])

model_3.fit(train_ds, validation_data=test_ds)
```


    <IPython.core.display.Javascript object>


    Warning: The `num_threads` constructor argument is not set and the number of CPU is os.cpu_count()=32 > 32. Setting num_threads to 32. Set num_threads manually to use more than 32 cpus.
    

    WARNING:absl:The `num_threads` constructor argument is not set and the number of CPU is os.cpu_count()=32 > 32. Setting num_threads to 32. Set num_threads manually to use more than 32 cpus.
    

    Use /tmpfs/tmp/tmp1dudne9k as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.128923. Found 241 examples.
    Reading validation dataset...
    

    [WARNING 23-05-23 11:13:14.2540 UTC gradient_boosted_trees.cc:1797] "goss_alpha" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-23 11:13:14.2540 UTC gradient_boosted_trees.cc:1808] "goss_beta" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-23 11:13:14.2541 UTC gradient_boosted_trees.cc:1822] "selective_gradient_boosting_ratio" set but "sampling_method" not equal to "SELGB".
    

    Num validation examples: tf.Tensor(103, shape=(), dtype=int32)
    Validation dataset read in 0:00:00.130643. Found 103 examples.
    Training model...
    Model trained in 0:00:00.246664
    Compiling model...
    

    [INFO 23-05-23 11:13:14.7639 UTC kernel.cc:1242] Loading model from path /tmpfs/tmp/tmp1dudne9k/model/ with prefix 0eda8015aa654fcb
    [INFO 23-05-23 11:13:14.7683 UTC decision_forest.cc:660] Model loaded with 39 root(s), 1217 node(s), and 3 input feature(s).
    [INFO 23-05-23 11:13:14.7683 UTC kernel.cc:1074] Use fast generic engine
    

    Model compiled.
    




    <keras.callbacks.History at 0x7ff304109d00>



Note that `year` is in the list of CATEGORICAL features (unlike the first run).

## Hyper-parameters

**Hyper-parameters** are parameters of the training algorithm that impact
the quality of the final model. They are specified in the model class
constructor. The list of hyper-parameters is visible with the *question mark* colab command (e.g. `?tfdf.keras.GradientBoostedTreesModel`).

Alternatively, you can find them on the [TensorFlow Decision Forest Github](https://github.com/tensorflow/decision-forests/blob/main/tensorflow_decision_forests/keras/wrappers_pre_generated.py) or the [Yggdrasil Decision Forest documentation](https://github.com/google/yggdrasil-decision-forests/blob/main/documentation/learners.md).

The default hyper-parameters of each algorithm matches approximatively the initial publication paper. To ensure consistancy, new features and their matching hyper-parameters are always disable by default. That's why it is a good idea to tune your hyper-parameters.


```python
# A classical but slighly more complex model.
model_6 = tfdf.keras.GradientBoostedTreesModel(
    num_trees=500, growing_strategy="BEST_FIRST_GLOBAL", max_depth=8)
model_6.fit(train_ds)
```

    Warning: The `num_threads` constructor argument is not set and the number of CPU is os.cpu_count()=32 > 32. Setting num_threads to 32. Set num_threads manually to use more than 32 cpus.
    

    WARNING:absl:The `num_threads` constructor argument is not set and the number of CPU is os.cpu_count()=32 > 32. Setting num_threads to 32. Set num_threads manually to use more than 32 cpus.
    

    Use /tmpfs/tmp/tmp8pjtiqsv as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.150881. Found 241 examples.
    Training model...
    

    [WARNING 23-05-23 11:13:15.1435 UTC gradient_boosted_trees.cc:1797] "goss_alpha" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-23 11:13:15.1435 UTC gradient_boosted_trees.cc:1808] "goss_beta" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-23 11:13:15.1435 UTC gradient_boosted_trees.cc:1822] "selective_gradient_boosting_ratio" set but "sampling_method" not equal to "SELGB".
    

    Model trained in 0:00:00.973713
    Compiling model...
    Model compiled.
    

    [INFO 23-05-23 11:13:16.2419 UTC kernel.cc:1242] Loading model from path /tmpfs/tmp/tmp8pjtiqsv/model/ with prefix 0c808d111ff0418e
    [INFO 23-05-23 11:13:16.2732 UTC decision_forest.cc:660] Model loaded with 210 root(s), 10554 node(s), and 7 input feature(s).
    [INFO 23-05-23 11:13:16.2732 UTC kernel.cc:1074] Use fast generic engine
    




    <keras.callbacks.History at 0x7ff3040fd8e0>




```python
# A more complex, but possibly, more accurate model.
model_7 = tfdf.keras.GradientBoostedTreesModel(
    num_trees=500,
    growing_strategy="BEST_FIRST_GLOBAL",
    max_depth=8,
    split_axis="SPARSE_OBLIQUE",
    categorical_algorithm="RANDOM",
    )
model_7.fit(train_ds)
```

    Warning: The `num_threads` constructor argument is not set and the number of CPU is os.cpu_count()=32 > 32. Setting num_threads to 32. Set num_threads manually to use more than 32 cpus.
    

    WARNING:absl:The `num_threads` constructor argument is not set and the number of CPU is os.cpu_count()=32 > 32. Setting num_threads to 32. Set num_threads manually to use more than 32 cpus.
    

    Use /tmpfs/tmp/tmpcghquc24 as temporary training directory
    Reading training dataset...
    

    [WARNING 23-05-23 11:13:16.3986 UTC gradient_boosted_trees.cc:1797] "goss_alpha" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-23 11:13:16.3986 UTC gradient_boosted_trees.cc:1808] "goss_beta" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-23 11:13:16.3986 UTC gradient_boosted_trees.cc:1822] "selective_gradient_boosting_ratio" set but "sampling_method" not equal to "SELGB".
    

    WARNING:tensorflow:5 out of the last 5 calls to <function CoreModel._consumes_training_examples_until_eof at 0x7ff38e0a50d0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    

    WARNING:tensorflow:5 out of the last 5 calls to <function CoreModel._consumes_training_examples_until_eof at 0x7ff38e0a50d0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    

    Training dataset read in 0:00:00.155373. Found 241 examples.
    Training model...
    

    [INFO 23-05-23 11:13:24.4547 UTC kernel.cc:1242] Loading model from path /tmpfs/tmp/tmpcghquc24/model/ with prefix 3cc7feeb39284462
    

    Model trained in 0:00:08.176121
    Compiling model...
    WARNING:tensorflow:5 out of the last 5 calls to <function InferenceCoreModel.make_predict_function.<locals>.predict_function_trained at 0x7ff3044313a0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    

    [INFO 23-05-23 11:13:24.7230 UTC decision_forest.cc:660] Model loaded with 1497 root(s), 84977 node(s), and 7 input feature(s).
    [INFO 23-05-23 11:13:24.7231 UTC abstract_model.cc:1311] Engine "GradientBoostedTreesGeneric" built
    [INFO 23-05-23 11:13:24.7231 UTC kernel.cc:1074] Use fast generic engine
    WARNING:tensorflow:5 out of the last 5 calls to <function InferenceCoreModel.make_predict_function.<locals>.predict_function_trained at 0x7ff3044313a0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    

    Model compiled.
    




    <keras.callbacks.History at 0x7ff3ac723ca0>



As new training methods are published and implemented, combination of hyper-parameters can emerge as good or almost-always-better than the default parameters. To avoid changing the default hyper-parameter values these good combination are indexed and available as hyper-parameter templates.

For example, the `benchmark_rank1` template is the best combination on our internal benchmarks. Those templates are versioned to allow training configuration stability e.g. `benchmark_rank1@v1`.


```python
# A good template of hyper-parameters.
model_8 = tfdf.keras.GradientBoostedTreesModel(hyperparameter_template="benchmark_rank1")
model_8.fit(train_ds)
```

    Resolve hyper-parameter template "benchmark_rank1" to "benchmark_rank1@v1" -> {'growing_strategy': 'BEST_FIRST_GLOBAL', 'categorical_algorithm': 'RANDOM', 'split_axis': 'SPARSE_OBLIQUE', 'sparse_oblique_normalization': 'MIN_MAX', 'sparse_oblique_num_projections_exponent': 1.0}.
    Warning: The `num_threads` constructor argument is not set and the number of CPU is os.cpu_count()=32 > 32. Setting num_threads to 32. Set num_threads manually to use more than 32 cpus.
    

    WARNING:absl:The `num_threads` constructor argument is not set and the number of CPU is os.cpu_count()=32 > 32. Setting num_threads to 32. Set num_threads manually to use more than 32 cpus.
    

    Use /tmpfs/tmp/tmpnxhdgqgf as temporary training directory
    

    [WARNING 23-05-23 11:13:24.9198 UTC gradient_boosted_trees.cc:1797] "goss_alpha" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-23 11:13:24.9198 UTC gradient_boosted_trees.cc:1808] "goss_beta" set but "sampling_method" not equal to "GOSS".
    [WARNING 23-05-23 11:13:24.9198 UTC gradient_boosted_trees.cc:1822] "selective_gradient_boosting_ratio" set but "sampling_method" not equal to "SELGB".
    

    Reading training dataset...
    WARNING:tensorflow:6 out of the last 6 calls to <function CoreModel._consumes_training_examples_until_eof at 0x7ff38e0a50d0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    

    WARNING:tensorflow:6 out of the last 6 calls to <function CoreModel._consumes_training_examples_until_eof at 0x7ff38e0a50d0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    

    Training dataset read in 0:00:00.154354. Found 241 examples.
    Training model...
    Model trained in 0:00:03.194846
    Compiling model...
    

    [INFO 23-05-23 11:13:28.1708 UTC kernel.cc:1242] Loading model from path /tmpfs/tmp/tmpnxhdgqgf/model/ with prefix 1d250bfdc06945af
    [INFO 23-05-23 11:13:28.2709 UTC decision_forest.cc:660] Model loaded with 900 root(s), 34054 node(s), and 7 input feature(s).
    [INFO 23-05-23 11:13:28.2709 UTC kernel.cc:1074] Use fast generic engine
    

    WARNING:tensorflow:6 out of the last 6 calls to <function InferenceCoreModel.make_predict_function.<locals>.predict_function_trained at 0x7ff2947eed30> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    

    WARNING:tensorflow:6 out of the last 6 calls to <function InferenceCoreModel.make_predict_function.<locals>.predict_function_trained at 0x7ff2947eed30> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    

    Model compiled.
    




    <keras.callbacks.History at 0x7ff2947c09d0>



The available templates are available with `predefined_hyperparameters`. Note that different learning algorithms have different templates, even if the name is similar.


```python
# The hyper-parameter templates of the Gradient Boosted Tree model.
print(tfdf.keras.GradientBoostedTreesModel.predefined_hyperparameters())
```

    [HyperParameterTemplate(name='better_default', version=1, parameters={'growing_strategy': 'BEST_FIRST_GLOBAL'}, description='A configuration that is generally better than the default parameters without being more expensive.'), HyperParameterTemplate(name='benchmark_rank1', version=1, parameters={'growing_strategy': 'BEST_FIRST_GLOBAL', 'categorical_algorithm': 'RANDOM', 'split_axis': 'SPARSE_OBLIQUE', 'sparse_oblique_normalization': 'MIN_MAX', 'sparse_oblique_num_projections_exponent': 1.0}, description='Top ranking hyper-parameters on our benchmark slightly modified to run in reasonable time.')]
    

## Feature Preprocessing

Pre-processing features is sometimes necessary to consume signals with complex
structures, to regularize the model or to apply transfer learning.
Pre-processing can be done in one of three ways:

1.  Preprocessing on the Pandas dataframe. This solution is easy to implement
    and generally suitable for experimentation. However, the
    pre-processing logic will not be exported in the model by `model.save()`.

2.  [Keras Preprocessing](https://keras.io/guides/preprocessing_layers/): While
    more complex than the previous solution, Keras Preprocessing is packaged in
    the model.

3.  [TensorFlow Feature Columns](https://www.tensorflow.org/tutorials/structured_data/feature_columns):
    This API is part of the TF Estimator library (!= Keras) and planned for
    deprecation. This solution is interesting when using existing preprocessing
    code.

Note: Using [TensorFlow Hub](https://www.tensorflow.org/hub)
pre-trained embedding is often, a great way to consume text and image with
TF-DF. For example, `hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim128/2")`. See the [Intermediate tutorial](intermediate_colab.ipynb) for more details.

In the next example, pre-process the `body_mass_g` feature into `body_mass_kg = body_mass_g / 1000`. The `bill_length_mm` is consumed without pre-processing. Note that such
monotonic transformations have generally no impact on decision forest models.


```python
%set_cell_height 300

body_mass_g = tf.keras.layers.Input(shape=(1,), name="body_mass_g")
body_mass_kg = body_mass_g / 1000.0

bill_length_mm = tf.keras.layers.Input(shape=(1,), name="bill_length_mm")

raw_inputs = {"body_mass_g": body_mass_g, "bill_length_mm": bill_length_mm}
processed_inputs = {"body_mass_kg": body_mass_kg, "bill_length_mm": bill_length_mm}

# "preprocessor" contains the preprocessing logic.
preprocessor = tf.keras.Model(inputs=raw_inputs, outputs=processed_inputs)

# "model_4" contains both the pre-processing logic and the decision forest.
model_4 = tfdf.keras.RandomForestModel(preprocessing=preprocessor)
model_4.fit(train_ds)

model_4.summary()
```


    <IPython.core.display.Javascript object>


    Warning: The `num_threads` constructor argument is not set and the number of CPU is os.cpu_count()=32 > 32. Setting num_threads to 32. Set num_threads manually to use more than 32 cpus.
    

    WARNING:absl:The `num_threads` constructor argument is not set and the number of CPU is os.cpu_count()=32 > 32. Setting num_threads to 32. Set num_threads manually to use more than 32 cpus.
    

    Use /tmpfs/tmp/tmpwbry8jnx as temporary training directory
    Reading training dataset...
    

    /tmpfs/src/tf_docs_env/lib/python3.9/site-packages/keras/engine/functional.py:639: UserWarning: Input dict contained keys ['island', 'bill_depth_mm', 'flipper_length_mm', 'sex', 'year'] which did not match any model input. They will be ignored by the model.
      inputs = self._flatten_to_reference_inputs(inputs)
    

    Training dataset read in 0:00:00.208845. Found 241 examples.
    Training model...
    Model trained in 0:00:00.067024
    Compiling model...
    Model compiled.
    WARNING:tensorflow:5 out of the last 12 calls to <function InferenceCoreModel.yggdrasil_model_path_tensor at 0x7ff2946a71f0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    

    [INFO 23-05-23 11:13:28.7054 UTC kernel.cc:1242] Loading model from path /tmpfs/tmp/tmpwbry8jnx/model/ with prefix 965f86f6494c4a04
    [INFO 23-05-23 11:13:28.7220 UTC decision_forest.cc:660] Model loaded with 300 root(s), 6118 node(s), and 2 input feature(s).
    [INFO 23-05-23 11:13:28.7221 UTC kernel.cc:1074] Use fast generic engine
    WARNING:tensorflow:5 out of the last 12 calls to <function InferenceCoreModel.yggdrasil_model_path_tensor at 0x7ff2946a71f0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    

    Model: "random_forest_model_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     model (Functional)          {'body_mass_kg': (None,   0         
                                 1),                                 
                                  'bill_length_mm': (None            
                                 , 1)}                               
                                                                     
    =================================================================
    Total params: 1
    Trainable params: 0
    Non-trainable params: 1
    _________________________________________________________________
    Type: "RANDOM_FOREST"
    Task: CLASSIFICATION
    Label: "__LABEL"
    
    Input Features (2):
    	bill_length_mm
    	body_mass_kg
    
    No weights
    
    Variable Importance: INV_MEAN_MIN_DEPTH:
        1. "bill_length_mm"  0.961538 ################
        2.   "body_mass_kg"  0.466010 
    
    Variable Importance: NUM_AS_ROOT:
        1. "bill_length_mm" 288.000000 ################
        2.   "body_mass_kg" 12.000000 
    
    Variable Importance: NUM_NODES:
        1. "bill_length_mm" 1513.000000 ################
        2.   "body_mass_kg" 1396.000000 
    
    Variable Importance: SUM_SCORE:
        1. "bill_length_mm" 41750.624108 ################
        2.   "body_mass_kg" 25570.177907 
    
    
    
    Winner takes all: true
    Out-of-bag evaluation: accuracy:0.929461 logloss:0.636188
    Number of trees: 300
    Total number of nodes: 6118
    
    Number of nodes by tree:
    Count: 300 Average: 20.3933 StdDev: 3.23604
    Min: 13 Max: 29 Ignored: 0
    ----------------------------------------------
    [ 13, 14) 12   4.00%   4.00% #
    [ 14, 15)  0   0.00%   4.00%
    [ 15, 16) 17   5.67%   9.67% ##
    [ 16, 17)  0   0.00%   9.67%
    [ 17, 18) 38  12.67%  22.33% #####
    [ 18, 19)  0   0.00%  22.33%
    [ 19, 20) 57  19.00%  41.33% #######
    [ 20, 21)  0   0.00%  41.33%
    [ 21, 22) 84  28.00%  69.33% ##########
    [ 22, 23)  0   0.00%  69.33%
    [ 23, 24) 52  17.33%  86.67% ######
    [ 24, 25)  0   0.00%  86.67%
    [ 25, 26) 32  10.67%  97.33% ####
    [ 26, 27)  0   0.00%  97.33%
    [ 27, 28)  7   2.33%  99.67% #
    [ 28, 29)  0   0.00%  99.67%
    [ 29, 29]  1   0.33% 100.00%
    
    Depth by leafs:
    Count: 3209 Average: 3.86725 StdDev: 1.2158
    Min: 1 Max: 8 Ignored: 0
    ----------------------------------------------
    [ 1, 2)   13   0.41%   0.41%
    [ 2, 3)  359  11.19%  11.59% ####
    [ 3, 4)  948  29.54%  41.13% #########
    [ 4, 5) 1020  31.79%  72.92% ##########
    [ 5, 6)  551  17.17%  90.09% #####
    [ 6, 7)  234   7.29%  97.38% ##
    [ 7, 8)   76   2.37%  99.75% #
    [ 8, 8]    8   0.25% 100.00%
    
    Number of training obs by leaf:
    Count: 3209 Average: 22.5304 StdDev: 28.4234
    Min: 5 Max: 116 Ignored: 0
    ----------------------------------------------
    [   5,  10) 2042  63.63%  63.63% ##########
    [  10,  16)  229   7.14%  70.77% #
    [  16,  21)   46   1.43%  72.20%
    [  21,  27)   58   1.81%  74.01%
    [  27,  33)   96   2.99%  77.00%
    [  33,  38)   94   2.93%  79.93%
    [  38,  44)   38   1.18%  81.12%
    [  44,  49)   12   0.37%  81.49%
    [  49,  55)   36   1.12%  82.61%
    [  55,  61)   55   1.71%  84.33%
    [  61,  66)   55   1.71%  86.04%
    [  66,  72)   62   1.93%  87.97%
    [  72,  77)   66   2.06%  90.03%
    [  77,  83)   78   2.43%  92.46%
    [  83,  89)   74   2.31%  94.76%
    [  89,  94)   64   1.99%  96.76%
    [  94, 100)   48   1.50%  98.25%
    [ 100, 105)   35   1.09%  99.35%
    [ 105, 111)   16   0.50%  99.84%
    [ 111, 116]    5   0.16% 100.00%
    
    Attribute in nodes:
    	1513 : bill_length_mm [NUMERICAL]
    	1396 : body_mass_kg [NUMERICAL]
    
    Attribute in nodes with depth <= 0:
    	288 : bill_length_mm [NUMERICAL]
    	12 : body_mass_kg [NUMERICAL]
    
    Attribute in nodes with depth <= 1:
    	450 : body_mass_kg [NUMERICAL]
    	437 : bill_length_mm [NUMERICAL]
    
    Attribute in nodes with depth <= 2:
    	854 : bill_length_mm [NUMERICAL]
    	848 : body_mass_kg [NUMERICAL]
    
    Attribute in nodes with depth <= 3:
    	1201 : bill_length_mm [NUMERICAL]
    	1183 : body_mass_kg [NUMERICAL]
    
    Attribute in nodes with depth <= 5:
    	1483 : bill_length_mm [NUMERICAL]
    	1382 : body_mass_kg [NUMERICAL]
    
    Condition type in nodes:
    	2909 : HigherCondition
    Condition type in nodes with depth <= 0:
    	300 : HigherCondition
    Condition type in nodes with depth <= 1:
    	887 : HigherCondition
    Condition type in nodes with depth <= 2:
    	1702 : HigherCondition
    Condition type in nodes with depth <= 3:
    	2384 : HigherCondition
    Condition type in nodes with depth <= 5:
    	2865 : HigherCondition
    Node format: NOT_SET
    
    Training OOB:
    	trees: 1, Out-of-bag evaluation: accuracy:0.895349 logloss:3.77201
    	trees: 12, Out-of-bag evaluation: accuracy:0.891213 logloss:2.48003
    	trees: 22, Out-of-bag evaluation: accuracy:0.895833 logloss:2.04075
    	trees: 32, Out-of-bag evaluation: accuracy:0.904564 logloss:1.5999
    	trees: 45, Out-of-bag evaluation: accuracy:0.908714 logloss:1.4535
    	trees: 55, Out-of-bag evaluation: accuracy:0.917012 logloss:1.17579
    	trees: 65, Out-of-bag evaluation: accuracy:0.925311 logloss:1.03351
    	trees: 75, Out-of-bag evaluation: accuracy:0.925311 logloss:0.891803
    	trees: 85, Out-of-bag evaluation: accuracy:0.925311 logloss:0.759392
    	trees: 95, Out-of-bag evaluation: accuracy:0.925311 logloss:0.766698
    	trees: 105, Out-of-bag evaluation: accuracy:0.929461 logloss:0.771698
    	trees: 115, Out-of-bag evaluation: accuracy:0.921162 logloss:0.772715
    	trees: 125, Out-of-bag evaluation: accuracy:0.925311 logloss:0.773843
    	trees: 135, Out-of-bag evaluation: accuracy:0.921162 logloss:0.77514
    	trees: 145, Out-of-bag evaluation: accuracy:0.925311 logloss:0.776201
    	trees: 156, Out-of-bag evaluation: accuracy:0.925311 logloss:0.769409
    	trees: 167, Out-of-bag evaluation: accuracy:0.93361 logloss:0.768375
    	trees: 178, Out-of-bag evaluation: accuracy:0.93361 logloss:0.768264
    	trees: 188, Out-of-bag evaluation: accuracy:0.929461 logloss:0.769529
    	trees: 199, Out-of-bag evaluation: accuracy:0.929461 logloss:0.76694
    	trees: 209, Out-of-bag evaluation: accuracy:0.929461 logloss:0.768219
    	trees: 219, Out-of-bag evaluation: accuracy:0.929461 logloss:0.635432
    	trees: 229, Out-of-bag evaluation: accuracy:0.929461 logloss:0.634153
    	trees: 239, Out-of-bag evaluation: accuracy:0.929461 logloss:0.633093
    	trees: 249, Out-of-bag evaluation: accuracy:0.929461 logloss:0.634412
    	trees: 259, Out-of-bag evaluation: accuracy:0.929461 logloss:0.636291
    	trees: 269, Out-of-bag evaluation: accuracy:0.929461 logloss:0.637012
    	trees: 279, Out-of-bag evaluation: accuracy:0.929461 logloss:0.637831
    	trees: 289, Out-of-bag evaluation: accuracy:0.929461 logloss:0.638668
    	trees: 299, Out-of-bag evaluation: accuracy:0.929461 logloss:0.635998
    	trees: 300, Out-of-bag evaluation: accuracy:0.929461 logloss:0.636188
    
    

The following example re-implements the same logic using TensorFlow Feature
Columns.


```python
def g_to_kg(x):
  return x / 1000

feature_columns = [
    tf.feature_column.numeric_column("body_mass_g", normalizer_fn=g_to_kg),
    tf.feature_column.numeric_column("bill_length_mm"),
]

preprocessing = tf.keras.layers.DenseFeatures(feature_columns)

model_5 = tfdf.keras.RandomForestModel(preprocessing=preprocessing)
model_5.fit(train_ds)
```

    WARNING:tensorflow:From /tmpfs/tmp/ipykernel_10871/2850711544.py:5: numeric_column (from tensorflow.python.feature_column.feature_column_v2) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use Keras preprocessing layers instead, either directly or via the `tf.keras.utils.FeatureSpace` utility. Each of `tf.feature_column.*` has a functional equivalent in `tf.keras.layers` for feature preprocessing when training a Keras model.
    

    WARNING:tensorflow:From /tmpfs/tmp/ipykernel_10871/2850711544.py:5: numeric_column (from tensorflow.python.feature_column.feature_column_v2) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use Keras preprocessing layers instead, either directly or via the `tf.keras.utils.FeatureSpace` utility. Each of `tf.feature_column.*` has a functional equivalent in `tf.keras.layers` for feature preprocessing when training a Keras model.
    

    Warning: The `num_threads` constructor argument is not set and the number of CPU is os.cpu_count()=32 > 32. Setting num_threads to 32. Set num_threads manually to use more than 32 cpus.
    

    WARNING:absl:The `num_threads` constructor argument is not set and the number of CPU is os.cpu_count()=32 > 32. Setting num_threads to 32. Set num_threads manually to use more than 32 cpus.
    

    Use /tmpfs/tmp/tmp86_7ftlw as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.147273. Found 241 examples.
    Training model...
    

    [INFO 23-05-23 11:13:29.0283 UTC kernel.cc:1242] Loading model from path /tmpfs/tmp/tmp86_7ftlw/model/ with prefix 698955d0b1c448d3
    

    Model trained in 0:00:00.046979
    Compiling model...
    Model compiled.
    WARNING:tensorflow:6 out of the last 13 calls to <function InferenceCoreModel.yggdrasil_model_path_tensor at 0x7ff294645ee0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    

    [INFO 23-05-23 11:13:29.0459 UTC decision_forest.cc:660] Model loaded with 300 root(s), 6118 node(s), and 2 input feature(s).
    [INFO 23-05-23 11:13:29.0460 UTC kernel.cc:1074] Use fast generic engine
    WARNING:tensorflow:6 out of the last 13 calls to <function InferenceCoreModel.yggdrasil_model_path_tensor at 0x7ff294645ee0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    




    <keras.callbacks.History at 0x7ff2945cc460>



## Training a regression model

The previous example trains a classification model (TF-DF does not differentiate
between binary classification and multi-class classification). In the next
example, train a regression model on the
[Abalone dataset](https://archive.ics.uci.edu/ml/datasets/abalone). The
objective of this dataset is to predict the number of shell's rings of an
abalone.

**Note:** The csv file is assembled by appending UCI's header and data files. No preprocessing was applied.

<center>
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/LivingAbalone.JPG/800px-LivingAbalone.JPG" width="200"/></center>


```python
# Download the dataset.
!wget -q https://storage.googleapis.com/download.tensorflow.org/data/abalone_raw.csv -O /tmp/abalone.csv

dataset_df = pd.read_csv("/tmp/abalone.csv")
print(dataset_df.head(3))
```

      Type  LongestShell  Diameter  Height  WholeWeight  ShuckedWeight   
    0    M         0.455     0.365   0.095       0.5140         0.2245  \
    1    M         0.350     0.265   0.090       0.2255         0.0995   
    2    F         0.530     0.420   0.135       0.6770         0.2565   
    
       VisceraWeight  ShellWeight  Rings  
    0         0.1010         0.15     15  
    1         0.0485         0.07      7  
    2         0.1415         0.21      9  
    


```python
# Split the dataset into a training and testing dataset.
train_ds_pd, test_ds_pd = split_dataset(dataset_df)
print("{} examples in training, {} examples for testing.".format(
    len(train_ds_pd), len(test_ds_pd)))

# Name of the label column.
label = "Rings"

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)
```

    2956 examples in training, 1221 examples for testing.
    


```python
%set_cell_height 300

# Configure the model.
model_7 = tfdf.keras.RandomForestModel(task = tfdf.keras.Task.REGRESSION)

# Train the model.
model_7.fit(train_ds)
```


    <IPython.core.display.Javascript object>


    Warning: The `num_threads` constructor argument is not set and the number of CPU is os.cpu_count()=32 > 32. Setting num_threads to 32. Set num_threads manually to use more than 32 cpus.
    

    WARNING:absl:The `num_threads` constructor argument is not set and the number of CPU is os.cpu_count()=32 > 32. Setting num_threads to 32. Set num_threads manually to use more than 32 cpus.
    

    Use /tmpfs/tmp/tmpt0vkopwg as temporary training directory
    Reading training dataset...
    Training dataset read in 0:00:00.187019. Found 2956 examples.
    Training model...
    

    [INFO 23-05-23 11:13:30.2652 UTC kernel.cc:1242] Loading model from path /tmpfs/tmp/tmpt0vkopwg/model/ with prefix a4aa0ce679954a6d
    

    Model trained in 0:00:01.378013
    Compiling model...
    

    [INFO 23-05-23 11:13:31.0578 UTC decision_forest.cc:660] Model loaded with 300 root(s), 267130 node(s), and 8 input feature(s).
    [INFO 23-05-23 11:13:31.0579 UTC kernel.cc:1074] Use fast generic engine
    

    Model compiled.
    




    <keras.callbacks.History at 0x7ff29477a850>




```python
# Evaluate the model on the test dataset.
model_7.compile(metrics=["mse"])
evaluation = model_7.evaluate(test_ds, return_dict=True)

print(evaluation)
print()
print(f"MSE: {evaluation['mse']}")
print(f"RMSE: {math.sqrt(evaluation['mse'])}")
```

    WARNING:tensorflow:5 out of the last 5 calls to <function InferenceCoreModel.make_test_function.<locals>.test_function at 0x7ff29456fca0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    

    WARNING:tensorflow:5 out of the last 5 calls to <function InferenceCoreModel.make_test_function.<locals>.test_function at 0x7ff29456fca0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    

    2/2 [==============================] - 0s 12ms/step - loss: 0.0000e+00 - mse: 4.6461
    {'loss': 0.0, 'mse': 4.6460957527160645}
    
    MSE: 4.6460957527160645
    RMSE: 2.1554803995202705
    
