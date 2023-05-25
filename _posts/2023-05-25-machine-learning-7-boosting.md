Welcome to the **7th** tutorial! In the [last tutorial](https://www.kaggle.com/fengdanye/machine-learning-6-basic-ensemble-learning), we introduced basic concepts of ensemble learning, as well as the bagging and pasting techniques. In this tutorial, we will focus on the <font color='blue'>boosting techniques</font>. This tutorial contains more theoretical/technical details than many previous tutorials, but I will try to also include qualitative explanation. However, please try to work through the theories as they will help you understand the boosting techniques.

**Note that there are some notation changes compared to the previous tutorials**. For example, $m$ now is the stage number, instead of sample number, and superscript $(m)$ does not label the sample anymore, but the stage. To help understand some of the boosting algorithms, I have also added an Appendix on the CART algorithm to [my tutorial on random forests](https://www.kaggle.com/fengdanye/machine-learning-5-random-forests). So make sure you read that.

Finally, an apology for spending a whole month on this tutorial before publishing it. I was reading papers to make sure I get things right. This tutorial is longer than usual, and one should prepare to spend more time on this tutorial than any other previous ones. If you have any questions/comments/corrections, let me know!

Now let's begin!

**Table of Content**
* Boosting - overview
* AdaBoost
    * Algorithm
        * Classification
            * Binary-class AdaBoost
            * Multi-class AdaBoost (SAMME)
            * Real-valued SAMME
        * Regression
    * Sklearn functions and examples
        * Example 7.1
        * Example 7.2
* Numerical optimization in function space
    * Review - optimization in parameter space
        * Gradient descent
        * Newton's method
    * Numerical optimization in function space
        * Gradient descent
        * Newton's method
    * Finite data
        * Foward stage-wise additive modeling (FSAM)
* Gradient boosting machine (GBM)
    * General theory
        * Algorithm
    * Gradient tree boosting
        * Introduction
        * Algorithm
        * Regularization
        * Sklearn functions and examples
            * Example 7.3
            * Example 7.4
* Newton boosting
    * General theory
        * Algorithm
    * XGBoost
        * Introduction
        * Regularization
        * Algorithms
        * XGBoost functions and examples
            * Example 7.5
            * Example 7.6|
* References


```python
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, accuracy_score, auc, roc_curve, r2_score

import os
print(os.listdir("../input"))
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Cell In[2], line 3
          1 import numpy as np 
          2 import pandas as pd
    ----> 3 import matplotlib.pyplot as plt
          4 from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor
          5 from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    

    ModuleNotFoundError: No module named 'matplotlib'



```python
plt.rc('axes', lw = 1.5)
plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('xtick.major', size = 5, width = 3)
plt.rc('ytick.major', size = 5, width = 3)
```

# Boosting - overview
Boosting is an ensemble learning technique that fits data by combining many **weak learners (also called base models, or base learners)** into a strong learner. These weak learners usually belong to the same class, e.g. decision trees. Boosting algorithms try to find the optimal linear combination of $M$ weak learners from the same class to minimize prediction error. These weak learners can be considered a set of **basis functions**. The mathematical expression is:
<center>
$F(x)=f_0(x)+\sum_{m=1}^{M}\theta_m \phi_m(x)$
</center>
Where $F(x)$ is the prediction of target value given $x$, $M$ is the number of weak leaners in the ensemble, and $f_0(x)$ is an intial guess. The $\phi_m(x)$ are the weak learners, and $\theta_m$ are the coefficients.

There are many types of boosting techniques. The first practical boosting algorithm is (generally believed to be) <font color='blue'>Adaptive Boosting (AdaBoost)</font>. Then, there are <font color='blue'>Gradient Boosting (GBM)</font>, which correpsonds to gradient descent in function space, and <font color='blue'>XGBoost</font> which correpsonds to Newton's optimization method in function space. The latter two are very popular in current practice. Solving for the weak leaner parameters as well as the coefficients usually end up in a stage wise fashion, which we call **Forward Stagewise Addidtive Modeling (FSAM)**. We will introduce all concepts in detail in the following sections.

# AdaBoost
Adaptive Boosting (AdaBoost) trains one weak learner at a time. **Each weak learner improves on the previous one by assigning larger weights to the previously incorrectly predicted samples**. In this way, the later weak learners focus more on the samples that are difficult to predict. For classification, the final prediction of the ensemble is the class that gets the **weighted majority vote** from all weak learners. For regression, the final prediction is the **weighted median** of predictions made by all regressors.

## Algorithm
Originally, AdaBoost was developed for binary classification ([Freund and Schapire 1996](https://cseweb.ucsd.edu/~yfreund/papers/boostingexperiments.pdf)). Later, the technique was expanded to multi-class classification ([Zhu et al. 2006](https://web.stanford.edu/~hastie/Papers/samme.pdf)) and regression ([Drucker 1997](https://pdfs.semanticscholar.org/8d49/e2dedb817f2c3330e74b63c5fc86d2399ce3.pdf)). We will go through them one by one.
### Classification
#### Binary-class AdaBoost
Assuming we have $n$ samples in the training set, the algorithm starts by initializing equal weights for each training sample:
<center>
    $w_i=\frac{1}{n}, i=1,2,...,n$
</center>
Then:
* For $m = 1$ to $M$:
    * Fit a classifier $T^{(m)}(x)$ to the training data using weights [$w_1$, $w_2$,..., $w_n$]. The higher the weights, the more the classifier tries to get the corresponding samples right. Mathematically speaking, the weak learner’s goal is to find $T^{(m)}(x)$ which minimizes the training error $\mathrm{Pr}_{i \sim w_i}[y_i \neq T^{(m)}(x_i)]$ . Note that this error is measured with respect to the distribution {$w_i$} that was provided to the
weak learner.
    * Compute **weighted error rate** of the current classifier: $err^{(m)}=\sum_{i=1}^n w_i \mathrm{I}(y_i \neq T^{(m)}(x_i))/\sum_{i=1}^n w_i$. Here $y_i$ is the true label of sample $x_i$. The $\mathrm{I}(y_i \neq T^{(m)}(x_i))$ is 1 if  $y_i \neq T^{(m)}(x_i)$, and 0 otherwise.
    * Compute the **classifier weight** in the ensemble: $\alpha^{(m)}=\mathrm{log}((1-err^{(m)})/err^{(m)})$. The smaller the error rate, the more influential the classifier is.
    * **Update sample weights** to boost the incorrectly classified samples: $w_i \leftarrow w_i \cdot exp[\alpha^{(m)}\cdot \mathrm{I}(y_i \neq T^{(m)}(x_i))]$, $i=1,2,...,n$. That is, the misclassified samples have larger weight (when $\alpha^{(m)}>0$).
    * Re-normalize sample weights: $w_i \leftarrow w_i/\sum_{i=1}^n w_i$.
* Output: $\hat{y}(x) = \underset{k}{\operatorname{argmax}}\sum_{m=1}^{M} \alpha^{(m)} \mathrm{I}(T^{(m)}(x) = k)$. The prediction is based on the weighted majority of votes. Here $k=0$ or $1$.
    
The take-away is that **AdaBoost trains one classifier at a time to focus on the previously misclassified cases. The classifiers with lower error rates have larger weights (more votes) in final prediction**. Obviously there are many ways to penalize misclassified cases - why is the algorithm specifically written in this way? You will see in the later section that this algorithm is equivalent to the broadly used forward stagewise additive modeling (FSAM) in boosting algorithms with an exponential loss function. For now, understanding the above algorithm is enough for you to proceed.

#### Multi-class AdaBoost (SAMME)
**You might have noticed that for the binary-class algorithm to actually increase the weights for misclassified samples, $\alpha^{(m)}$ should be positive**. This means the error rate $err^{(m)}$ should be smaller than $1/2$. For a binary classification, this is not two hard to satisfy, since the random error rate with equal sample weights is $1/2$. However, for a $K$-class classification ($K>2$), the random error rate becomes $(K-1)/K$, and $\alpha^{(m)}$ is more likely to be negative. This is a major reason why the previous algorithm does not tend to work well in multi-class cases. To fix this issue, [Zhu et al. 2006](https://web.stanford.edu/~hastie/Papers/samme.pdf) proposed to change the expression of $\alpha^{(m)}$ to:
<center>
    $\alpha^{(m)}=\mathrm{log}\frac{1-err^{(m)}}{err^{(m)}} + log(K-1)$
</center>
The additional $log(K-1)$ term makes sure $\alpha^{(m)}$ is positive as long as the error rate $err^{(m)}$ is smaller than $(K-1)/K$. **This algorithm is refered to as the <font color='blue'>*SAMME*</font> algorithm**. *SAMME* stands for *Stagewise Additive Modeling using a Multi-class Exponential loss function*. We will explain where the name comes from in a later section. When $K=2$, SAMME reduces to binary-class AdaBoost. When $K>2$, we have $log(K-1) > 0$, which means $\alpha^{(m)}$ is bigger than the binary-class case. Consequently, the multi-class AdaBoost penalizes misclassified samples more heavily.

#### Real-valued SAMME
[Zhu et al. 2006](https://web.stanford.edu/~hastie/Papers/samme.pdf) also proposed a variant of the SAMME algorithm that uses **real-valued weighted probability estimates** to update the model, instead of the discrete error rate in SAMME. This algorithm is often called <font color='blue'>**SAMME.R**</font>. **Sklean's <font color='green'>AdaBoostClassifier()</font> provides support for both SAMME and SAMME.R**. I will not go into details how SAMME.R works. If you are interested, [Zhu et al. 2006](https://web.stanford.edu/~hastie/Papers/samme.pdf) is the original paper that describes the algorithm and can be a good read.

### Regression
**As a quick reminder, the key concepts of AdaBoost classification are**: 
* At each iteration, the weights of misclassified samples were increased, and the predictor focuses more on getting the misclassified samples right.
* The classifiers have more say in the final prediction if they have lower error rates.  

**AdaBoost regression follows the same concepts**. The algorithm goes as follows:
* Intialize weights $w_i=1/n$, $i=1,2,..., n$
* Initalize iteration number $m=0$.
* Repeat below while average loss $\bar{L}$ (defined below) is less than 0.5:
    * Increment iteration number $m = m +1$.
    * **Pick $n$ samples with replacement to form the training set for this iteration. Note that for each pick the probability that sample $x_i$ is chosen is $w_i$**.
    * Train a base regressor on the bootstrapped training set. The trained regressor predicts $\hat{y}^{(m)}_i$ on sample $x_i$.
    * Calculate the **loss for each training sample**: $L_i^{(m)} = L(y_i, \hat{y}^{(m)}_i) = L(|y_i-\hat{y}^{(m)}_i|)$. The default of **Sklean's <font color='green'>AdaBoostRegressor()</font>** is $L_i = |y_i-\hat{y}^{(m)}_i|/sup (|y_i-\hat{y}^{(m)}_i|)$, where $sup$ stands for supremum.
    * Calculate **average loss** of this regressor: $\bar{L}^{(m)}=\sum_{i=1}^{n} w_iL_i^{(m)}$.
    * Calculate **confidence** of this regressor: $\beta^{(m)}=\mathrm{log}((1-\bar{L}^{(m)})/\bar{L}^{(m)})$. The smaller the average loss, the more "confident" the regressor is. This is similar to the $\alpha^{(m)}$ in AdaBoost classification.
    * Update **sample weights**: $w_i \leftarrow w_i \cdot exp[\beta^{(m)} \cdot (L_i^{(m)}-1)]$, $i=1,2,...,n$. In classification, we multiply misclassified sample weights by $exp(\alpha^{(m)})$. Here in regression, the multiplication is $exp[\beta^{(m)} \cdot (L_i^{(m)}-1)]$. The larger the sample loss $L_i^{(m)}$, the more "misclassified" this sample is.
    * Re-normalize sample weights: $w_i \leftarrow w_i/\sum_{i=1}^n w_i$.
* We ended up with $M$ trained regressors that each improves on the previous.
* Output:
    * Given a test sample $x$, and $M$ trained regressors from the previous step, we obtain a prediction from each regressor: $\hat{y}^{(1)}$, $\hat{y}^{(2)}$, ..., $\hat{y}^{(M)}$.
    * The weighted median of $\hat{y}^{(1)}$, $\hat{y}^{(2)}$, ..., $\hat{y}^{(M)}$ given weights $\beta^{(1)}$, $\beta^{(2)}$, ..., $\beta^{(M)}$ is the final prediction. You can obtain the weighted median as follows:
        * Re-label the regressors so that $\hat{y}^{(1)} < \hat{y}^{(2)} < ... < \hat{y}^{(M)}$. Re-label the weights $\beta^{(1)}$, $\beta^{(2)}$, ..., $\beta^{(M)}$ accordingly.
        * Sum up $\beta$s until we reach the *smallest* $m$ that satisfies $\sum_{i=1}^{m}\beta^{(i)}\geq 1/2 \cdot \sum_{i=1}^{M}\beta^{(i)}$.
        * $\hat{y}^{(m)}$ is the final prediction of the ensemble.


Note that each iteration is more "difficult" than the previous, which means the subsequent regressor is harder to train than the previous. Therefore, the average loss $\bar{L}$ tends to increase with interations, and finally the algorihtm terminates when $\bar{L}$ exceeds the bound. The above algorithm is based on [Drucker 1997](https://pdfs.semanticscholar.org/8d49/e2dedb817f2c3330e74b63c5fc86d2399ce3.pdf). If you read the paper, you will notice that I have modified the definition of $\beta^{(m)}$ and $w_i$ to help with intuitive comparison with the classification algorithm. But really the algorithm describes here is the *same* as the one in the paper. 

**Sklearn's <font color='green'>AdaBoostRegressor()</font> uses this algorithm**.

## Sklearn functions and examples
Sklearn provides the following functions for AdaBoost:
* **AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0, loss=’linear’, random_state=None)**
* **AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm=’SAMME.R’, random_state=None)**

The **base_estimator** specifies the estimator trained in each iteration. The default base estimator for <font color='green'>AdaBoostRegressor</font> is <font color='green'>DecisionTreeRegressor(max_depth=3)</font>, and the default for <font color='green'>AdaBoostClassifier</font> is <font color='green'>DecisionTreeClassifier(max_depth=1)</font>. Note that the base trees are relatively shallow. This is because the trees act as weak learners, and we do not want individual weak learner to learn the data too well (overfitting). **These shallow trees might have high bias on their own, but the boosting technique makes sure the ensemble of the weak learners performs well**.

The **n_estimators** correspond to the num of iterations (thus number of estimators) $M$ in the above algorithms.

**learning_rate** can shrink the contribution of each base estimator. If we denote learning rate as $\eta$, then:
* In AdaBoost classification, $\alpha^{(m)}=\eta \cdot \mathrm{log}((1-err^{(m)})/err^{(m)})$
* In AdaBoost regression, $\beta^{(m)}=\eta \cdot \mathrm{log}((1-\bar{L}^{(m)})/\bar{L}^{(m)})$  

By default, learning rate is $1.0$.

### Example 7.1
In this example, we will try out Sklearn's <font color='green'>AdaBoostClassifier()</font>. Just like my previous tutorial, we will use the Red Wine Quality dataset. Let's load the data first:


```python
data = pd.read_csv('../input/winequality-red.csv')
data['category'] = data['quality'] >= 7 # again, binarize for classification
data.head()
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    Cell In[3], line 1
    ----> 1 data = pd.read_csv('../input/winequality-red.csv')
          2 data['category'] = data['quality'] >= 7 # again, binarize for classification
          3 data.head()
    

    File c:\users\82109\appdata\local\programs\python\python38\lib\site-packages\pandas\io\parsers\readers.py:912, in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
        899 kwds_defaults = _refine_defaults_read(
        900     dialect,
        901     delimiter,
       (...)
        908     dtype_backend=dtype_backend,
        909 )
        910 kwds.update(kwds_defaults)
    --> 912 return _read(filepath_or_buffer, kwds)
    

    File c:\users\82109\appdata\local\programs\python\python38\lib\site-packages\pandas\io\parsers\readers.py:577, in _read(filepath_or_buffer, kwds)
        574 _validate_names(kwds.get("names", None))
        576 # Create the parser.
    --> 577 parser = TextFileReader(filepath_or_buffer, **kwds)
        579 if chunksize or iterator:
        580     return parser
    

    File c:\users\82109\appdata\local\programs\python\python38\lib\site-packages\pandas\io\parsers\readers.py:1407, in TextFileReader.__init__(self, f, engine, **kwds)
       1404     self.options["has_index_names"] = kwds["has_index_names"]
       1406 self.handles: IOHandles | None = None
    -> 1407 self._engine = self._make_engine(f, self.engine)
    

    File c:\users\82109\appdata\local\programs\python\python38\lib\site-packages\pandas\io\parsers\readers.py:1661, in TextFileReader._make_engine(self, f, engine)
       1659     if "b" not in mode:
       1660         mode += "b"
    -> 1661 self.handles = get_handle(
       1662     f,
       1663     mode,
       1664     encoding=self.options.get("encoding", None),
       1665     compression=self.options.get("compression", None),
       1666     memory_map=self.options.get("memory_map", False),
       1667     is_text=is_text,
       1668     errors=self.options.get("encoding_errors", "strict"),
       1669     storage_options=self.options.get("storage_options", None),
       1670 )
       1671 assert self.handles is not None
       1672 f = self.handles.handle
    

    File c:\users\82109\appdata\local\programs\python\python38\lib\site-packages\pandas\io\common.py:859, in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
        854 elif isinstance(handle, str):
        855     # Check whether the filename is to be opened in binary mode.
        856     # Binary mode does not support 'encoding' and 'newline'.
        857     if ioargs.encoding and "b" not in ioargs.mode:
        858         # Encoding
    --> 859         handle = open(
        860             handle,
        861             ioargs.mode,
        862             encoding=ioargs.encoding,
        863             errors=errors,
        864             newline="",
        865         )
        866     else:
        867         # Binary mode
        868         handle = open(handle, ioargs.mode)
    

    FileNotFoundError: [Errno 2] No such file or directory: '../input/winequality-red.csv'


Then, split the dataset into a training set and a test set:


```python
X = data[data.columns[0:11]].values
y = data['category'].values.astype(np.int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42) 
# random_state is fixed to guarantee repeatable results for this tutorial - remove it in actual practice

print('X train size: ', X_train.shape)
print('X test size: ', X_test.shape)
print('y train size: ', y_train.shape)
print('y test size: ', y_test.shape)
```


```python
adaClf = AdaBoostClassifier(algorithm = 'SAMME', random_state=12)
adaClf.fit(X_train,y_train)
```

Let's see how this basic AdaBoost Classifier performs. In AdaBoost, the predicted class probabilities is the **weighted mean predicted class probabilities of the classifiers in the ensemble**.


```python
p_pred = adaClf.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, p_pred)

plt.subplots(figsize=(8,6))
plt.plot(fpr, tpr)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('False positive rate', fontsize = 16)
plt.ylabel('True positive rate', fontsize = 16)
plt.show()
```


```python
print('AUC is: ', auc(fpr, tpr))
```

The AUC of this AdaBoost classifier is acceptable, but really not the best. Let's taking a closer look at what the classifier is doing by printing the estimator weights and errors:


```python
print(adaClf.estimator_weights_)
```


```python
print(adaClf.estimator_errors_)
```

It looks like the errors increases and then fluctuates around 0.45, and as a result the estimators weights decreases and then fluctuates around 0.10. This is kind of anti-intuitive since one would think the next estimator improves on the previous one. Let's print out each estimators prediction and try to understand this observation. **To save space, we will print out the first 20 estimators and whether they predicted correctly on the first 10 training samples (out of 1119 samples)**.


```python
for estimator in adaClf.estimators_[:20]:
    y_pred = estimator.predict(X_train)
    correct = y_train==y_pred
    print(correct[:10]) # print only the first 10 training samples to save space
```

Now we can see that, **<font color='blue'>sometimes, when AdaBoost tries to correct the previously misclassified cases, it makes mistakes on the previously correct predictions</font>**. For example, in the above cell you can see that **the second iteration corrected the two incorrect predictions in the first iteration, but misclassified four previously correct samples**. This is probably a common phenomenon with depth-1 decision tree base learners, as the tree is set to make mistakes when it just splits the training set into two based on one feature and one threshold. The fortunate thing is, if error rate is high, the algorithm will assign a low estimator weight and the estimator will not affect the decision too much. 

**<font color='blue'>A bigger potential issue is, sometimes a training set contains outliers, noises, or in general just some samples that are really difficult to predict correctly. The weight of such samples will increase exponentially and the algorithm is forced to focus almost solely on these samples, leading to an overfit of the model to these samples. As a result, the model might have got these "difficult" examples right, but the previously correct predictions becomes incorrect</font>**. If you suspect such problem, it is a good idea to inspect the training samples to see if there are any outliers.

Why does this AdaBoost classifier performs less well as the Random Forest classifier in [tutorial 5](https://www.kaggle.com/fengdanye/machine-learning-5-random-forests/)? Here are my (unconfirmed) speculations:
* 50 decision trees is not enough for this AdaBoost classifier. The Random Forest classifier had 500.
* The learning rate is too high, which makes the weights of misclassified samples increase too rapidly. This makes the AdaBoost easily overfits to high-weight samples. Random Forest does not have this problem. 
* There are outliers/difficult samples in the training set that's getting all the attention from AdaBoost. Random Forest does not have this problem. Random Forest also has ways of introducing randomness for each tree to avoid overfitting.
* The SAMME.R algorithm performs better than SAMME (however, you won't be able to print meaningful estimator weights when using SAMME.R).

I also find it interesting to think that Random Forest introduces *randomness* into different trees with bagging and feature subsampling, whereas AdaBoost also introduces some types of variability between trees but do so by introducing *focus* on the previously misclassified cases and less accurate classifiers are penalized. 

### Example 7.2
In this example, we will try out Sklearn's <font color='green'>AdaBoostRegressor()</font>. The input will be the 11 features, the output will be the wine quality value.


```python
X = data[data.columns[0:11]].values
y = data['quality'].values.astype(np.int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

print('X train size: ', X_train.shape)
print('X test size: ', X_test.shape)
print('y train size: ', y_train.shape)
print('y test size: ', y_test.shape)
```

We will just try out the default AdaBoost Regressor:


```python
adaReg = AdaBoostRegressor(random_state=3)
adaReg.fit(X_train, y_train)
```

Then test its performance on the test set:


```python
y_pred = adaReg.predict(X_test)
plt.plot(y_test, y_pred, linestyle='', marker='o')
plt.xlabel('true values', fontsize = 16)
plt.ylabel('predicted values', fontsize = 16)
plt.show()
print('The r2_score on the test set is: ',r2_score(y_test, y_pred))
```

Note that this performance is better than the "500 linear regressors with Random Patches" model we used in the [last tutorial](https://www.kaggle.com/fengdanye/machine-learning-6-basic-ensemble-learning). However, in terms of r2_score, Random Forest and Extra Trees remain as better choices. You can try AdaBoostRegressor with 500 trees, and its performance will not be as good as Random Forest and Extra Trees. **It might just be that for this dataset, it is better to introduce randomness between estimators, than to introduce focus on the inaccurate predictions (e.g. there might be outliers in the dataset that's leading AdaBoost to the wrong direction)**. 

# Numerical optimization in function space
Gradient boosting ([Friedman 1999](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)) and XGBoost ([Chen and Guestrin 2016](https://arxiv.org/pdf/1603.02754.pdf)) came after AdaBoost. These techniques are among the most popular in today's data science. In this section, we will go over key methods of numerical optimiaztion in <font color='blue'> function </font> space, **which will greatly help you understand gradient boosting and XGBoost**.

This part of the tutorial is based on [Friedman 1999](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) and [Nielsen 2016](https://brage.bibsys.no/xmlui/bitstream/handle/11250/2433761/16128_FULLTEXT.pdf).

## Review - optimization in parameter space
Let's start with reviewing concepts of optimization in parameter space. Let's assume the data ($x$,$y$) we have are continuous, following a distribution $f(x,y)$. Define a parameterized model $F(x;\theta)$ that predict the response($\hat{y}$) of $x$. We can define the "cost" of the model as a function of $\theta$:
<center>
 $\Phi(\theta) = E_{x,y}[L(y, F(x; \theta))] \equiv \int_x \int_y L(y, F(x; \theta)) f(x,y)dx dy$    
</center>
Here $f(x,y)$ is the probability density function. We will determine the value of $\theta$ that minimizes the $\Phi(\theta)$ in a step-wise fashion:
<center>
 $\theta = \sum_{m=0}^{M} \theta_m$    
</center>
At iteration $m$, $\theta$ is updated as follows: $\theta^{(m)}=\theta^{(m-1)}+\theta_m$. $\theta_0$ is an initial guess. Below we discuss **two ways** of updating $\theta$.

### Gradient descent
At each iteration $m$, we take a step along the **negative gradient** of $\Phi(\theta)$ with respect to $\theta$. Such step is in the steepdest descent direction and is guaranteed to reduce the cost given a reasonable step length. Before the update at iteration $m$, the negative gradient is:
<center>
    $-g_m = -\frac{\partial \Phi(\theta)}{\partial \theta} \mid_{\theta = \theta^{(m-1)}}$
</center>
The step length $\rho_m$ are usually determined through <font color='blue'>line search</font>:
<center>
    $\rho_m =  \underset{\rho}{\operatorname{argmin}} \Phi(\theta^{(m-1)}-\rho g_m)$
</center>
The step taken at iteration $m$ is therefore:
<center>
    $\theta_m = -\rho_m g_m$
</center>
Gradient descent is a first-order method, and it only requires the $\Phi(\theta)$ to be differentiable.


### Newton's method
Newton's method can be understood as follows:  
Before the update at iteration $m$, we have $\theta = \theta^{(m-1)}$. The *ideal* update of $\theta$ should satisfy:
<center>
    $\frac{\partial \Phi(\theta)}{\partial \theta} \mid _{\theta = \theta^{(m-1)}+\theta_m} = 0$
</center>
That is, the update directly take $\Phi(\theta)$ to its minima. Note that
<center>
    $\frac{\partial \Phi(\theta)}{\partial \theta} \mid _{\theta = \theta^{(m-1)}+\theta_m} = \frac{\partial \Phi(\theta^{(m-1)}+\theta_m)}{\partial (\theta^{(m-1)}+\theta_m)} = \frac{\partial \Phi(\theta^{(m-1)}+\theta_m)}{\partial (\theta_m)}$
</center>
Therefore, for each step we want to solve:
<center>
    $\frac{\partial \Phi(\theta^{(m-1)}+\theta_m)}{\partial (\theta_m)} = 0$
</center>

We can solve the above equation *approximately* by expanding $\Phi(\theta^{(m-1)}+\theta_m)$ using a second-order [Taylor expansion](https://en.wikipedia.org/wiki/Taylor_series):
<center>
    $\Phi(\theta^{(m-1)}+\theta_m) \approx \Phi(\theta^{(m-1)})+  \frac{\partial \Phi(\theta)}{\partial \theta}\mid_{\theta=\theta^{(m-1)}} \theta_m + \frac{1}{2!} \frac{\partial^2 \Phi(\theta)}{\partial \theta ^2}\mid_{\theta=\theta^{(m-1)}}\theta_m^2$
</center>
Let's denote $g_m = \frac{\partial \Phi(\theta)}{\partial \theta}\mid_{\theta=\theta^{(m-1)}}$, and $h_m = \frac{\partial^2 \Phi(\theta)}{\partial \theta ^2}\mid_{\theta=\theta^{(m-1)}}$. Then:
<center>
    $\frac{\partial \Phi(\theta^{(m-1)}+\theta_m)}{\partial (\theta_m)} \approx g_m + h_m\theta_m= 0$
</center>
Therefore the update at iteration $m$ is:
<center>
    $\theta_m = - g_m/h_m$
</center>
Newton's method is a second-order method and requires $\Phi(\theta)$ to be twice differentiable.

## Numerical optimization in function space
In the function space, **we regard $F(x)$ predicted at each $x$ a "parameter" to optimize**. Again, we have:
<center>
 $\Phi(F) = E_{x,y}[L(y, F(x))] \equiv \int_x \int_y L(y, F(x)) f(x,y)dx dy = \int_x \left[ \int_y L(y, F(x)) f(y|x)dy \right]f(x)dx \equiv E_{x}\left[E_y \left(L(y, F(x))\mid x \right)\right]$    
</center>
Therefore, minimizing $\Phi(F)$ is equivalent to minimizing:
<center>
 $\phi(F) = E_y \left[L(y, F(x)) \mid x\right]$  
</center>
**at each $x$**.  

Similar to optimization in parameter space, we define:
<center>
 $F(x)=\sum_{m=0}^{M}f_m(x)$  
</center>
and
<center>
 $F^{(m)}(x) = F^{(m-1)}(x)+f_m(x)$  
</center>
**at each $x$**.  Here, $f_0(x)$ is an initial guess.

### Gradient descent
Now, the "parameter" becomes the $F(x)$ for each $x$. The gradient of $\phi(F)$ with regard to $F$ is:
<center>
 $g_m(x) = \left[ \frac{\partial \phi(F(x))}{\partial F(x)} \right]_{F(x)=F^{(m-1)}(x)}= \left[ \frac{\partial E_y \left[L(y, F(x)) \mid x\right]}{\partial F(x)} \right]_{F(x)=F^{(m-1)}(x)} =  E_y \left[ \frac{\partial L(y, F(x))}{\partial F(x)}\mid x  \right]_{F(x)=F^{(m-1)}(x)}$  
</center>
The interchange between differentiation and intergration is based on the assumption that sufficient regularity is present. Loss functions usually satisfy this requirement, so you don't need to worry to much about the exceptions.

At each iteration $m$, the update is:
<center>
$f_m(x) = -\rho_m g_m(x)$. 
</center>
The step length $\rho_m$ is given by line search:
<center>
$\rho_m = \underset{\rho}{\operatorname{argmin}} E_{x,y} \left[ L(y, F^{(m-1)}(x)-\rho g_m(x)) \right]$
</center>

### Newton's method
Similar to what we did in parameter space, at each iteration $m$ and each $x$, we want to solve for:
<center>
    $\frac{\partial \phi \left(F^{(m-1)}(x)+f_m(x)\right)}{\partial (f_m(x))} = 0$
</center>
That is,
<center>
    $\frac{\partial}{\partial (f_m(x))} E_y \left[ L(y, F^{(m-1)}(x) + f_m(x)) \mid x \right]= 0$
</center>

Note that
<center>
    $ E_y \left[ L(y, F^{(m-1)}(x) + f_m(x)) \mid x \right] = \int_y L(y, F^{(m-1)}(x) + f_m(x)) f(y \mid x) dy \approx  \int_y \left[ L(y, F^{(m-1)}(x)) + \left[\frac{\partial L(y, F(x))}{\partial F(x)}\right]_{F(x)=F^{(m-1)}(x)}f_m(x) + \frac{1}{2} \left[\frac{\partial ^2 L(y, F(x))}{\partial F(x)^2}\right]_{F(x)=F^{(m-1)}(x)}f_m^2(x) \right] f(y \mid x) dy$
</center>
Define
<center>
    $g_m(x) = \int_y  \left[\frac{\partial L(y, F(x))}{\partial F(x)}\right]_{F(x)=F^{(m-1)}(x)}f(y \mid x) dy \equiv E_y \left[ \frac{\partial L(y, F(x))}{\partial F(x)} \mid x\right]_{F(x)=F^{(m-1)}(x)}$
</center>
and
<center>
    $h_m(x) = \int_y   \left[\frac{\partial ^2 L(y, F(x))}{\partial F(x)^2} \right]_{F(x)=F^{(m-1)}(x)}f(y \mid x) dy \equiv E_y \left[ \frac{\partial^2 L(y, F(x))}{\partial F(x)^2} \mid x\right]_{F(x)=F^{(m-1)}(x)}$
</center>
Then we have:
<center>
    $E_y \left[ L(y, F^{(m-1)}(x) + f_m(x)) \mid x \right] \approx E_y \left[ L(y, F^{(m-1)}(x)) \mid x \right] + g_m(x)f_m(x) + \frac{1}{2}h_m(x)f_m(x)^2$
</center>
Therefore the solution to $\frac{\partial}{\partial (f_m(x))} E_y \left[ L(y, F^{(m-1)}(x) + f_m(x)) \mid x \right]= 0$ is approximated by
<center>
    $f_m(x) = - g_m(x)/h_m(x)$
</center>:

## Finite data
As a quick recap, the numerical optimization in function space is expressed as
<center>
 $F(x)=\sum_{m=0}^{M}f_m(x) = f_0(x) + \sum_{m=1}^{M}f_m(x)$  
</center>
However, the above disscusion of optimization is all based on the assumption that we have continuous data $(x,y)$ with distribution $f(x,y)$. **In real data, we only have finite samples $\{ x_i, y_i \}$, $i=1,2,..,n$**. For any $x$ value outside of $\{ x_i\}$, $i=1,2,..,n$, we won't be able to directly update $F(x)$ using $ -\rho_m g_m(x)$ or $-g_m(x)/h_m(x)$ (because $E_y \left[ ...\mid x \right]$ cannot be estimated at $x$ values outside of training sample points). **Some underlying assumption of the model has to be made for the optimization to work**. Generall, we can assume a parameterized form of $f_m(x)$:
<center>
 $F(x)=\sum_{m=0}^{M}f_m(x) = f_0(x) + \sum_{m=1}^{M}\theta_m \phi(x;a_m)$
</center>
Under the context of boosting, $\phi(x;a_m)$ is a class of weak learners parameterized by $a_m$, and $\theta_m$ is the corresponding coefficient in front of the weak learner. The assumption of the class of the weak learners will constrain $F(x)$ in a certain space. If the weak learn is chosen to be decision trees, then $a_m$ describes the structure of the tree, as well as the weights $w_j$ of leaf node $j$. See [tutorial 5](https://www.kaggle.com/fengdanye/machine-learning-5-random-forests/) Appendix for more information on the decision trees. **<font color='blue'>Boosting algorithm can be regarded as a problem of optimization in function space, with the steps constrained as a certain class of weak learners</font>**. The objective function is:
<center>
    $\hat{\Phi}(F) = \sum_{i=1}^n L(y_i, F(x_i))$
</center>
Or equivalently:
<center>
    $\hat{\phi}(F(x_i)) = L(y_i, F(x_i))$
</center>
for each $x_i$, $i=1,2,...,n$.

###  Foward stage-wise additive modeling (FSAM)
Most boosting algorithm is solved in a state-wise fashion. At each iteration $m$, we solve:
<center>
$\{\theta_m, a_m\} = \underset{\{\theta, a\}}{\operatorname{argmin}} \sum_{i=1}^n L(y_i, F^{(m-1)}(x_i)+\theta \phi(x_i; a))$
</center>
given the weak learner class. This method is called **foward stage-wise additive modeling, or FSAM**. <font color='blue'>It is proven that the AdaBoost algorithm is equivalent to solving the above equation exactly for the exponential loss function $L(y, F) = exp(-yF)$</font> ([Friedman et al. 2000](https://web.stanford.edu/~hastie/Papers/AdditiveLogisticRegression/alr.pdf)) <font color='blue'> under the constraint that $\phi$ are classifiers with output $-1$ or $1$. On the other hand, gradient boosting and Newton boosting solve the above equition approximately through gradient descent and Newthon's method in the function space, respectively. For gradient boosting and Newton boosting, various loss functions can be used, as long as they are differentiable (gradient boosting) or twice differentiable (Newton boosting)</font>. 

# Gradient boosting machine (GBM)
## General theory
Gradient boosting was proposed by [Friedman 1999](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf). **The method corrsponds to gradient descent in the function space**. Given a finite training set $\{ x_i, y_i \}$, $i=1,2,..,n$, the gradient of objective function with respect to $F$ at each sample point and at iteration $m$ is:
<center>
    $g_m(x_i) = \left[ \frac{\partial \hat{\phi}(F(x_i)) }{\partial F(x_i)} \right]_{F(x_i)=F^{(m-1)}(x_i)} = \left[ \frac{\partial L(y_i, F(x_i)) }{\partial F(x_i)} \right]_{F(x_i)=F^{(m-1)}(x_i)}$
</center>
where $i=1,2,...,n$. Recall that in forward stage-wise additive modeling, we want to find the step in the function space $f_m(x_i) = \theta_m \phi(x_i;a_m)$ that minize the objective function:
<center>
$\{\theta_m, a_m\} = \underset{\{\theta, a\}}{\operatorname{argmin}} \sum_{i=1}^n L(y_i, F^{(m-1)}(x_i)+\theta \phi(x_i; a))$
</center>
With the method of gradient descent in function space, **we would want the direction of the vector $\{ \phi(x_i; a_m) \}_{i=1}^n$ to be as aligned with the negative gradient $\{-g_m(x_i)\}_{i=1}^n$ as possible**. Of course, ideally we would want $\{ \phi(x_i; a_m) \}_{i=1}^n$ to be in the exactly same direction as $\{-g_m(x_i)\}_{i=1}^n$. But since $\phi(x_i; a_m)$ are constrained by a class of weak learners, we can only find the parameters $a_m$ that best align  $\{ \phi(x_i; a_m) \}_{i=1}^n$ with the negative gradient. This is obtained by:
<center>
$a_m =  \underset{\{\beta, a\}}{\operatorname{argmin}} \sum_{i=1}^{n} \left[ \left( -g_m(x_i) \right) - \beta \phi(x_i;a) \right] ^2$
</center>
The minimization step can be regarded as finding a constrained step direction for gradient descent. The step leagth $\rho_m$ can be obtained through *line search*:
<center>
$\rho_m =  \underset{\rho}{\operatorname{argmin}} \sum_{i=1}^n L(y_i, F^{(m-1)}(x_i)+\rho \phi(x_i; a_m))$
</center>
In [Friedman 1999](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf), the final step is expressed as:
<center>
$f_m(x) =  \eta \rho_m \phi(x;a_m)$
</center>
where $0 < \eta \leq 1$ is the **learning rate**. The learning rate can shrink the step length at each iteration, introducing regularization to the algorithm.

### Algorithm
The general gradient boosting algorithm is as follows:
* Initialize $F^{(0)}(x) = f_0(x) = \underset{\theta}{\operatorname{argmin}}\sum_{i=1}^{n}L(y_i, \theta)$
* For $m=1,2,...,M$, do:
    * $g_m(x_i) = \left[ \frac{\partial L(y_i, F(x_i)) }{\partial F(x_i)} \right]_{F(x_i)=F^{(m-1)}(x_i)}$, $i=1,2,...,n$
    * $a_m =  \underset{\{\beta, a\}}{\operatorname{argmin}} \sum_{i=1}^{n} \left[ \left( -g_m(x_i) \right) - \beta \phi(x_i;a) \right] ^2$
    * $\rho_m =  \underset{\rho}{\operatorname{argmin}} \sum_{i=1}^n L(y_i, F^{(m-1)}(x_i)+\rho \phi(x_i; a_m))$
    * $f_m(x) =  \eta \rho_m \phi(x;a_m)$
    * $F^{(m)}(x) = F^{(m-1)}(x) + f_m(x)$
* To predict:
    * Given $x$, predict $F(x) = F^{(M)}(x) = \sum_{m=0}^{M}f_m(x)$.

## Gradient tree boosting
### Introduction
**Gradient tree boosting is a specific case of gradient boosing, with decision trees as base learners**. It is very commonly used in current practice and it usually presents satisfying performance. Before proceeding, you should read the Appendix in [tutorial 5](https://www.kaggle.com/fengdanye/machine-learning-5-random-forests), if you haven't done so. Recall that in the Appendix of  [tutorial 5](https://www.kaggle.com/fengdanye/machine-learning-5-random-forests), we learned that a decision tree can be expressed as:
<center>
$f(x) = \sum_{j=1}^{T} w_j {\mathrm I}(x\in R_j)$
</center>
where $w_j$ is called the **weight** of *j*th leaf node, and $\{R_j\} ( j=1,2,...,T)$ is called the **structure** of the tree. The $T$ is the number of leaf nodes of this tree. The ${\mathrm I}(x\in R_j)$ equals one if sample $x$ belongs to area $R_j$, and zero otherwise. With decision trees as the base learners, the additive model of $F(x)$ can be expressed as:
<center>
 $F(x)=f_0(x) + \sum_{m=1}^{M}f_m(x) = f_0(x) + \sum_{m=1}^{M} \sum_{j=1}^{T_m}w_{jm}{\mathrm I}(x\in R_{jm})$  
</center>

**Just like what we did in the general GBM algorithm, at each iteration $m$, we want to align the base learner trees with the negative gradients. The cost of such alignment is again the sum of squared loss**:
<center>
$J_m = \sum_{i=1}^{n} \left[ \left(-g_m(x_i)\right) - \left( \sum_{j=1}^{T_m} w_{jm}{\mathrm I}(x\in R_{jm}) \right) \right]^2 = const +  \sum_{i=1}^{n} \left[ 2g_m(x_i) \sum_{j=1}^{T_m} w_{jm}{\mathrm I}(x\in R_{jm}) +  \sum_{j=1}^{T_m} w_{jm}^2{\mathrm I}(x\in R_{jm}) \right]$
</center>
The above sum over samples can be substituted by a sum over the leaf nodes:
<center>
$J_m = const +  \sum_{j=1}^{T_m} \sum_{i; \ x_i \in R_{jm}}\left[ 2g_m(x_i) w_{jm} +w_{jm}^2 \right] = const +  \sum_{j=1}^{T_m} \left[ 2 \left( \sum_{i; \ x_i \in R_{jm}} g_m(x_i) \right) w_{jm} +  \sum_{i; \ x_i \in R_{jm}}w_{jm}^2\right]$
</center>
If we define $G_{jm} = \sum_{i; \ x_i \in R_jm} g_m(x_i)$, define the number of samples belonging to region $R_{jm}$ (leaf node $j$) as $n_{jm}$, discard the constant in $J_m$, and divide $J_m$ by 2 (none of these will change the optimization result), the cost can be expressed as:
<center>
$J_m =  \sum_{j=1}^{T_m} \left[  G_{jm} w_{jm} +  \frac{1}{2}n_{jm}w_{jm}^2\right]$
</center>
This cost function will be used to train a decision tree.

To train a tree, we follow the CART algorithm introduced in the Appendix of [tutorial 5](https://www.kaggle.com/fengdanye/machine-learning-5-random-forests). **The algorithm first decides the weights given a fixed structure, and then learn the structure given the weights**. With fixed structure, $G_{jm}$ and $n_{jm}$ are fixed, the weights are thus given by the values that minimizes the cost:
<center>
    $w_{jm}^{*} = -\frac{G_{jm}}{n_{jm}}$
</center>
where $j=1,2,...,T_m$.

Plugging the weights back to the cost function gives us:
<center>
    $J_m^* = -\frac{1}{2} \sum_{j=1}^{T_m} \frac{G_{jm}^2}{n_{jm}}$
</center>
Consequently, when considering possible splits during tree training, the gain of a potential split is:
<center>
${\mathrm {Gain}}=J_{\mathrm {before}} - J_{\mathrm {after}} = L_k^* - (L_L^*+L_R^*) = \frac{1}{2} \left[ \frac{G_L^2}{n_L} + \frac{G_R^2}{n_R} - \frac{G_{km}^2}{n_{km}} \right]$ 
</center>
where $k$ is the node that is being split.

After the above steps, a tree has been trained to be the base learner for iteration $m$. Recall that in the general GBM algorithm, we have a line search step where we search for the best step length $\rho_m$. **For the case of gradient *tree* boosting, [Friedman 1999](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) proposed that instead of doing one general line search, we can do $T_m$ line search steps, one for each leaf node. This is equivalent to a final update on the weights $w_{jm}$**:
<center>
$\{ w_{jm} \}_{j=1}^{T_m} =  \underset{\{ w_{j} \}_{j=1}^{T_m}}{\operatorname{argmin}} \sum_{i=1}^{n} L\left(y_i, F^{(m-1)}(x_i) + \sum_{j=1}^{T_m}w_j {\mathrm I}(x_i\in R_{jm})\right)$
</center>
<font color='blue'>Keep in mind that the line search step is minimizing the prediction cost at $m$th iteration, and it is separate from the cost of fitting the decision tree to the negative gradients </font>. The above minimization problem can be divided into $T$ independent minimization problems:
<center>
$w_{jm} =  \underset{w_{j}}{\operatorname{argmin}} \sum_{i;\ x_i \in R_{jm}} L\left(y_i, F^{(m-1)}(x_i) + w_j\right)$
</center>
for $j=1,2,...,T_m$.

The alogrithm of gradient tree boosting is summarize below.

### Algorithm
* Initialize $F^{(0)}(x) = f_0(x) = \underset{\theta}{\operatorname{argmin}}\sum_{i=1}^{n}L(y_i, \theta)$
* For $m=1,2,...,M$, do:
    * $g_m(x_i) = \left[ \frac{\partial L(y_i, F(x_i)) }{\partial F(x_i)} \right]_{F(x_i)=F^{(m-1)}(x_i)}$, $i=1,2,...,n$
    * Train decision trees to align with the negative gradients. Determine the structure of the tree $\{R_{jm}\}_{j=1}^{T_m}$ by selecting splits that maximizes the gain $\frac{1}{2} \left[ \frac{G_L^2}{n_L} + \frac{G_R^2}{n_R} - \frac{G_{km}^2}{n_{km}} \right]$.
    * Determine the leaf weights of the learned structure: $w_{jm} =  \underset{w_{j}}{\operatorname{argmin}} \sum_{i;\ x_i \in R_{jm}} L\left(y_i, F^{(m-1)}(x_i) + w_j\right)$, where $j=1,2,...,T_m$.
    * $f_m(x) = \eta \sum_{j=1}^{T_m}w_{jm} {\mathrm I}(x\in R_{jm})$, where $\eta$ is the learning rate.
    * $F^{(m)}(x) = F^{(m-1)}(x) + f_m(x)$
* To predict:
    * Given $x$, predict $F(x) = F^{(M)}(x) = \sum_{m=0}^{M}f_m(x)$.

### Regularization
There are two common ways to regularize gradient tree boosing models:
1. **Constrain the trees**. For example, you can set maximum number of leaf nodes allowed. You can also limit the total number of trees in the ensemble $M$.
2. **Random subsampling**.  Similar to the bagging method introduced in the [last tutorial](https://www.kaggle.com/fengdanye/machine-learning-6-basic-ensemble-learning), we can randomly subsample from the training set to train each tree. That is, different subsamples of the training set are used to train different trees in the ensemble. The difference between bagging and the random subsampling here is that here we sample *without* replacement. In Sklearn's gradient boosting functions, the subsampling fraction is the parameter called "subsample". When the subsampling fraction is smaller than $1.0$, the model is called <font color='blue'>stochastic gradient boosting</font>.

### Sklearn functions and examples
Sklearn provides the following functions for gradient tree boosting:
* **GradientBoostingClassifier()**
* **GradientBoostingRegressor()**

One very important parameter for the functions is the definition of the **loss function** $L(y, F)$ (parameter "loss" in sklearn). In fact, **the algorithm for gradient boosting classification and regression only differ in the loss function they use**. The Sklearn [documentation](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting) has a full list of loss functions for classification and regression in section "1.11.4.5.1. Loss Functions". The default loss function for <font color='green'>GradientBoostingClassifier()</font> is the negative binomial log-likelihood loss function ("deviance") for binary classification. The default loss function for <font color='green'>GradientBoostingRegressor()</font> is the squared loss ("ls"). Note that if the loss function is set to the exponential loss ("exponential"), the gradient boosting recovers the AdaBoost algorithm.

The **learning_rate** parameter corresponds to the $\eta$ in the above algorithms. The **n_estimator** is the number of trees in the ensemble and corresponds to the $M$ in the above algorithms. The **subsample** is the subsampling ratio for stochastic gradient boosting. Most of the other parameters, such as max_features and max_leaf_nodes, are for each individual tree. These tree parameters were introduced in [tutorial 5](https://www.kaggle.com/fengdanye/machine-learning-5-random-forests/).

#### Example 7.3
Now let's try out <font color='green'>GradientBoostingClassifier()</font> on the Red Wine Quality dataset:


```python
data = pd.read_csv('../input/winequality-red.csv')
data['category'] = data['quality'] >= 7 # again, binarize for classification

X = data[data.columns[0:11]].values
y = data['category'].values.astype(np.int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42) 
# random_state is fixed to guarantee repeatable results for this tutorial - remove it in actual practice

print('X train size: ', X_train.shape)
print('X test size: ', X_test.shape)
print('y train size: ', y_train.shape)
print('y test size: ', y_test.shape)
```


```python
clf = GradientBoostingClassifier(random_state = 27)
clf.fit(X_train, y_train)
```


```python
p_pred = clf.predict_proba(X_test)[:,1]
fpr,tpr,thresholds = roc_curve(y_test, p_pred)

plt.subplots(figsize=(8,6))
plt.plot(fpr, tpr)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('False positive rate', fontsize = 16)
plt.ylabel('True positive rate', fontsize = 16)
plt.show()
```


```python
print("AUC is: ", auc(fpr, tpr))
```

The default parameters of <font color='green'>GradientBoostingClassifier</font> produced an AUC of 0.90. This is pretty good! Now let's see if we can use <font color='green'>GridSearchCV</font> (introduced in [tutorial 4](https://www.kaggle.com/fengdanye/machine-learning-4-support-vector-machine)) to make the model even better:


```python
tuned_parameters = {'learning_rate':[0.05,0.1,0.5,1.0], 'subsample':[0.4,0.6,0.8,1.0]}

clf = GridSearchCV(GradientBoostingClassifier(random_state = 27), tuned_parameters, cv=5, scoring = 'roc_auc')
clf.fit(X_train, y_train)
```


```python
print('The best model is: ', clf.best_params_)
```

In our past tutorials, I would stop here and look at the best model's performance. But really the above grid search is just a coarse one. **You can always do a finer search around the best model found in the coarse search**:


```python
tuned_parameters = {'learning_rate':[0.09,0.1,0.11], 'subsample':[0.7,0.75,0.8,0.85,0.9]}

clf = GridSearchCV(GradientBoostingClassifier(random_state = 27), tuned_parameters, cv=5, scoring = 'roc_auc')
clf.fit(X_train, y_train)
```


```python
print('The best model is: ', clf.best_params_)
```

You can do another finer grid search after this if you'd like, but I will stop here. Let's check out the model's performance in terms of ROC curve and AUC value:


```python
p_pred = clf.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, p_pred)

plt.subplots(figsize=(8,6))
plt.plot(fpr, tpr)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('False positive rate', fontsize = 16)
plt.ylabel('True positive rate', fontsize = 16)
plt.show()
```


```python
print("AUC is: ", auc(fpr, tpr))
```

Recall that in [tutorial 5](https://www.kaggle.com/fengdanye/machine-learning-5-random-forests), we obtained an AUC of 0.909 on the same dataset using a 500-tree Random Forest with GridSearchCV. Taking into the consideration the fluctuation caused by *random_state*, the AUC of the gradient tree boosting model (0.904) can be considered as good as the AUC of the 500-tree Random Forest (0.909). If you want to be strict about the comparison, you can run the classifier many times to obtain a vector of AUCs for gradient tree boosting and Random Forest, respectively, and run a t-test to decide the significance of the difference between the mean AUCs of the two classifiers.

**The take-away is, with only 100 trees in the ensemble, the gradient boosting model performs as well as a 500-tree Random Forest. This shows the power of boosting - each tree is designed to reduce cost based on the performance from the previous iteration, instead of repeating the same training as the previous trees**. Feel free to try a GridSearchCV with more trees in the gradient tree boosting ensemble - you should be able to see that the gradient boosting model outperforms the Random Forest model.

#### Example 7.4
Now let's try out <font color='green'>GradientBoostingRegressor()</font>. The goal is to predict wine quality given the 11 wine features.


```python
X = data[data.columns[0:11]].values
y = data['quality'].values.astype(np.int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

print('X train size: ', X_train.shape)
print('X test size: ', X_test.shape)
print('y train size: ', y_train.shape)
print('y test size: ', y_test.shape)
```


```python
reg = GradientBoostingRegressor(random_state = 5)
reg.fit(X_train,y_train)
```


```python
y_pred = reg.predict(X_test)
plt.plot(y_test, y_pred, linestyle='', marker='o')
plt.xlabel('true values', fontsize = 16)
plt.ylabel('predicted values', fontsize = 16)
plt.show()
print('The r2_score on the test set is: ',r2_score(y_test, y_pred))
```

The r2_score improves slightly compared to the defualt <font color='green'>AdaBoostRegressor()</font>. Feel free to play with the hyperparameters to see if you can make the gradient boosting model better. Hint: use <font color='green'>GridSearchCV</font> .

# Newton Boosting
## General theory
**Newton boosting corresponds to Newton's optimization method in function space**. XGBoost belongs to this class of boosting, although XGBoost was actually created before Newton boosting was proposed in [Nielsen 2016](https://brage.bibsys.no/xmlui/bitstream/handle/11250/2433761/16128_FULLTEXT.pdf). Introducing XGBoost through Newton Boosting makes a lot of sense to me, so in this tutorialI will introduce Newton boosting first, then XGBoost.

Recall that given a finite training set $\{ x_i, y_i \}$, $i=1,2,..,n$, the objective function is:
<center>
    $\hat{\Phi}(F) = \sum_{i=1}^n L(y_i, F(x_i))$
</center>
and $F(x)$ is estimated through additive modeling:
<center>
 $F(x)=\sum_{m=0}^{M}f_m(x) = f_0(x) + \sum_{m=1}^{M}f_m(x)$  
</center>
At each iteration $m$, let's denote the "newton" step as $\phi(x;a_m)$. Recall that this is a weak learner parameterized by $a_m$. Later you will see that $f_m(x)=\eta \phi(x;a_m)$. We want this step to minimize the following objective function:
<center>
$\sum_{i=1}^{n} L(y_i, F^{(m-1)}(x_i) + \phi(x_i;a_m)) \approx \sum_{i=1}^n \left[ L(y_i, F^{(m-1)}(x_i)) + \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\mid _{F(x_i)=F^{(m-1)}(x_i)}\phi(x_i;a_m) + \frac{1}{2} \frac{\partial ^2 L(y_i, F(x_i))}{\partial F(x_i)^2} \mid _{F(x_i)=F^{(m-1)}(x_i)} \phi^2(x_i;a_m) \right]$
</center>
The second step is an Taylor expansion, just like what we did in the "Numerical optimization in function space" section.  

Define
<center>
    $g_m(x_i) = \left[ \frac{\partial L(y_i, F(x_i)) }{\partial F(x_i)} \right]_{F(x_i)=F^{(m-1)}(x_i)}$
</center>
and
<center>
    $h_m(x_i) = \left[ \frac{\partial^2 L(y_i, F(x_i)) }{\partial F(x_i)^2} \right]_{F(x_i)=F^{(m-1)}(x_i)}$
</center>
Then $a_m$ can be evaluated as follows:
<center>
    $a_m = \underset{a}{\operatorname{argmin}} \sum_{i=1}^n \left[ g_m(x_i) \phi(x_i;a) + \frac{1}{2}h_m(x_i) \phi^2(x_i;a) \right]$
</center>
We can complete the square by adding a constant:
<center>
    $a_m = \underset{a}{\operatorname{argmin}} \sum_{i=1}^n  \frac{1}{2}h_m(x_i) \left[ \left( -\frac{g_m(x_i)}{h_m(x_i)} \right) - \phi(x_i; a) \right]^2$
</center>
Therefore solving the weak learner is a problem of solving a weighted least-squares regression problem.

$f_m(x)$ is then defineds as $f_m(x)=\eta \phi(x;a_m)$, where $\eta$ is the learning rate (shrinkage).

### Algorithm
* Initialize $F^{(0)}(x) = f_0(x) = \underset{\theta}{\operatorname{argmin}}\sum_{i=1}^{n}L(y_i, \theta)$
* For $m=1,2,...,M$, do:
    * $g_m(x_i) = \left[ \frac{\partial L(y_i, F(x_i)) }{\partial F(x_i)} \right]_{F(x_i)=F^{(m-1)}(x_i)}$, $i=1,2,...,n$
    * $h_m(x_i) = \left[ \frac{\partial^2 L(y_i, F(x_i)) }{\partial F(x_i)^2} \right]_{F(x_i)=F^{(m-1)}(x_i)}$, $i=1,2,...,n$
    * $a_m =   \underset{a}{\operatorname{argmin}} \sum_{i=1}^n  \frac{1}{2}h_m(x_i) \left[ \left( -\frac{g_m(x_i)}{h_m(x_i)} \right) - \phi(x_i; a) \right]^2$
    * $f_m(x) =  \eta \phi(x;a_m)$
    * $F^{(m)}(x) = F^{(m-1)}(x) + f_m(x)$
* To predict:
    * Given $x$, predict $F(x) = F^{(M)}(x) = \sum_{m=0}^{M}f_m(x)$.

## XGBoost
### Introduction
**XGBoost can be considered as a subclass of Newton tree boosting**. From the previous section, we have shown that at each iteration $m$, we want to minimize the following cost function:
<center>
    $J_m = \sum_{i=1}^n \left[ g_m(x_i)\phi(x_i; a_m) + \frac{1}{2} h_m(x_i) \phi ^2(x_i;a_m) \right]$
</center>
where $g_m(x_i)$ and $h_m(x_i)$ were defined in the previous section. **For Newton tree boosting, we can substitute $\phi(x_i;a_m) = \sum_{j=1}^{T_m}w_{jm}{\mathrm I}(x_i\in R_{jm})$ into $J_m$**:
<center>
    $J_m = \sum_{i=1}^n \left[ g_m(x_i)\sum_{j=1}^{T_m}w_{jm}{\mathrm I}(x_i\in R_{jm}) + \frac{1}{2} h_m(x_i) \left( \sum_{j=1}^{T_m}w_{jm}{\mathrm I}(x_i\in R_{jm}) \right)^2 \right]= \sum_{i=1}^n \left[ g_m(x_i)\sum_{j=1}^{T_m}w_{jm}{\mathrm I}(x_i\in R_{jm}) + \frac{1}{2} h_m(x_i) \sum_{j=1}^{T_m}w_{jm}^2{\mathrm I}(x_i\in R_{jm})  \right]$
</center>
Due to the disjoint nature of ${R_{jm}}$, we can express $J_m$ as:
<center>
    $J_m = \sum_{j=1}^{T_m} \sum_{i; \ x_i \in R_{jm}} \left[ g_m(x_i)w_{jm} + \frac{1}{2}h_m(x_i)w_{jm}^2 \right]$
</center>

Now, define
<center>
 $G_{jm} = \sum_{i; \ x_i \in R_jm} g_m(x_i)$
</center>
and
<center>
 $H_{jm} = \sum_{i; \ x_i \in R_jm} h_m(x_i)$
</center>
then
<center>
    $J_m = \sum_{j=1}^{T_m} \left[ G_{jm}w_{jm} + \frac{1}{2}H_{jm}w_{jm}^2 \right]$
</center>


Just like what we did in the gradient tree boosting section, the trees will be trained in the following steps:
* Learn the weights for a given structure
* Learn the structure
* Learn the final weights

Gicen a fixed structure of the decision tree, the weights are given by (the weights should minize $J_m$ for a given tree structure):
<center>
    $w_{jm}^* = -\frac{G_{jm}}{H_{jm}}$
</center>
for $j=1,2,...,T_m$. 

Plugging the weights back to the cost function gives us:
<center>
    $J_m^* = -\frac{1}{2} \sum_{j=1}^{T_m} \frac{G_{jm}^2}{H_{jm}} = \sum_{j=1}^{T_m} L_j^*$
</center>
Consequently, when considering possible splits during tree training, the gain of a potential split is:
<center>
${\mathrm {Gain}}=J_{\mathrm {before}} - J_{\mathrm {after}} = L_k^* - (L_L^*+L_R^*) = \frac{1}{2} \left[ \frac{G_L^2}{H_L} + \frac{G_R^2}{H_R} - \frac{G_{km}^2}{H_{km}} \right]$ 
</center>
where $k$ is the node that is being split.

For Newton boosting, there is no line search after a tree strucutre is determined. Therefore $w_{jm}^*$ are the final weights.


### Regularization
Just like in the gradient tree boosing model, **contraining the individual trees and random subsampling** can be used to regularize a Newton tree boosting model. However, other regularization techniques also exist. Specifically, <font color='blue'>XGBoost</font> implements the following regularization techniques:
* **Random subspace method**: a random subsample of features (sampled *without* replacement) is used to train the tree at each iteration. The parameter that controls this process is sometimes called "column subsampling fraction", $w_c$.
* **Extra penalization term** to the cost function $J_m$ at iteration $m$:
<center>
$\Omega_m = \gamma T_m + \frac{1}{2}\lambda \| w_m \|^2_2 + \alpha \| w_m \|_1$. 
</center>    
Here $\| w_m \|^2_2 = \sum_{j=1}^{T_m}w_{jm}^2$, and $\| w_m \|_1 = \sum_{j=1}^{T_m}|w_{jm}|$. The first term $\gamma T_m$ **penalizes the number of leaf nodes**, the second term $\frac{1}{2}\lambda \| w_m \|^2_2$ is a **l2 penalization on the leaf weights**, and the third term $\alpha \| w_m \|_1$ is **l1 penalization** on the leaf weights.

In practice, you can tune the hyperparameters $w_c$, $\gamma$, $\lambda$, and $\alpha$ to regularize an XGBoost model. Now let's take a look at how each of the penailzation term affects $J_m$ and consequently the weights $w_{jm}^*$ and tree structure $\{ R_{jm} \}$.

#### Penalization of the number of leaf nodes
The cost function of iteration $m$ becomes:
<center>
    $J_m = \sum_{j=1}^{T_m} \sum_{i; \ x_i \in R_{jm}} \left[ g_m(x_i)w_{jm} + \frac{1}{2}h_m(x_i)w_{jm}^2 \right] + \gamma T_m = \sum_{j=1}^{T_m} \left[ G_{jm}w_{jm} + \frac{1}{2}H_{jm}w_{jm}^2 + \gamma \right]$
</center>
The weights are still
<center>
    $w_{jm}^* = -\frac{G_{jm}}{H_{jm}}$
</center>
for $j=1,2,...,T_m$. 

Plugging the weights back to the cost function gives us:
<center>
    $J_m^* = \sum_{j=1}^{T_m} \left[ -\frac{1}{2}\frac{G_{jm}^2}{H_{jm}} + \gamma \right]$
</center>
the gain of a potential split is:
<center>
${\mathrm {Gain}}=J_{\mathrm {before}} - J_{\mathrm {after}} = L_k^* - (L_L^*+L_R^*) = \frac{1}{2} \left[ \frac{G_L^2}{H_L} + \frac{G_R^2}{H_R} - \frac{G_{km}^2}{H_{km}} \right] - \gamma$ 
</center>
where $k$ is the node that is being split.

**If $\gamma$ is nonzero, it becomes harder to have a positive gain when spliting a node. As we have discussed in tutorial 5, this encourages a more strict pruning of the tree.**

#### l2 penalization on the leaf weights
The cost function of iteration $m$ becomes:
<center>
    $J_m = \sum_{j=1}^{T_m} \sum_{i; \ x_i \in R_{jm}} \left[ g_m(x_i)w_{jm} + \frac{1}{2}h_m(x_i)w_{jm}^2 \right] + \frac{\lambda}{2}  \sum_{j=1}^{T_m} w_{jm}^2= \sum_{j=1}^{T_m} \left[ G_{jm}w_{jm} + \frac{1}{2}\left( H_{jm} + \lambda \right) w_{jm}^2 \right]$
</center>
The weights are therefore given by:
<center>
    $w_{jm}^* = -\frac{G_{jm}}{H_{jm} + \lambda }$
</center>
for $j=1,2,...,T_m$. 

Plugging the weights back to the cost function gives us:
<center>
    $J_m^* = \sum_{j=1}^{T_m} \left[ -\frac{1}{2}\frac{G_{jm}^2}{H_{jm}+\lambda} \right]$
</center>
the gain of a potential split is:
<center>
${\mathrm {Gain}}=J_{\mathrm {before}} - J_{\mathrm {after}} = L_k^* - (L_L^*+L_R^*) = \frac{1}{2} \left[ \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{G_{km}^2}{H_{km}+\lambda} \right]$ 
</center>
where $k$ is the node that is being split.

#### l1 penalization on the leaf weights
The cost function of iteration $m$ becomes:
<center>
    $J_m = \sum_{j=1}^{T_m} \sum_{i; \ x_i \in R_{jm}} \left[ g_m(x_i)w_{jm} + \frac{1}{2}h_m(x_i)w_{jm}^2 \right] + \alpha \sum_{j=1}^{T_m} |w_{jm}| = \sum_{j=1}^{T_m} \left[ \left( G_{jm} + \alpha \cdot \mathrm{sign}(w_{jm}) \right)w_{jm}+ \frac{1}{2}H_{jm}w_{jm}^2 \right] \equiv  \sum_{j=1}^{T_m} \left[ T_{\alpha}\left( G_{jm}\right)w_{jm}+ \frac{1}{2}H_{jm}w_{jm}^2 \right]$
</center>
where 
<center>
$T_{\alpha}(G) \equiv G_{jm} + \alpha \cdot \mathrm{sign}(w_{jm})  = \mathrm{sign}(G)\mathrm{max}(0, |G| - \alpha)$
</center>
The weights are therefore given by:
<center>
    $w_{jm}^* = -\frac{T_{\alpha}(G_{jm})}{H_{jm}}$
</center>
for $j=1,2,...,T_m$. 

Plugging the weights back to the cost function gives us:
<center>
    $J_m^* = \sum_{j=1}^{T_m} \left[ -\frac{1}{2}\frac{T_{\alpha(}G_{jm})^2}{H_{jm}} \right]$
</center>
the gain of a potential split is:
<center>
${\mathrm {Gain}}=J_{\mathrm {before}} - J_{\mathrm {after}} = L_k^* - (L_L^*+L_R^*) = \frac{1}{2} \left[ \frac{T_{\alpha}(G_L)^2}{H_L} + \frac{T_{\alpha}(G_R)^2}{H_R} - \frac{T_{\alpha}(G_{km})^2}{H_{km}} \right]$ 
</center>
where $k$ is the node that is being split.


**If there are more than one types of regularization present at the same time, you can derive $w_{jm}^*$ and the gain similarly by going through the tree training steps described earlier.**

### Algorithms
* Initialize $F^{(0)}(x) = f_0(x) = \underset{\theta}{\operatorname{argmin}}\sum_{i=1}^{n}L(y_i, \theta)$
* For $m=1,2,...,M$, do:
    * $g_m(x_i) = \left[ \frac{\partial L(y_i, F(x_i)) }{\partial F(x_i)} \right]_{F(x_i)=F^{(m-1)}(x_i)}$, $i=1,2,...,n$
    * $h_m(x_i) = \left[ \frac{\partial^2 L(y_i, F(x_i)) }{\partial F(x_i)^2} \right]_{F(x_i)=F^{(m-1)}(x_i)}$, $i=1,2,...,n$
    * Determine the structure of the tree $\{R_{jm}\}_{j=1}^{T_m}$ by selecting splits that maximizes the gain. The gain is dependent on what types of regularization is present. See the above section.
    * Determine the leaf weights $\{ w_{jm}^* \}_{j=1}^{T_m}$ of the learnt structure. The specific expression is dependent on what types of regularization is present. See the above section.
    * $f_m(x) =  \eta \sum_{j=1}^{T_m} w_{jm}^*{\mathrm I}(x\in R_{jm})$
    * $F^{(m)}(x) = F^{(m-1)}(x) + f_m(x)$
* To predict:
    * Given $x$, predict $F(x) = F^{(M)}(x) = \sum_{m=0}^{M}f_m(x)$.

### XGBoost functions and examples
The XGBoost package provides a Scikit-learn API for those who are familiar with the sklearn functions. You can read more about this on the [official documentation](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn).

The two functions you can use are:
* XGBoostClassifier()
* XGBoostRegressor()

Both functions have a default learning rate $\eta=0.1$, and number of trees $M=100$. The default regularization parameters are: $\gamma = 0$, $\lambda=1$ (*reg_lambda*), and $\alpha=0$ (*reg_alpha*). That is, only the l2 regularization is present by default. To randomly subsample training instances, set *subsample* to be smaller than 1.To randomly subsamplle the input features ("random subsapce"), set *colsample_bytree* or *colsample_bylevel* to be smaller than 1. Note that *colsample_bytree* subsamples features for the training of each tree, where *colsample_bylevel* subsamples features for each split. To define $L(y, F)$, set the *objective* parameter.

#### Example 7.5
In this example, we will use an XGBoost classifier to classify the red wine. Let's first import the XGBoost module and read the data:


```python
from xgboost import XGBClassifier
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Cell In[4], line 1
    ----> 1 from xgboost import XGBClassifier
    

    ModuleNotFoundError: No module named 'xgboost'



```python
data = pd.read_csv('../input/winequality-red.csv')
data['category'] = data['quality'] >= 7 # again, binarize for classification

X = data[data.columns[0:11]].values
y = data['category'].values.astype(np.int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42) 
# random_state is fixed to guarantee repeatable results for this tutorial - remove it in actual practice

print('X train size: ', X_train.shape)
print('X test size: ', X_test.shape)
print('y train size: ', y_train.shape)
print('y test size: ', y_test.shape)
```

Then, construct a default XGBoost classifier and train it on the training set:


```python
clf = XGBClassifier(random_state = 2)
clf.fit(X_train, y_train)
```

Evaluate the performance of the classifier on the test set:


```python
p_pred = clf.predict_proba(X_test)[:,1]
fpr,tpr,thresholds = roc_curve(y_test, p_pred)

plt.subplots(figsize=(8,6))
plt.plot(fpr, tpr)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('False positive rate', fontsize = 16)
plt.ylabel('True positive rate', fontsize = 16)
plt.show()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[5], line 1
    ----> 1 p_pred = clf.predict_proba(X_test)[:,1]
          2 fpr,tpr,thresholds = roc_curve(y_test, p_pred)
          4 plt.subplots(figsize=(8,6))
    

    NameError: name 'clf' is not defined



```python
print("AUC is: ", auc(fpr, tpr))
```

The AUC is higher than what we got in gradient tree boosting. You can also play with the hyperparameters to find the best XGBoost model. Here I provide a simple example of searching in the space of regularization parameters:


```python
tuned_parameters = {'gamma':[0,1,5],'reg_alpha':[0,1,5], 'reg_lambda':[0,1,5]}

clf = GridSearchCV(XGBClassifier(random_state = 2), tuned_parameters, cv=5, scoring='roc_auc')
clf.fit(X_train, y_train)
```


```python
print('The best model is: ', clf.best_params_)
```

In this case, the best model penalizes the number of leaf nodes, but does not penalize leaf weights. We can then print the best model's performance on the test set:


```python
p_pred = clf.predict_proba(X_test)[:,1]
fpr,tpr,thresholds = roc_curve(y_test, p_pred)

plt.subplots(figsize=(8,6))
plt.plot(fpr, tpr)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('False positive rate', fontsize = 16)
plt.ylabel('True positive rate', fontsize = 16)
plt.show()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[6], line 1
    ----> 1 p_pred = clf.predict_proba(X_test)[:,1]
          2 fpr,tpr,thresholds = roc_curve(y_test, p_pred)
          4 plt.subplots(figsize=(8,6))
    

    NameError: name 'clf' is not defined



```python
print("AUC is: ", auc(fpr, tpr))
```

#### Example 7.6
In this example, we will run an XGBoost regressor on the wine quality dataset. Again, let's first import the module and read the data:


```python
from xgboost import XGBRegressor
```


```python
X = data[data.columns[0:11]].values
y = data['quality'].values.astype(np.int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

print('X train size: ', X_train.shape)
print('X test size: ', X_test.shape)
print('y train size: ', y_train.shape)
print('y test size: ', y_test.shape)
```


```python
reg = XGBRegressor(random_state = 21)
reg.fit(X_train, y_train)
```


```python
y_pred = reg.predict(X_test)
plt.plot(y_test, y_pred, linestyle='', marker='o')
plt.xlabel('true values', fontsize = 16)
plt.ylabel('predicted values', fontsize = 16)
plt.show()
print('The r2_score on the test set is: ',r2_score(y_test, y_pred))
```

Again, you can explore the hyperparameters space to improve the r2 score of the regressor. Here is my attempt:


```python
tuned_parameters = {'gamma':[0,1,5], 'reg_lambda': [1,5,10], 'reg_alpha':[0,1,5], 'subsample': [0.6,0.8,1.0]}

reg = GridSearchCV(XGBRegressor(random_state = 21, n_estimators=500), tuned_parameters, cv=5, scoring='r2')
reg.fit(X_train, y_train)
```


```python
print('The best model is: ', reg.best_params_)
```


```python
y_pred = reg.predict(X_test)
plt.plot(y_test, y_pred, linestyle='', marker='o')
plt.xlabel('true values', fontsize = 16)
plt.ylabel('predicted values', fontsize = 16)
plt.show()
print('The r2_score on the test set is: ',r2_score(y_test, y_pred))
```

Overall, it seems that Random Forests and Extra Trees are better at solving this regression probelm (see [tutorial 6](https://www.kaggle.com/fengdanye/machine-learning-6-basic-ensemble-learning)).

This is the end of the tutorial! I know that there are a lot of theories in this tutorial, but I believe understanding the theories is important for understanding the boosting method. If you have any questions or comments, please let me know! If you enjoyed this tutorial, please upvote :)

For my previous tutorials, please see: https://www.kaggle.com/fengdanye/kernels

------------------------------------
Created 2019-02-05

# References
1. Y. Freund and R.E. Schapire. "[Experiments with a New Boosting Algorithm](https://cseweb.ucsd.edu/~yfreund/papers/boostingexperiments.pdf)", *Machine Learning: Proceedings of the Thirteenth International Conference* (1996).
2. J. Zhu, S. Rosset, H. Zou and T. Hastie. "[Multi-class AdaBoost](https://web.stanford.edu/~hastie/Papers/samme.pdf)", 2006.
3. H. Drucker. "[Improving Regressors using Boosting Techniques](https://pdfs.semanticscholar.org/8d49/e2dedb817f2c3330e74b63c5fc86d2399ce3.pdf)", 1997.
4. J. H. Friedman. "[Greedy Function Approximation: A Gradient Boosting Machine](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)", 1999.
5. T. Chen and C. Guestrin. "[XGBoost: A Scalable Tree Boosting System](https://arxiv.org/pdf/1603.02754.pdf)", 2016.
6. D. Nielsen. "[Tree Boosting with XGBoost](https://brage.bibsys.no/xmlui/bitstream/handle/11250/2433761/16128_FULLTEXT.pdf)", 2016.
7. J. H. Friedman, T. Hastie, and R. Tibshirani. ["Additive Logistive Regression: A Statistical View of Boosting](https://web.stanford.edu/~hastie/Papers/AdditiveLogisticRegression/alr.pdf)", *The Annals of Statistics* 28 (2000), p337-407.
