## Machine Learning 6. Basic Ensemble Learning

[지난 튜토리얼](https://www.kaggle.com/code/cosmosjung/machine-learning-5-random-forests)에서는 랜덤 포레스트에 대해 이야기했습니다. 랜덤 포레스트는 결정 트리의 앙상블로 구성되며 __앙상블 학습__의 e대표적인 예 중 하나입니다.

이 튜토리얼은 앙상블 학습에 대한 2부로 이루어진 튜토리얼의 첫 번째 부분으로, <font color='blue'>배깅 및 페이스팅</font>와 같은 기본 개념과 기술을 다룹니다. 이어지는 2번째 튜토리얼에서는 <font color='blue'>부스팅 및 스태킹</font>과 같은 고급 기술과 강력한 기술을 다루게 될 것입니다. Kaggle에서 가장 자주 사용되는 부스팅 기술 중 하나인 <font color='blue'>XGBoost</font>도 다음 튜토리얼에서도 다룰 예정입니다. 
앙상블 학습을 처음 하는 경우에는 이 튜토리얼부터 시작하는 것을 권장합니다. 이는 앙상블 학습의 기본 개념과 기술을 이해하는 것이 고급 기술을 이해하는 기초이기 때문입니다.


--------------------------------

**목차**
    
- 앙상블 학습 분류
    - 소개
        - 투표 규칙
    - 다양한 모델로 구성된 앙상블 학습
    - 동일한 모델로 구성된 앙상블 학습
        - 훈련 인스턴스의 랜덤 샘플링
            - 배깅
                 - obb 점수
            - 페이스팅
                - 예제
            - 특성의 랜덤 샘플링
                - 예제
            - 랜덤 임계값 - 엑스트라 트리
                - 예제
    - 모델 성능 요약
- 앙상블 학습 회귀
    - 소개
    - 예제 - BaggingRegressor
        - obb 점수
    - 예제 - RandomForestRegressor
    - 예제 - ExtraTreesRegressor
    - 모델 성능 요약


```python
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import os
print(os.listdir("../input")) # input data 확인
```


```python
# plot 설정
plt.rc('axes', lw = 1.5)
plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('xtick.major', size = 5, width = 3)
plt.rc('ytick.major', size = 5, width = 3)
```

# 앙상블 학습 분류
## 소개

이전 튜토리얼에서 로지스틱 회귀 분석, SVM, 결정 트리 등의 여러 분류 모델을 소개했습니다. 지금까지는 샘플 $\vec{x}$를 분류하기 위해 하나의 문제에서 하나의 분류 모델을 사용하는 데 주로 중점을 두었습니다. 그러나 앞으로는 여러 분류기를 결합하고 각각의 분류기의 결정값으로 부터 통합된 결정 값을 찾을 수 도 있습니다. 일반적으로 이 방법은 단일 분류기보다 더 좋은 결과를 보장합니다 (예: 테스트 세트에 대한 더 나은 일반화, 과대적합되기가 어려움). 랜덤 포레스트의 예를 사용한 지난 튜토리얼에서 이를 보았습니다. 여러 분류 모델의 예측에 따라 인스턴스를 분류하는 기술을 __앙상블 학습 분류__ 라고 합니다.

### 투표 규칙

이제 앙상블에 5개의 분류 모델이 있고 단일 인스턴스 $\vec{x}$에 대한 클래스를 예측한다고 가정해 보겠습니다. 각 분류 모델은 독립적으로 값을 분류 할 것입니다. 이 경우 앙상블은 __과반수 투표__를 기반으로 클래스를 예측합니다.

<img src="https://imgur.com/eAruUj3.png" width="600px"/>

보시다시피, 네 개의 분류 모델이 "Class 2"로 예측하고 한 개의 분류 모델만 "Class 1"으로 예측했으므로 앙상블은 최종 예측을 "클래스 2"로 결정합니다. 이 투표 규칙은 때때로 __직접 투표__ 라고 불립니다.

이제부터 지금까지 소개한 많은 분류 모델들은 단순히 클래스를 예측할 뿐만 아니라 각 클래스에 대한 예측 확률도 제공한다는 것을 고려해야 합니다. 앙상블의 모든 분류모델이 예측 확률을 갖는 경우, 소위 __간접 투표__ 규칙을 사용할 수도 있습니다.

<img src="https://imgur.com/ud382N9.png" width="600px"/>

여기서 각 클래스의 확률은 앙상블의 모든 분류 모델에 걸쳐 평균화되며 평균 확률이 가장 높은 클래스가 예측됩니다. __많은 경우 간접 투표는 더 많은 정보를 수용하고 매우 신뢰할 수 있는 예측(즉, 높은 확률의 예측)에 더 높은 가중치를 부여하기 때문에 직접 투표 방식보다 더 나은 성능을 발휘합니다.__ 예를 들어 [지난 튜토리얼](https://www.kaggle.com/code/cosmosjung/machine-learning-5-random-forests)에서 소개한 대로 Scikit-learn의 랜덤 포레스트 분류기는 기본적으로 간접 투표 방식을 사용합니다.

#### SVM에서 예측 확률을 얻는 방법에 대한 참고 사항 (관심이 있는 경우에만 해당)

SVM은 기본적으로 예측 확률을 생성하지 않는 분류모델 중 하나입니다. 그러나 Scikit-learn의 SVC에서 "proability" 파라미터를 _True_ 로 설정하여 확률을 강제로 계산할 수 있습니다. 해당 파라미터를 설정하면, predict_proba() 함수를 호출하여 클래스 확률을 얻을 수 있습니다. 확률을 구하는 방법을 이해하려면 [설명서](https://scikit-learn.org/stable/modules/svm.html#scores-and-probabilities)와 [스택 오버플로우 답변](https://stackoverflow.com/questions/15111408/how-does-sklearn-svm-svcs-function-predict-proba-work-internally)을 참조 할 수 있습니다. 기본 아이디어는 다음과 같습니다.

* 먼저 교차 검증 방식으로 SVM을 교육합니다. 각 폴드에는 훈련 세트와 홀드 아웃 세트가 있습니다. $\vec{w}$ 및 $b$는 훈련 세트에 대한 훈련을 통해 얻은 다음, 홀드 아웃 세트에서 $\vec{w}\cdot \vec{x}+b$가 계산됩니다. 이것이 5배 교차 검증인 경우(Scikit-learn의 SVC의 경우), 5개의 홀드아웃 세트가 있으며, 그 중 하나가 전체 데이터 세트입니다. 앞서 언급한 링크에서 $\vec{w}\cdot \vec{x}+b$는 $f$로 표시됩니다.

- 홀드아웃 세트에서 계산된 $\vec{w}\cdot \vec{x}+b$은 로지스틱 시그모이드 함수 $P(y=1|f)=\frac{1}{1+exp(Af+B)}$를 훈련하는 데 사용됩니다. 여기서 $f=\vec{w}\cdot \vec{x}+b$는 홀드아웃 세트에 있습니다. $P(y=1)>0.5$은 $y=1$의 값으로 수렴하며, $P(y=1)	\leq0.5$는 $y=0$의 값으로 수렴합니다. 훈련 중 매개 변수 $A$ 및 $B$는 교차 엔트로피 손실 함수를 최소화하도록 최적화됩니다. 이는 기본적으로 SVM 점수에 대한 로지스틱 회귀 분석입니다.

* 이제 $A$ 및 $B$가 있으므로 SVM은 전체 데이터 세트에 대해 다시 훈련합니다. 주어진 인스턴스 $\vec{x}$에 대해 재교육된 SVM은 $f=\vec{w}\cdot \vec{x}+b$ 값을 생성한 다음 $P(y=1|f)=\frac{1}{1+exp(Af+B)}$를 통해 확률 값을 생성합니다.

SVM 점수에 대해 로지스틱 회귀 분석을 수행하는 방법을 __플랫 스케일링__(**Platt scaling**)이라고 합니다. 이 방법에 대해 자세히 알고 싶다면 Platt의 논문 "[Probabilistic outputs for SVMs and comparisons to regularized likelihood methods](http://www.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf)"를 참고 할 수 있습니다. 다중 클래스의 경우는 Wu, Lin 과 Weng이 작성한 "[Probability estimates for multi-class classification by pairwise coupling](https://www.csie.ntu.edu.tw/~cjlin/papers/svmprob/svmprob.pdf)", JMLR 5:975-1005, 2004에 의해 확장되었습니다.

추가적으로 SVM에 예측 확률 계산을 요청하면 5배의 교차 검증이 있으므로 교육 속도가 크게 느려집니다.

이제부터는 분류 모델의 앙상블에 대해 이야기할 때, 두 가지 가능성이 있습니다.

* 여러 분류 모델을 포함하는 앙상블, 예를 들어 로지스틱 회귀 분류자, SVM 분류 모델 과 결정 트리를 모두 동시에 포함할 수 있는 앙상블 학습 모델이 있습니다.
* 동일한 유형의 분류 모델로 구성된 앙상블입니다. 랜덤 포레스트 모델이 이에 속합니다.

<img src="https://imgur.com/mlRpzUf.png" width="400px"/>

우선 서로 다른 분류 모델에 대한 앙상블 학습에 대해 이야기하겠습니다.

## 다양한 모델로 구성된 앙상블 학습
다음과 같이 다양한 분류 모델로 간단한 앙상블을 구성합니다.

<img src="https://imgur.com/3G6054b.png" width="400px"/>

이 앙상블에는 __랜덤 포레스트 분류기, SVM 분류기 및 로지스틱 회귀 분류기__ 의 세 가지 분류기를 사용합니다. 참고적으로 랜덤 포레스트 자체는 그 자체도 앙상블 학습 분류기입니다. 훈련 데이터로는 다시 레드 와인 품질 데이터 세트를 사용합니다.


```python
wineData = pd.read_csv('../input/winequality-red.csv')

wineData['category'] = wineData['quality'] >= 7

X = wineData[wineData.columns[0:11]].values
y = wineData['category'].values.astype(np.int)
```


```python
wineData.head()
```

와인 품질은 "good"($y=1$, quality>=7) 또는 "bad"($y=0$, quality<7)으로 이진화됩니다. 입력 $X$는 고정 산도 및 pH와 같은 11가지 특성으로 구성됩니다. 이제는 데이터 세트를 훈련 세트와 테스트 세트로 분할합니다.


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print('X train size: ', X_train.shape)
print('y train size: ', y_train.shape)
print('X test size: ', X_test.shape)
print('y test size: ', y_test.shape)
```

다시 말하지만, "random_state"는 이 튜토리얼에서 반복 가능한 결과를 보장하도록 설정됩니다. 실제 모델이 작동할 때에는 random_state 인수를 제거해야 합니다.

투표 분류 모델을 구성하기 위해, 우리는 scikit-learn의 **VotingClassifier**를 사용합니다. 투표 규칙은 "soft"로 설정합니다. 이는 간접 투표 방식만이 예측 확률을 생성하고, ROC 곡선을 그리기 위해서는 확률 값이 필요하기 때문입니다.


```python
# Below, random_state is only used to guarantee repeatable result for the tutorial. 
rfClf = RandomForestClassifier(n_estimators=500, random_state=0) # 500 trees. 
svmClf = SVC(probability=True, random_state=0) # force a probability calculation
logClf = LogisticRegression(random_state=0)

clf = VotingClassifier(estimators = [('rf',rfClf), ('svm',svmClf), ('log', logClf)], voting='soft') # construct the ensemble classifier
```

__앙상블 분류기를 구성하려면, VotingClassifier()에 원하는 예측치를 추정하기 위해 필요한 모델들을 포함시켜 주면 됩니다.__ 이 경우에는, "estimators = [('rf', rfClf), ('svm', svmClf), ('log', logClf)"가 있습니다. 이제 훈련 세트를 통해 분류 모델을 훈련합니다.


```python
clf.fit(X_train, y_train) # train the ensemble classifier
```

이제 해당 분류 모델이 테스트 세트에서 어떻게 작동하는지 확인합니다.


```python
from sklearn.metrics import precision_score, accuracy_score
y_true, y_pred = y_test, clf.predict(X_test)
print('precision on the test set: ', precision_score(y_true, y_pred))
print('accuracy on the test set: ', accuracy_score(y_true, y_pred))
```

해당 모델은 68.2%의 정밀도와 87.7%의 정확도를 달성했습니다. [지난 튜토리얼](https://www.kaggle.com/code/cosmosjung/machine-learning-5-random-forests)에서 단일 랜덤 포레스트 분류모델로 58.3%의 정밀도와 87.7%의 정확도를 달성했습니다. 정확도는 향상되지 않았지만, 앙상블 학습 분류를 사용하여 정밀도가 상당히 향상되었습니다. 정밀도 = TP/(TP+FP)임을 기억하십시오. 정밀도가 향상되었다는 것은 모든 예측 양성 중에서 참 양성의 비율이 증가했다는 것을 나타냅니다.

물론, 계산할 수 있는 측정치는 정밀도와 정확도 이상입니다. 성능 측정의 전체 목록에 대한 정의는 이 [Wikipedia 페이지](https://en.wikipedia.org/wiki/Confusion_matrix)에서 확인할 수 있습니다.

이제 ROC 곡선을 그리고 AUC를 계산해 보겠습니다.


```python
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
phat = clf.predict_proba(X_test)[:,1]
```


```python
plt.subplots(figsize=(8,6))
fpr, tpr, thresholds = roc_curve(y_test, phat)
plt.plot(fpr, tpr)
x = np.linspace(0,1,num=50)
plt.plot(x,x,color='lightgrey',linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
```


```python
print('AUC is: ', auc(fpr,tpr))
```

이 AUC(0.914)는 [지난 튜토리얼](https://www.kaggle.com/code/cosmosjung/machine-learning-5-random-forests)의 단일 랜덤 포레스트 분류 모델(0.909)에서 계산한 AUC에 약대해 약간 값이 개선되었습니다. 

ROC 곡선 및 AUC에 대한 소개는 [이전 튜토리얼](https://www.kaggle.com/code/cosmosjung/machine-learning-5-random-forests)을 참조하십시오.

## 동일한 모델로 구성된 앙상블 학습

__랜덤 포레스트는 동일한 분류 모델로 구성된 앙상블 학습 중 가장 인기 있는 모델 중 하나입니다.__ 랜덤 포레스트에서 각 트리는 동일한 하이퍼 파라미터(예: max_depth 및 min_sample_leaf)를 가지지만 중복이 허용된 훈련 세트를 통해 훈련됩니다. max_features가 1보다 작으면 랜덤 포리스트의 트리도 랜덤하게 샘플링된 특성의 서브 샘플을 훈련 데이터로 하여 분할됩니다. [지난 튜토리얼](https://www.kaggle.com/code/cosmosjung/machine-learning-5-random-forests)에서, 우리는 랜덤 포레스트의 훈련과 사용의 예를 보았습니다.

__이제, 랜덤 포레스트의 맥락에서 벗어나 보겠습니다. 좀 더 일반적인 경우를 생각해 보겠습니다. 여기서 같은 종류의 분류모델로 이루어진 앙상블을 가지고 있다고 가정합니다.__ 이는 SVM의 앙상블, 로지스틱 회귀 분류기의 앙상블, 결정 트리의 앙상블 또는 랜덤 포레스트 분류기의 앙상블일 수 있습니다. __앙상블이 잘 수행되기 위해서는 개별 분류기가 가능한 한 독립적이기여야 합니다.__ 이전 단원에서는 다른 유형의 분류 모델을 사용하여 이 문제를 해결했습니다. 분류 모델이 같은 유형인 경우, 각 분류기에 무작위성을 도입하여 그렇게 할 수 있습니다. 이 튜토리얼에서는 이를 위한 세 가지 방법에 대해 설명합니다.

1. 각 분류기에 대한 훈련 인스턴스의 랜덤 샘플 추출
2. 각 분류기를 훈련시키는 데 사용되는 랜덤 특성 샘플
3. 결정 트리의 경우 - 각 특성에 대해 랜덤 임계값 사용 (최적 임계값을 찾는 것과 비교).

위 3가지 조건에 대해서 그것들을 하나씩 살펴 볼 것입니다.

### 훈련 인스턴스의 랜덤 샘플링
#### 배깅(Bagging)

배깅(Bagging)은 훈련 인스턴스를 _대체_ 할 샘플을 무작위로 뽑는 방법을 말합니다. 통계학에서는 대체할 샘플을 뽑는 과정을 _부트스트랩(bootstrapping)_ 이라고도 합니다. **대체 샘플이라는 용어는 훈련 세트에서 무작위로 한 인스턴스를 취한 후 이 인스턴스를 훈련 세트를 대체하여 넣는 것을 의미합니다. 다음 인스턴스를 선택할 때, 선택할 다음 인스턴스는 이전에 선택한 인스턴스와 동일할 수 있습니다.** 다음은 배깅의 간단한 예입니다.

<img src="https://imgur.com/XA7mf26.png" width="400px"/>

보다시피 동일한 인스턴스가 서브 샘플의 형태로 여러 번 나타날 수 있습니다. 이것이 바로 배깅 방식의 특징입니다.

##### oob 점수

배깅 중에 각 서브 샘플은 하나의 분류기를 훈련시키는 데 사용됩니다. 각 분류기에 대해 훈련 중에 사용되지 않는 샘플을 __Out-of-Bag 인스턴스__ 또는 __oob 인스턴스__ 라고 합니다.

<img src="https://imgur.com/ssSY5Kj.png" width="500px"/>

이러한 oob 인스턴스는 훈련 중에 볼 수 없는 데이터 세트인 테스트 세트와 동일한 기능을 수행하므로 분류기의 성능을 평가하는 데 사용할 수 있습니다. 배깅 방법을 사용하여 oob 점수를 평가하기 위해 Scikit-learn의 __Bagging Classifier__ 를 사용하고 ob_score=True로 설정합니다. [source code of BaggingClassifier](https://github.com/scikit-learn/scikit-learn/blob/7389dba/sklearn/ensemble/bagging.py#L430)(line 583-618)에서 oob 점수가 다음과 같이 계산됨을 알 수 있습니다.

* 훈련된 각 분류기에 대해 해당 oob 인스턴스를 찾습니다.
* 훈련된 분류기를 사용하여 oob 인스턴스를 예측합니다.
    * 분류기에 predict_proba()가 있으면 각 ob 인스턴스에 대한 각 클래스의 확률을 예측합니다.
    * 분류기에 predict_proba()가 없으면 각 ob 인스턴스가 속한 클래스를 예측합니다.
* 앙상블의 모든 분류 모델에 대해 이 작업을 수행합니다.
* 전체 훈련 세트의 각 인스턴스에 대해 이 인스턴스가 oob 인스턴스인 분류 모델을 찾습니다. 편의를 위해 이러한 분류 모델을 이 인스턴스에 대한 "oob 분류 모델"라고 부릅니다.
    * 분류기에 predict_proba()가 있는 경우 이 인스턴스의 클래스를 이 인스턴스의 모든 "oob 분류 모델"에서 평균 확률이 가장 높은 클래스로 예측합니다.
    * 분류기에 predict_proba()가 없으면 이 인스턴스의 클래스를 이 인스턴스에 대한 모든 "oob 분류 모델"의 대부분의 투표에서 뽑힌 클래스로 예측합니다.
* 전체 훈련 세트의 모든 인스턴스에 대해 이 작업을 수행합니다. 편의를 위해 전체 훈련 세트의 예측 값의 이름을 y_oob로 지정합니다.
- __최종 oob 점수는 위 예측의 정확도 점수입니다: accuracy_score(y_true, y_oob)__. y_true 및 y_oob 모두 m 크기를 가지며, 여기서 m은 총 훈련 인스턴스 수입니다.

__요약하면, Bagging Classifier의 oob 점수를 통해 테스트 세트에서 앙상블 분류기의 정확도를 추정할 수 있습니다.__ 이 단원의 마지막 부분에서는 Bagging Classifier를 사용하여 oob 점수를 얻는 방법에 대한 예를 보여드리겠습니다.

#### 페이스팅(Pasting)

페이스팅은 _중복 없이_ 훈련 인스턴스를 무작위로 샘플링하는 방법을 말합니다. 즉, 특정 서브 샘플에서 동일한 인스턴스는 다음과 같이 한 번만 나타날 수 있음을 의미합니다.

<img src="https://imgur.com/5ZMvOoL.png" width="400px"/>

속성 _bootstrap_ 이 _False_ 로 설정된 경우, Scikit-learn의 BaggingClassifier도 페이스팅을 수행할 수 있습니다.

#### 예제

이 예제에서는 단일 로지스틱 회귀 분석 분류 모델의 성능과 로지스틱 회귀 분석 분류 모델의 앙상블 학습 모델의 성능을 비교합니다. 먼저 데이터를 읽어 보겠습니다.


```python
wineData = pd.read_csv('../input/winequality-red.csv')

wineData['category'] = wineData['quality'] >= 7

X = wineData[wineData.columns[0:11]].values
y = wineData['category'].values.astype(np.int)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print('X train size: ', X_train.shape)
print('y train size: ', y_train.shape)
print('X test size: ', X_test.shape)
print('y test size: ', y_test.shape)
```

__먼저 단일 로지스틱 회귀 분석 분류기를 사용하여 정밀도, 정확도 및 AUC 측면에서 어떻게 수행되는지 살펴보겠습니다.__ [튜토리얼 3](https://www.kaggle.com/fengdanye/machine-learning-3-logistic-and-softmax-regression)에서 로지스틱 회귀 모델을 도입하고 매우 유사한 교육/예측을 수행했지만, 해당 튜토리얼에서는 훈련-테스트 분할을 수행하지 않았습니다. 여기서는 동일한 훈련/예측을 수행하지만 훈련-테스트 분할을 사용합니다.

첫 번째 단계는 교육 데이터를 표준화하는 것입니다.


```python
scaler = StandardScaler()
X_train_stan = scaler.fit_transform(X_train)
```

이제 표준화된 훈련 세트에서 로지스틱 회귀 분류기를 훈련하고 테스트 세트에서 성능을 평가합니다.


```python
logReg = LogisticRegression(random_state=0, solver='lbfgs') # random_state is only set to guarantee for repeatable result for the tutorial
logReg.fit(X_train_stan, y_train)

X_test_stan = scaler.transform(X_test) # don't forget this step!
y_pred = logReg.predict(X_test_stan)

print('precision on the test set: ', precision_score(y_test, y_pred))
print('accuracy on the test set: ', accuracy_score(y_test, y_pred))
```


```python
phat = logReg.predict_proba(X_test_stan)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, phat)

plt.subplots(figsize=(8,6))
plt.plot(fpr, tpr)
x = np.linspace(0,1,num=50)
plt.plot(x,x,color='lightgrey',linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
```


```python
print('AUC is: ', auc(fpr,tpr))
```

__이제, 500개의 로지스틱 회귀 분류기를 배깅 방법의 앙상블 학습으로 훈련해 보겠습니다.__ 이를 위해 Scikit-learn의 __BaggingClassifier__ 를 사용합니다


```python
bagClf = BaggingClassifier(LogisticRegression(random_state=0, solver='lbfgs'), n_estimators = 500, oob_score = True, random_state = 90)
```

기본적으로 BaggingClassifier에는 _max_bootstrap=1.0_ 과_bootstrap=True_ 속성이 있습니다. __이는 이 앙상블 분류기가 중복을 허용하여(즉, 배깅) 서브 샘플을 만들고 있으며, 서브 샘플 크기는 전체 훈련 세트의 크기와 같다는 것을 의미합니다.__ 또한 정확도 추정을 위해 _ob_score_ 속성을 _True_ 로 설정했습니다.

이제 앙상블 분류기를 훈련하고 평가하겠습니다.


```python
bagClf.fit(X_train_stan, y_train)
print(bagClf.oob_score_) # The oob score is an estimate of the accuracy of the ensemble classifier, as introduced earlier. 
```


```python
y_pred = bagClf.predict(X_test_stan)
phat = bagClf.predict_proba(X_test_stan)[:,1]

print('precision on the test set: ', precision_score(y_test, y_pred))
print('accuracy on the test set: ', accuracy_score(y_test, y_pred))
```


```python
fpr, tpr, thresholds = roc_curve(y_test, phat)
plt.subplots(figsize=(8,6))
plt.plot(fpr, tpr)
x = np.linspace(0,1,num=50)
plt.plot(x,x,color='lightgrey',linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
```


```python
print('AUC is: ', auc(fpr,tpr))
```

앙상블 분류기는 단일 로지스틱 회귀 분류기에 비해 약간 개선되었지만, 이전에 사용한 단순한 3-분류기(SVM + 랜덤 포레스트 + 로지스틱) 앙상블의 성능과 비교할 정도로 성능이 향상되지는 않았습니다. 500개의 로지스틱 회귀 분류기로 구성된 이 앙상블은 또한 500개의 결정 트리가 있는 [지난 튜토리얼](https://www.kaggle.com/code/cosmosjung/machine-learning-5-random-forests)에서 사용한 랜덤 포레스트보다 성능이 떨어집니다. **한 가지 긍정적인 부분은 500개의 로지스틱 회귀 분류기로 구성된 이 앙상블이 훈련 인스턴스만 무작위로 샘플링하는 반면, 랜덤 포레스트는 (_max_feature_ 속성을 사용하여) 훈련 인스턴스와 특성을 둘 다 무작위로 샘플링한다는 점입니다.** 이는 현재 앙상블에서 개별 분류 모델 간의 독립성 부족으로 이어질 수 있으므로 성능이 약간 저하될 수 있습니다.

다음 단원에서는 입력 특성을 무작위로 샘플링하는 방법에 대해 설명합니다.

### 특성의 랜덤 샘플링

개별 분류기 간의 독립성을 더욱 높이기 위해, __각 분류기를 서로 다른 임의의 특성 서브 샘플에 대해 훈련시킬 수 있습니다.__ 와인 품질 데이터 세트에는 다음과 같은 11가지 입력 특성이 있습니다.


```python
wineData.head(0)
```

샘플 크기가 3인 위의 특성을 _대체하는_ 랜덤 샘플링은 다음과 같습니다.

<img src="https://imgur.com/aeRjeR6.png" width="500px"/>

다시 말하지만, _중복하지 않고_ 표본을 추출하는 경우 서브 샘플에 반복적인 특성이 없습니다. __특성의 랜덤 샘플링을 수행하기 위해 다시 Bagging Classifier를 사용할 수 있습니다.__ 두 가지 주요 인자는 _bootstrap_features_와 _max_features_입니다. _bootstrap_features_는 중복 샘플링 여부를 결정하고 _max_features_는 입력 특성에서 고려한 특성의 비율을 결정합니다. 예를 들어, 총 특성 수와 동일한 크기의 특성의 중복 허용 샘플을 그리도록 _bootstrap_scripts = True_ 및 _max_scripts = 1.0_을 설정할 수 있습니다.

훈련 인스턴스을 랜덤 샘플링하지 않고 특성만 랜덤 샘플링 하는 것을 __랜덤 서브스페이스__ 방법이라고 합니다. 이는 _bootstrap = False, max_buffer = 1.0_ 과 _bootstrap_buffer = True _ and/or _max_buffer < 1.0_ 에 해당합니다. 물론 특성과 훈련 인스턴스를 동시에 샘플링할 수 도 있습니다. 이 방법을 __랜덤 패치라고 합니다. 랜덤 패치 방법에서 각 분류기는 훈련 인스턴스 및 특성의 해당 서브 샘플로 훈련됩니다.__ oob 점수는 앞에서 설명한 대로 계산됩니다. 

추가적인 세부 사항으로 각 분류기가 oob 샘플에 대해 예측을 할 때는 분류기가 특성의 서브 샘플만 사용합니다.

#### 예제

이제 다시 500개의 로지스틱 회귀 분류기의 앙상블을 사용해 보겠습니다. 하지만, 이번에는 __훈련 인스턴스와 특성을 모두 무작위로 샘플링합니다.__


```python
bagClf = BaggingClassifier(LogisticRegression(random_state=0, solver='lbfgs'), n_estimators = 500, 
                           bootstrap_features = True, max_features = 1.0, oob_score = True, random_state = 90)
# Notice that bootstrap_features is set to True.
```


```python
bagClf.fit(X_train_stan, y_train)
print(bagClf.oob_score_) # The oob score is an estimate of the accuracy of the ensemble classifier
```


```python
y_pred = bagClf.predict(X_test_stan)
phat = bagClf.predict_proba(X_test_stan)[:,1]

print('precision on the test set: ', precision_score(y_test, y_pred))
print('accuracy on the test set: ', accuracy_score(y_test, y_pred))
```


```python
fpr, tpr, thresholds = roc_curve(y_test, phat)
plt.subplots(figsize=(8,6))
plt.plot(fpr, tpr)
x = np.linspace(0,1,num=50)
plt.plot(x,x,color='lightgrey',linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
```


```python
print('AUC is: ', auc(fpr,tpr))
```

랜덤 패치 방법을 사용하여 예측의 정밀도가 상당히 증가했지만 (0.54 -> 0.61), 정확도와 AUC는 거의 동일하게 유지되었습니다.

### 랜덤 임계값 - 엑스트라 트리

결정 트리의 특정한 경우, 우리는 각 노드에 무작위 임계 값을 도입하여 개별 트리를 추가로 무작위화할 수 있습니다. 이 나무들은 **극도로 무작위화된 나무\[:Extremely Randomized Trees](또는 엑스트라 나무)라고 불립니다.** 엑스트라 트리 분류기는 랜덤 포레스트 분류기와 유사하게 작동합니다. 각 노드의 최적 분할을 위해 특성의 랜덤 서브 샘플을 탐색합니다. __차이점은 랜덤 포레스트는 각 분할에서 최적(특성, 임계값) 조합을 탐색하는 반면, 엑스트라 트리에서는 각 후보 특성의 임계값을 무작위로 생성하고(특성당 하나의 임계값), 그 중에서 가장 적합한 것이 선택된다는 것입니다([설명서](https://scikit-learn.org/stable/modules/ensemble.html#forest) 참조).__ [used by Extra Trees](https://github.com/scikit-learn/scikit-learn/blob/7389dba/sklearn/tree/tree.py#L1146)를 기반으로 한 [source code for random splitter](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_splitter.pyx#L652)에서도 이를 확인할 수 있습니다(관심이 있는 경우에만 참고하시기 바랍니다). 랜덤 분할기(splitter)는 기본적으로 한 번에 하나의 특성을 생성하고, 이 특성에 대한 임의의 임계값을 선택하고, 분할의 성능을 평가함으로써 작동합니다. 이 분할이 이전 최고값보다 나은 경우, 현재 최고값으로 대체됩니다. 이 반복은 _max_features_가 생성된 경우(중복 없이) 또는 생성된 특성의 수가 _max_features_보다 적지만 나머지 특성이 모두 상수인 경우에 종료됩니다. 이것이 랜덤 분할기(splitter)가 "최상의 랜덤 분할을 선택한다."고 말하는 이유입니다.

__또한 기본적으로 Scikit-learn의 RandomForestClassifier은 bootstrap = True인 반면 ExtraTreeClassifier은 bootstrap = False입니다. 이는 ExtraTressClassifier가 랜덤 샘플링 없이 전체 훈련 인스턴스를 사용한다는 것을 의미합니다.__ (RandomForestClassifier 및 ExtraTreeClassifier 모두 샘플 크기 = 전체 훈련 세트 크기이므로 bootstrap=False는 전체 훈련 인스턴스가 사용됨을 의미합니다.)

#### 예제

이제 와인 품질 데이터 세트에서 **ExtraTreesClassifier**를 사용해 보겠습니다. [지난 튜토리얼](https://www.kaggle.com/code/cosmosjung/machine-learning-5-random-forests)에서 수행한 작업과 마찬가지로 **GridSearchCV**를 사용하여 최상의 하이퍼 파라미터를 탐색합니다. GridSearchCV는 [4번째 튜토리얼](https://www.kaggle.com/fengdanye/machine-learning-4-support-vector-machine)에 소개되어 있습니다.


```python
tuned_parameters = {'n_estimators':[500],'n_jobs':[-1], 'max_features': [0.5,0.6,0.7,0.8,0.9,1.0], 
                    'max_depth': [10,11,12,13,14],'min_samples_leaf':[1,10,100],'random_state':[0]} 

clf = GridSearchCV(ExtraTreesClassifier(), tuned_parameters, cv=5, scoring='roc_auc')
clf.fit(X_train, y_train)
```


```python
print('The best model is: ', clf.best_params_)
print('This model produces a mean cross-validated score (auc) of', clf.best_score_)
```

GridSesearchCV는 _max_depth = 13, max_depth = 0.9_ 와 _min_depth_leaf = 1_ 을 갖는 최상의 모델을 결정했습니다. 이제 엑스트라 트리 분류기가 테스트 세트에서 어떻게 수행되는지 살펴보겠습니다.


```python
y_pred = clf.predict(X_test)
print('precision on the evaluation set: ', precision_score(y_test, y_pred))
print('accuracy on the evaluation set: ', accuracy_score(y_test, y_pred))
```


```python
phat = clf.predict_proba(X_test)[:,1]
plt.subplots(figsize=(8,6))
fpr, tpr, thresholds = roc_curve(y_test, phat)
plt.plot(fpr, tpr)
x = np.linspace(0,1,num=50)
plt.plot(x,x,color='lightgrey',linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
```


```python
print('AUC is: ', auc(fpr,tpr))
```

이 엑스트라 트리 분류기는 테스트 세트에서 가장 높은 정확도와 AUC 점수를 가지고 있습니다!

## 모델 성능 요약

이 튜토리얼과 [지난 튜토리얼](https://www.kaggle.com/code/cosmosjung/machine-learning-5-random-forests)에서는 와인 품질 데이터 세트에 대한 다양한 앙상블 학습 분류기를 교육했습니다. 항상 _random_state=42_ 로 훈련/테스트 세트를 분할했기 때문에, 이 모든 분류기는 동일한 데이터에 대해 훈련되고 동일한 데이터에 대해 테스트되었습니다. 이를 통해 여러 앙상블 분류기의 성능을 비교할 수 있습니다. 물론 분류기에 대한 _random_state_ 설정도 성능에 영향을 미치며, 실제 훈련에서는 대개 _random_state_ 가 무작위로 생성됩니다.(즉, _random_state = None_) 그러나 반복 가능한 결과를 위해 이 튜토리얼에서 특정 _random_state_ 값을 설정하였습니다.

다음은 레드 와인 품질 데이터 세트에 대한 다양한 앙상블 분류기의 성능 요약 표입니다.

| Classifier name | Precision on test set   | Accuracy on test set | AUC on test set | Comments |
|------|------|------|------|------|
|   Random Forest with 500 trees (with GridSearchCV)  | 0.58 | 0.88 | 0.91 | from the [last tutorial](https://www.kaggle.com/fengdanye/machine-learning-5-random-forests) |
|RF-SVM-logistic classifier | **0.68** | 0.88 | 0.91| simple ensemble of three different classifiers |
| 500 logisitic classifiers with bagging | 0.54 | 0.87 | 0.88| bagging method - randomly sample training instances|
| 500 logisitic classifiers with Random Patches | 0.61 | 0.87 | 0.88| Random Patches - randomly sample both training instances and features|
|Extra Trees with 500 trees (with GridSearchCV)| 0.62 | **0.89** | **0.93**| Extra trees use best random split|

The best score has been bolded. Some discussion:
최고 점수는 굵게 표시되었습니다. 아래에서는 몇 가지 논의사항을 다룹니다.

* 정밀도에 가장 관심이 있다면 RF-SVM-logisitc classifier가 가장 좋습니다. 정확도 또는 AUC에 가장 관심이 있다면 Extra Tree classifier가 가장 좋습니다.
* 랜덤 패치 방법은 정밀도 측면에서 500개의 로지스틱 분류 모델을 개선합니다.
* 엑스트라 트리 분류기가 랜덤 포레스트 분류기보다 성능이 우수합니다.
* 랜덤 포레스트 분류기에 SVM 및 로지스틱 분류기를 추가하면 정밀도가 크게 향상됩니다. 

# 앙상블 학습 회귀
## 소개

앙상블 학습 회귀는 분류와 유사하게 작동합니다. 그러나 회귀에서는 다른 유형의 회귀 모델를 사용하는 것보다 같은 유형의 회귀 모델을 사용하는 것이 더 일반적입니다. Scikit-learn은 회귀 모델에 대해서 분류 모델과 같이 VotingClassifier()와 동등한 모델을 가지지 않으므로, 다양한 유형의 회귀 모델로 앙상블을 구축하려면 추가적인 작업을 해야 합니다. 반면에, 같은 유형의 회귀 모델로 앙상블을 구축하는 것은 빠르고 쉽습니다. Scikit-learn은 이러한 목적을 위해 **BaggingRegressor, RandomForestRegressor 및 ExtraTreeRegressor** 와 같은 클래스를 제공합니다.

이어지는 내용에서 회귀 모델의 유형별로 하나의 예제를 보여드리겠습니다.__이번에는 레드 와인 품질 데이터 세트를 회귀 문제로 풀 것입니다. 입력은 11가지 특성이며 출력은 와인 품질(0-10)입니다.__ 그리고 정밀도, 정확도, AUC를 평가하는 대신 예측 값에 대한 __r2 점수__ 로 평가할 것입니다. r2 점수는 종종 "결정 계수"라고도 하며 많은 통계 패키지에서 "R 제곱"으로 표시됩니다. 당신은 이 [위키피디아 페이지](https://en.wikipedia.org/wiki/Coefficient_of_determination)에서 r2 점수의 정의를 찾을 수 있습니다. r2 점수는 1.0에 가까울수록 좋습니다.

먼저 데이터를 읽어 보겠습니다.


```python
wineData = pd.read_csv('../input/winequality-red.csv')
wineData.head()
```


```python
X = wineData[wineData.columns[0:11]].values
y = wineData['quality'].values.astype(np.float)
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=5)

print('X train size: ', X_train.shape)
print('y train size: ', y_train.shape)
print('X test size: ', X_test.shape)
print('y test size: ', y_test.shape)
```

## 예제 - BaggingRegressor

Scikit-learn의 **BaggingRegressor**로 시작하겠습니다. BaggingRegressor는 BaggingClassifier와 매우 유사하게 작동합니다. 기본적으로 BaggingRegressor는 훈련 인스턴스의 중복 가능한 샘플을 통해 앙상블의 각 회귀 모델을 훈련합니다. 또한 입력 특성에 대해서도 중복을 허용하도록 _bootstrap_message = True_ 로 설정합니다.

### oob 점수
 
BaggingClassifier와 마찬가지로 BaggingRegressor는 oob 점수를 계산할 수 있습니다. oob 점수는 다음과 같이 계산됩니다([소스 코드 참조](https://github.com/scikit-learn/scikit-learn/blob/7389dba/sklearn/ensemble/bagging.py#L991)).
 
* 훈련된 각 회귀 분석기에 대해 해당 oob 인스턴스를 찾습니다.
* 훈련된 회귀 분석기를 사용하여 oob 인스턴스를 예측합니다. 특성 샘플링이 활성화된 경우 선택한 특성만 예측에 사용됩니다.
* 앙상블의 모든 회귀 모에 대해 이 작업을 수행합니다.
* 전체 훈련 세트의 각 인스턴스에 대해 이 인스턴스가 oob 인스턴스인 회귀 모델을 찾습니다. 편의상, 저는 이 경우에 이 회귀 모델들을 "oob 회귀 모델"라고 부를 것입니다. 이 인스턴스의 값을 이 인스턴스의 모든 "oob 회귀 모델"에서 평균 예측 값으로 예측합니다.
* 전체 훈련 세트의 모든 인스턴스에 대해 이 작업을 수행합니다. 편의를 위해 전체 훈련 세트의 예측값의 이름을 y_oob로 지정합니다.
* __최종 oob 점수는 위 예측의 r2_score(y_true, y_ob)입니다.__ y_true 및 y_oob 모두 m 크기를 가지며, 여기서 m은 총 훈련 인스턴스 수입니다.

__BaggingRegressor의 oob socre를 통해 테스트 세트에서 앙상블 회귀 분석기의 r2 점수를 추정할 수 있습니다.__


```python
# Standardize the data
scaler = StandardScaler()
X_train_stan = scaler.fit_transform(X_train)
X_test_stan = scaler.transform(X_test)
```


```python
# use an ensemble of 500 linear regressors. Use Random Patches method.
bagReg = BaggingRegressor(LinearRegression(), n_estimators = 500, 
                           bootstrap_features = True, max_features = 1.0, oob_score = True, random_state = 0)

bagReg.fit(X_train_stan, y_train)
print("oob score is: ", bagReg.oob_score_) # The oob score is an estimate of the r2 score of the ensemble classifier
```


```python
from sklearn.metrics import r2_score
y_pred = bagReg.predict(X_test_stan)
print("The r2 score on the test set is: ", r2_score(y_test, y_pred))
```

## 예제 - RandomForestRegressor

이제 **RandomForestRegressor**를 사용해 보겠습니다. RandomForestClassifier와 달리 RandomForestRegressor는 기본적으로 각 분할에서 전체 특성 샘플을 탐색합니다. 그러나 여기서는 GridSearchCV를 사용하여 *max_features, max_depth* 및 *min_sample_leaf* 의 최적 조합을 탐색합니다.


```python
tuned_parameters = {'n_estimators':[500],'n_jobs':[-1], 'max_features': [0.5,0.6,0.7,0.8,0.9,1.0], 
                    'max_depth': [16,20,24],'min_samples_leaf':[1,10,100],'random_state':[0]} 

reg = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, scoring='r2')
reg.fit(X_train, y_train)
```


```python
print('The best model is: ', reg.best_params_)
print('This model produces a mean cross-validated score (r2) of', reg.best_score_)
```


```python
y_pred = reg.predict(X_test)
print("The r2 score on the test set is: ", r2_score(y_test, y_pred))
```

## 예제 - ExtraTreesRegressor
Similar to the Random Forest case, **ExtraTreesRegressor** differs from ExtraTreesClassifier in that the regressor by default considers all features to select the best split. But again, we will use GridSearchCV to search for the best combination of *max_features*, *max_depth* and *min_samples_leaf*.

랜덤 포레스트 예제와 유사하게 **ExtraTreesRegressor**는 기본적으로 최적 분할을 선택하기 위해 모든 특성을 고려한다는 점에서 ExtraTreeClassifier와 다릅니다. 그러므로 위와 유사하게 GridSearchCV를 사용하여 *max_features, max_depth 및 min_sample_leaf* 의 최적 조합을 탐색합니다.


```python
tuned_parameters = {'n_estimators':[500],'n_jobs':[-1], 'max_features': [0.5,0.6,0.7,0.8,0.9,1.0], 
                    'max_depth': [20,24,28],'min_samples_leaf':[1,10,100],'random_state':[0]} 

reg = GridSearchCV(ExtraTreesRegressor(), tuned_parameters, cv=5, scoring='r2')
reg.fit(X_train, y_train)
```


```python
print('The best model is: ', reg.best_params_)
print('This model produces a mean cross-validated score (r2) of', reg.best_score_)
```


```python
y_pred = reg.predict(X_test)
print("The r2 score on the test set is: ", r2_score(y_test, y_pred))
```

## 모델 성능 

| 분류기 이름| r2 점수 | Comments |
|------|------|------|
| 랜덤 패치 방식의 500개의 선형 회귀 분석기 | 0.35 | BaggingRegressor() |
| Random Forest with 500 trees (with GridSearchCV)  | 0.48| RandomForestRegressor() |
| Extra Trees with 500 trees (with GridSearchCV)| **0.51** | ExtraTreesRegressor() |

* 앙상블 선형 회귀 분석기의 성능이 가장 낮습니다. 선형으로 가정하면 이 문제에 적합하지 않을 수 있기 때문에 이는 충분히 예측 가능합니다.
* 랜덤 포레스트와 엑스트라 트리 모두 앙상블 선형 회귀 분석기보다 성능이 훨씬 우수하며, 엑스트라 트리는 랜덤 포레스트보다 성능이 약간 우수합니다.
* 그러나 만족스러운 r2 점수를 얻는 회귀 모델은 없습니다. 엑스트라 회귀 분석기의 y_test 및 y_pred를 그립니다.


```python
plt.plot(y_test,y_pred, linestyle='',marker='o')
plt.xlabel('true y values', fontsize = 14)
plt.ylabel('predicited y values', fontsize = 14)
plt.show()
```

해당 튜토리얼은 아래의 링크를 바탕으로 정리되었습니다.

- [Machine Learning 6 Basic Ensemble Learning](https://www.kaggle.com/code/fengdanye/machine-learning-6-basic-ensemble-learning)
