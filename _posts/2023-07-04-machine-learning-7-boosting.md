# Machine Learning 7 Boosting

이 튜토리얼에서는 <font color='blue'>부스팅</font> 기술에 대해 중점적으로 설명합니다.

**이전 튜토리얼과 비교하여 몇 가지 표기법이 변경되었습니다.**    
$m$ 은 이제 샘플 번호 대신 단계 번호 입니다.
따라서, $(m)$라는 초첨자는 더 이상 샘플에 레이블을 지정하지 않고 단계를 지정합니다.

**목차**
* 부스팅(Boosting) - 개요
* 에이다 부스트 (AdaBoost)
    * 알고리즘
        * 분류 모델
            * Binary-class AdaBoost
            * Multi-class AdaBoost (SAMME)
            * Real-valued SAMME
        * 회귀 모델
    * Sklearn 함수 및 예제
        * 예제 7.1
        * 예제 7.2
* 함수 공간에서의 수치 최적화
    - review - 파라미터 공간의 최적화
        - 경사 하강법
        - 뉴턴 방법
    - 함수 공간에서의 수치 최적화
        - 경사하강법
        - 뉴턴 방법
    - Finite data (유한 데이터)
        *  FSAM (Foward stage-wise additive modeling)
* Gradient boosting machine - GBM
    * 이론
        * 알고리즘
    * 그레디언트 트리 부스팅
        * 소개
        * 알고리즘
        * 정규화
        * Sklearn 함수 및 예제
            * 예제 7.3
            * 예제 7.4
* Newton boosting
    * 이론
        * 알고리즘
    * XGBoost
        * 소개
        * 정규화
        * 알고리즘
        * XGBoost 함수 및 예제
            * 예제 7.5
            * 예제 7.6
* 참고 문헌


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


```python
plt.rc('axes', lw = 1.5)
plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('xtick.major', size = 5, width = 3)
plt.rc('ytick.major', size = 5, width = 3)
```

# 부스팅(Boosting) - 개요

부스팅은 많은 __성능이 약한 훈련 모델(기본 모델 또는 기본 학습자라고도 함)__를 결합하여 강한 훈련 모델을 만들어 데이터를 맞추는 앙상블 학습 기술입니다. 이러한 성능이 약한 훈련 모델에는 결정 트리가 속합니다. 부스팅 알고리즘은 예측 오류를 최소화하기 위해 동일한 클래스의 $M$ 약한 훈련 모델에서 최적의 선형 조합을 찾으려고 합니다. 이러한 약한 훈련 모델은 __기본 함수__로 간주될 수 있습니다. 수식은 다음과 같습니다.

<center>
$F(x)=f_0(x)+\sum_{m=1}^{M}\theta_m \phi_m(x)$
</center>

여기서 $F(x)$는 $x$가 주어진 목표 값의 예측이고, $M$은 앙상블의 모델의 수이며, $f_0(x)$는 초기 추측입니다. $\phi_m(x)$는 약한 훈련 모델이고 $\theta_m$는 파라미터입니다.

부스팅 기술에는 많은 종류가 있습니다. (일반적으로) 가장 실용적인 부스팅 알고리즘은 <font color='blue'>Adaptive Boost(AdaBoost)</font>입니다. 다음으로, 함수 공간에서 기울기 하강에 해당하는 <font color='blue'>GBM(Gradient Boost)</font>과 함수 공간에서 뉴턴의 최적화 방법에 해당하는 <font color='blue'>XGBoost</font>가 있습니다.후자의 두 가지는 현재 상황에서 매우 인기가 있습니다. 계수뿐만 아니라 약한 훈련 모델의 파라미터에 대한 해결은 일반적으로 단계적으로 이루어집니다. 이 방식을 **FSAM (Forward Stagewise Addidtive Modeling)** 이라고 부릅니다. 이어지는 단원에서 위에서 언급한 모든 개념을 자세히 소개할 것입니다.

# 에이타 부스트 (AdaBoost)

Adaptive Boost(AdaBoost)는 한 번에 한 개의 약한 훈련 모델을 훈련시킵니다. __각 훈련 모델은 이전에 잘못 예측된 샘플에 더 큰 가중치를 할당하여 이전 훈련 모델보다 성능이 향상됩니다.__ 이러한 방식으로, 나중에 성능이 약한 훈련 모델들은 예측하기 어려운 표본에 더 집중합니다. 분류의 경우, 앙상블의 최종 예측은 모든 훈련 모델로부터 __가중치가 부여된 다수표__ 를 얻는 클래스입니다. 회귀의 경우 최종 예측은 모든 회귀 분석가가 수행한 예측의 __가중 중위수__입니다.

## 알고리즘

원래 AdaBoost는 이진 분류를 위해 개발되었습니다.([Freund and Schapire 1996](https://cseweb.ucsd.edu/~yfreund/papers/boostingexperiments.pdf)) 이후, 이 기술은 다중 클래스 분류([Zhu et al. 2006](https://hastie.su.domains/Papers/samme.pdf))와 회귀([Drucker 1997](https://pdfs.semanticscholar.org/8d49/e2dedb817f2c3330e74b63c5fc86d2399ce3.pdf))로 확장되었습니다. 이후에 그것들을 하나씩 살펴볼 것입니다.


### 분류 모델
#### Binary-class AdaBoost

훈련 세트에 $n$개의 샘플이 있다고 가정하면 알고리즘은 각 훈련 샘플에 대해 동일한 가중치로 초기화하는 것으로 시작합니다.

<center>
    $w_i=\frac{1}{n}, i=1,2,...,n$
</center>

그러므로:

* $m = 1$ ~ $M$인 경우:
    * 가중치 [$w_1$, $w_2$,…, $w_n$]를 사용하여 분류기 $T^{(m)}(x)$를 훈련 데이터를 이용하여 fit 합니다. 가중치가 높을수록 분류기는 해당 표본을 올바르게 추출합니다. 수학적으로 말해서, 성능이 약한 훈련 모델의 목표는 훈련 오류 $\mathrm{Pr}_{i \sim w_i}[y_i \neq T^{(m)}(x_i)]$를 최소화하는 $T^{(m)}(x)$를 찾는 것입니다. 이 오류는 훈련 모델에게 제공된 분포 {$w_i$}와 관련하여 측정됩니다.
    * 현재 분류기의 __가중 오류율__ 계산: $err^{(m)}=\sum_{i=1}^n w_i \mathrm{I}(y_i \neq T^{(m)}(x_i))/\sum_{i=1}^n w_i$. 여기서 $y_i$는 샘플 $x_i$의 실제 레이블입니다. $\mathrm{I}(y_i \neq T^{(m)}(x_i)$는 $y_i \neq T^{(m)}(x_i)$이면 1이고 그렇지 않으면 0입니다.
    * 앙상블 학습에서 __분류기 가중치__ 를 계산합니다: $\alpha^{(m)}=\mathrm{log}((1-err^{(m)})/err^{(m)})$. 오류율이 작을수록 분류기의 영향력은 커집니다.
    * __샘플 가중치를 업데이트__ 하여 잘못 분류된 샘플을 증가시킵니다: $w_i \leftarrow w_i \cdot exp[\alpha^{(m)}\cdot \mathrm{I}(y_i \neq T^{(m)}(x_i))$, $i=1,2,...,n$. 즉, 잘못 분류된 샘플은 ($\alpha^{(m)}>0$일 때) 더 큰 가중치를 갖습니다.
    * 샘플 가중치를 다시 정규화합니다. $w_i \leftarrow w_i/\sum_{i=1}^n w_i$.
* Output: $\hat{y}(x) = \underset{k}{\operatorname{argmax}}\sum_{m=1}^{M} \alpha^{(m)} \mathrm{I}(T^{(m)}(x) = k)$. 그 예측은 가중 다수결에 근거합니다. 여기서 $k=0$ 또는 $1$입니다.

장점은 __AdaBoost가 이전에 잘못 분류된 사례에 초점을 맞추기 위해 한 번에 하나의 분류기를 훈련시킨다는 것입니다. 오류율이 낮은 분류기는 최종 예측에서 가중치(더 많은 표)가 더 큽니다.__ 분명히 잘못 분류된 사례를 고르는 많은 방법이 있습니다. - 왜 알고리즘이 특별히 이런 방식으로 작성되었을까요? 나중 단원에서 이 알고리즘은 부스팅 알고리즘에서 사용되는 수 손실 함수를 사용한 FSAM 방식과 동일하다는 것을 확인할 수 있습니다. 현재로서는 위의 알고리즘을 이해하는 것으로 충분합니다.

#### Multi-class AdaBoost (SAMME)

__이진 클래스 알고리즘이 실제로 잘못 분류된 샘플에 대한 가중치를 증가시키려면 $\alpha^{(m)}$가 양수여야 한다는 것을 눈치챘을 수 있습니다.__ 즉, 오류율 $err^{(m)}$는 $1/2$보다 작아야 합니다. 이진 분류의 경우, 동일한 샘플 가중치를 갖는 무작위 오류율이 $1/2$이기 때문에 이를 충족하는 것은 어렵지 않습니다. 그러나 $K$ 클래스 분류($K>2$)의 경우, 무작위 오류율은 $(K-1)/K$가 되며, $\alpha^{(m)}$는 음수일 가능성이 더 높습니다. 이것은 이전 알고리즘이 다중 클래스 사례에서 잘 작동하지 않는 주요 이유입니다. 이 문제를 해결하기 위해 [Zhu et al. 2006](https://web.stanford.edu/~hastie/Papers/samme.pdf)은 $\alpha^{(m)}$의 표현을 다음과 같이 변경할 것을 제안했습니다.

<center>
    $\alpha^{(m)}=\mathrm{log}\frac{1-err^{(m)}}{err^{(m)}} + log(K-1)$
</center>

추가 항 $log(K-1)$은 오류율 $err^{(m)}$가 $(K-1)/K$보다 작은 한 $\alpha^{(m)}$가 양수인지 확인합니다. __이 알고리즘은 <font color='blue'>*SAMME*</font> 알고리즘이라고 합니다.__ *SAMME* 는 _다중 클래스 지수 손실 함수를 사용하는 Stagewise Additive Modeling(단계별 가산 모델링)_ 을 의미합니다. 차후에 그 이름이 어디에서 유래했는지 나중에 설명할 것입니다. $K=2$인 경우 SAMME는 이진 클래스 AdaBoost로 줄어듭니다. $K>2$일 때 $log(K-1) > 0$이 있는데, 이는 $\alpha^{(m)}$가 이진 클래스의 경우보다 크다는 것을 의미합니다. 결과적으로, 다중 클래스 AdaBoost는 잘못 분류된 샘플에 더 큰 불이익을 줍니다.

#### Real-valued SAMME

[Zhu et al. 2006](https://web.stanford.edu/~hastie/Papers/samme.pdf)은 또한 SAMME의 이산 오류율 대신 __실제 값 가중 확률 추정치__ 를 사용하여 모델을 업데이트하는 SAMME 알고리듬의 변형을 제안했습니다. 이 알고리즘은 종종 <font color='blue'>**SAMME.R**</font>이라고 합니다. __Skikitlearn의 AdaBoostClassifier()는 SAMME와 SAMME.R을 모두 지원합니다.__ SAMME.R이 어떻게 작동하는지에 대해서는 자세히 설명하지 않습니다. 다만,관심이 있다면, [Zhu et al. 2006](https://web.stanford.edu/~hastie/Papers/samme.pdf)은 알고리즘을 설명하는 원본 논문으로 참고 하시기 바랍니다.

### 회귀 모델

**AdaBoost 분류 모델의 주요 개념은 다음과 같습니다.**: 
* 각 반복에서 잘못 분류된 표본의 가중치가 증가했으며 예측 변수는 잘못 분류된 표본을 올바르게 파악하는 데 더 중점을 둡니다.
* 분류 모델들은 오류율이 낮을 경우 최종 예측에서 더 많은 발언권을 가집니다.  

**AdaBoost 회귀 모델은 분류 모델과 동일한 개념을 따릅니다.** 알고리즘은 다음과 같습니다.
* 가중치 $w_i=1/n$, $i=1,2,…, n$ 초기화
* 반복 번호 $m=0$을 초기화합니다.
* 평균 손실 $\bar{L}$(아래 정의) 0.5보다 작을 동안 반복합니다.
    * 반복 번호 $m = m + 1$ 증가.
    * __이 반복에 대한 훈련 세트를 구성하려면 대체할 $n$개의 샘플을 선택합니다. 각 선택에 대해 샘플 $x_i$가 선택될 확률은 $w_i$입니다.__
    * 중복 가능한 훈련 세트에서 기본 회귀 모델을 훈련합니다. 훈련된 회귀 모델은 샘플 $x_i$에서 $\hat{y}^{(m)}_i$를 예측합니다.
    * __각 훈련 샘플에 대한 손실__ 계산: $L_i^{(m)} = L(y_i, \hat{y}^{(m)}_i) = L(|y_i-\hat{y}^{(m)}_i|)$. **Skikitlearn의<font color='green'>AdaBoostRegressor()</font>** 의 기본값은  $L_i = |y_i-\hat{y}^{(m)}_i|/sup (|y_i-\hat{y}^{(m)}_i|)$이ㅂ니다. 여기서 $sup$은 상한 경계를 나타냅니다.
    * 이 회귀 분석기의 __평균 손실__ 계산: $\bar{L}^{(m)}=\sum_{i=1}^{n} w_iL_i^{(m)}$.  
    * 이 회귀자의 __신뢰도__ 를 계산: $\beta^{(m)}=\mathrm{log}((1-\bar{L}^{(m)})/\bar{L}^{(m)})$. 평균 손실이 작을수록 회귀 분석 모델은 보다 "신뢰할 수 있는" 것입니다. 이는 AdaBoost 분류 모델의 $\alpha^{(m)}$와 유사합니다.
    * __샘플 가중치__ 업데이트: $w_i \leftarrow w_i \cdot exp[\beta^{(m)} \cdot (L_i^{(m)}-1)]$, $i=1,2,...,n$. 분류 모델에서, 잘못 분류된 샘플 가중치에 $exp(\alpha^{(m)})$를 곱합니다. 회귀 모델에서는 $exp[\beta^{(m)} \cdot (L_i^{(m)}-1)]$를 곱합니다. 샘플 손실 $L_i^{(m)}$이 클수록 이 샘플은 "잘못 분류된" 것입니다.
    * 샘플 가중치를 다시 정규화합니다: $w_i \leftarrow w_i/\sum_{i=1}^n w_i$.
* 우리는 각각 이전보다 향상된 $M$ 훈련된 회귀 모델을 구했습니다.
* Output:
    * 이전step의 테스트 샘플 $x$ 및 $M$ 훈련된 회귀 모델이 주어지면, 우리는 각 회귀 모델로부터 예측 값을 얻습니다. $\hat{y}^{(1)}$, $\hat{y}^{(2)}$, ..., $\hat{y}^{(M)}$.
    * $\hat{y}^{(1)}$, $\hat{y}^{(2)}$, ..., $\hat{y}^{(M)}$의 가중 중위수가 $\beta^{(1)}$, $\beta^{(2)}$, ..., $\beta^{(M)}$의 최종 예측에 가중치를 부여합니다. 다음과 같이 가중 중위수를 얻을 수 있습니다.
        * $\hat{y}^{(1)} < \hat{y}^{(2)} < ... < \hat{y}^{(M)}$가 되도록 회귀 모델의 레이블을 다시 지정합니다. 가중치 $\beta^{(1)}$, $\beta^{(2)}$, ..., $\beta^{(M)}$에 따라 레이블을 다시 지정합니다.
        * $\sum_{i=1}^{m}\beta^{(i)}\geq 1/2 \cdot \sum_{i=1}^{M}\beta^{(i)}$를 만족하는 _가장 작은_ $m$에 도달할 때까지 $\beta$를 합산합니다.
        * $\hat{y}^{(m)}$는 앙상블의 최종 예측입니다.

각 반복은 이전보다 "어렵습니다". 즉, 후속 회귀 모델이 이전보다 훈련하기가 더 어렵다는 것을 의미합니다. 따라서 평균 손실 $\bar{L}$은 삽입과 함께 증가하는 경향이 있으며, 최종적으로 $\bar{L}$가 그 범위를 초과할 때 알고리즘이 종료됩니다. 위의 알고리즘은 [Drucker 1997](https://pdfs.semanticscholar.org/8d49/e2dedb817f2c3330e74b63c5fc86d2399ce3.pdf)을 기반으로 합니다. 논문을 읽으면, 분류 알고리즘과의 직관적인 비교를 돕기 위해 $\beta^{(m)}$ 및 $w_i$의 정의를 수정한 것을 알 수 있습니다. 하지만 실제로 여기서 설명하는 알고리즘은 논문의 알고리즘과 _동일_ 합니다.

__Sklearn의 <font color='green'>AdaBoostRegressor()</font> 는 이 알고리즘을 사용합니다.__

## Sklearn 함수 및 예제

Sklearn은 AdaBoost에 다음과 같은 함수를 제공합니다:

* **AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0, loss=’linear’, random_state=None)**
* **AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm=’SAMME.R’, random_state=None)**

**base_estimator** 는 각 반복에서 훈련된 추정기를 지정합니다. <font color='green'>AdaBoostRegressor</font>의 기본 추정기는 <font color='green'>DecisionTreeRegressor(max_depth=3)</font> 이고, <font color='green'>AdaBoostClassifier</font> 의 기본 추정기는 <font color='green'>DecisionTreeClassifier(max_depth=1)</font>입니다. 말해두자면, 기본 트리는 비교적 얕습니다. 이는 트리가 약한 학습자(성능이 약한 훈련 모델) 역할을 하며, 개별 약한 학습자가 데이터를 너무 잘 학습하는 것을 원하지 않기 때문입니다(과대적합). __이 얕은 나무들은 스스로 높은 편향을 가질 수 있지만, 부스팅 기술은 약한 학습자들의 앙상블이 잘 수행되도록 합니다.__

**n_estimators**는 반복 횟수(따라서 모델의 갯수)이며, 이는 위 알고리즘에서  $M$에 해당합니다.

**learning_rate** can shrink the contribution of each base estimator. If we denote learning rate as $\eta$, then:

**learning_rate**는 각 기본 추정기의 기여도를 줄일 수 있습니다. 학습 률을 $\eta$로 표시하며, 이는 각 추정기에서 다음과 같습니다.

* AdaBoost 분류 모델에서, $\alpha^{(m)}=\eta \cdot \mathrm{log}((1-err^{(m)})/err^{(m)})$
* AdaBoost 회귀 모델에서, $\beta^{(m)}=\eta \cdot \mathrm{log}((1-\bar{L}^{(m)})/\bar{L}^{(m)})$  

기본적으로 학습률은 $1.0$입니다.

### 예제 7.1

이 예제에서는 Sklearn의 <font color='green'>AdaBoostClassifier()</font>를 사용해 보겠습니다. 이전 튜토리얼과 마찬가지로 레드 와인 품질 데이터 세트를 사용합니다. 먼저 데이터를 로드합니다.


```python
data = pd.read_csv('../input/winequality-red.csv')
data['category'] = data['quality'] >= 7 # again, binarize for classification
data.head()
```

그런 다음 데이터 세트를 훈련 세트와 테스트 세트로 분할합니다.


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

이 기본적인 AdaBoost 분류기가 어떻게 작동하는지 살펴보겠습니다. AdaBoost에서 예측 클래스 확률은 __앙상블의 각 분류기의 가중 평균 예측 클래스 확률입니다.__


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

이 AdaBoost 분류기의 AUC는 허용 가능하지만, 실제로 최상은 아닙니다. 이제 추정기의 가중치와 오류를 출력하여 분류기가 수행하는 작업을 자세히 살펴보겠습니다.


```python
print(adaClf.estimator_weights_)
```


```python
print(adaClf.estimator_errors_)
```

오차가 증가했다가 0.45 전 후로 변동하고, 그 결과 추정기 가중치가 감소했다가 0.10 전후로 변동하는 것으로 보입니다. 다음 추정치가 이전 추정치보다 향상되었다고 생각할 수 있기 때문에 이것은 직관적이지 않습니다. 각 추정치 예측을 출력하여 이 관측치를 이해해 보겠습니다. __공간을 절약하기 위해, 우리는 처음 20개의 추정치와 그들이 처음 10개의 훈련 샘플(1119개의 샘플 중)에 대해 정확하게 예측했는지 여부를 출력할 것입니다.__


```python
for estimator in adaClf.estimators_[:20]:
    y_pred = estimator.predict(X_train)
    correct = y_train==y_pred
    print(correct[:10]) # print only the first 10 training samples to save space
```

이제 우리는  **<font color='blue'>때때로 AdaBoost가 이전에 잘못 분류된 사례를 수정하려고 할 때 이전의 정확한 예측에 잘못을 범한다는 것을 알 수 있습니다</font>**. 예를 들어, __위의 셀에서 두 번째 반복이 첫 번째 반복에서 두 개의 잘못된 예측을 수정했지만 이전에 올바른 네 개의 표본을 잘못 분류했음을 알 수 있습니다.__ 이는 깊이 1의 결정 트리 기반 훈련 모델의 일반적인 현상으로, 트리가 하나의 기능과 하나의 임계값을 기준으로 교육 세트를 두 개로 분할할 때 잘못를 범하도록 설정되어 있기 때문입니다. 다행인 것은 오류율이 높으면 알고리즘이 낮은 추정기 가중치를 할당하고 따라서 그 추정기가 결과를 결정하는데 큰 영향을 미치지 않는다는 것입니다.

**<font color='blue'>더 큰 잠재적인 문제는 때때로 훈련 세트에 특이치, 노이즈 또는 일반적으로 정확하게 예측하기 어려운 일부 표본이 포함된다는 것입니다. 이러한 샘플의 가중치는 기하 급수적으로 증가할 것이며 알고리즘은 이러한 샘플에만 거의 집중하도록 강요되어 모델이 이러한 샘플에 과대적합되도록 합니다. 결과적으로, 모델은 이러한 "어려운" 예제를 정확하게 맞았을 수 있지만, 이전에 정확했던 예측은 올바르지 않게 됩니다</font>**. 이러한 문제가 의심되면 훈련 샘플을 검사하여 특이치가 있는지 확인하는 것이 좋습니다.

이 AdaBoost 분류기가 [튜토리얼 5](https://www.kaggle.com/code/cosmosjung/machine-learning-5-random-forests)의 랜덤 포레스트 분류기보다 성능이 떨어지는 이유는 무엇입니까? (확인되지 않은) 추측은 다음과 같습니다.

* 이 AdaBoost 분류기에는 50개의 결정 트리가 충분하지 않습니다. 랜덤 포레스트 분류기에는 500개의 모델이 있습니다.
* 학습률이 너무 커서, 잘못 분류된 샘플의 가중치가 너무 빠르게 증가합니다. 따라서 AdaBoost는 가중치가 큰 샘플에 쉽게 과적합됩니다. 랜덤 포레스트에는 이 문제가 없습니다.
* 훈련 세트에는 AdaBoost의 모든 관심을 끄는 특이치/어려운 샘플이 있습니다. 랜덤 포레스트에는 이 문제가 없습니다. 랜덤 포레스트에는 과대적합을 방지하기 위해 각 트리에 대해 랜덤성을 도입하는 방법도 있습니다.
* SAMME.R 알고리즘은 SAMME보다 성능이 우수합니다. (그러나 SAMME.R을 사용할 때는 의미 있는 추정기 가중치를 출력할 수 없습니다)

추가적으로 랜덤 포레스트가 배깅과 특성 서브 샘플링을 통해 다른 트리에 _무작위성_ 을 도입하는 반면, 에이다부스트는 트리 간의 일부 유형에 다양성을 도입하지만 이전에 잘못 분류된 사례에 _초점_ 을 맞추고 덜 정확한 분류 모델에는 불이익을 준다는 것에 대해 흥미롭게 생각해볼 수 있습니다.

### 예제 7.2

이 예제에서는 Sklearn의 <font color='green'>AdaBoostRegressor()</font>를 사용해 보겠습니다. 입력은 11가지 특성이 되고 출력은 와인 품질 값이 됩니다.


```python
X = data[data.columns[0:11]].values
y = data['quality'].values.astype(np.int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

print('X train size: ', X_train.shape)
print('X test size: ', X_test.shape)
print('y train size: ', y_train.shape)
print('y test size: ', y_test.shape)
```

기본 AdaBoost Regressor를 사용해 보겠습니다.


```python
adaReg = AdaBoostRegressor(random_state=3)
adaReg.fit(X_train, y_train)
```

그런 다음 테스트 세트에서 성능을 평가합니다.


```python
y_pred = adaReg.predict(X_test)
plt.plot(y_test, y_pred, linestyle='', marker='o')
plt.xlabel('true values', fontsize = 16)
plt.ylabel('predicted values', fontsize = 16)
plt.show()
print('The r2_score on the test set is: ',r2_score(y_test, y_pred))
```

이 성능은 [지난 튜토리얼](https://www.kaggle.com/code/cosmosjung/machine-learning-6-basic-ensemble-learning)에서 사용한 "랜덤 패치 기법을 사용한 500개의 선형 회귀" 모델보다 우수합니다. 그러나 r2_score의 경우 랜덤 포레스트와 엑스트라 트리가 더 나은 선택으로 남아 있습니다. 500그루의 트리로 에이다 부스트 회귀 모델을 시도해볼 수 있는데, 랜덤 포레스트와 엑스트라 트리만큼 성능이 좋지는 않을 것입니다. __이 데이터 세트의 경우 부정확한 예측에 초점을 맞추는 것보다 추정기 사이에 무작위성을 도입하는 것이 더 나을 수 있습니다. (예: 데이터 세트에 AdaBoost를 잘못된 방향으로 이끄는 특이치가 있을 수 있음)__

# 함수 공간에서의 수치 최적화

그레디언트 부스팅([Friedman 1999](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf))과 XGBoost([Chen and Guest 2016](https://arxiv.org/pdf/1603.02754.pdf))는 에이다부스트 이후에 나왔습니다. 이 기술들은 오늘날의 데이터 과학에서 가장 인기 있는 기술 중 하나입니다. 이 단원에서는 __그레디언트 부스팅과 XGBoost를 이해하는 데 큰 도움이 되는__ <font color='blue'>함수</font> 공간에서의 수치 최적화의 주요 방법에 대해 살펴보겠습니다.

튜토리얼의 이 부분은 [Friedman 1999](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)과 [Nielsen 2016](https://brage.bibsys.no/xmlui/bitstream/handle/11250/2433761/16128_FULLTEXT.pdf)을 기반으로 합니다.

## Review - 파라미터 공간의 최적화

파라미터 공간의 최적화 개념을 검토하는 것으로 시작하겠습니다. 가지고 있는 데이터($x$,$y$)가 분포 $f(x,y)$에 이어 연속적이라고 가정합니다. $x$의 반응($\hat{y}$)을 예측하는 파리미터화된 모델 $F(x;\theta)$를 정의합니다. 따라서, 모델의 "비용"을 $\theta$ 함수로 정의할 수 있습니다.

<center>
 $\Phi(\theta) = E_{x,y}[L(y, F(x; \theta))] \equiv \int_x \int_y L(y, F(x; \theta)) f(x,y)dx dy$    
</center>

여기서 $f(x,y)$는 확률 밀도 함수입니다. 우리는 단계적인 방식으로 $\Phi(\theta)$를 최소화하는 $\theta$의 값을 결정할 것입니다.

<center>
 $\theta = \sum_{m=0}^{M} \theta_m$    
</center>

$m$의 반복에서 $\theta$는 다음과 같이 갱신됩니다. $\theta^{(m)}=\theta^{(m-1)}+\theta_m$.$\theta_0$은 초기 추측입니다. 아래에서는 $\theta$를 갱신하는 __두 가지 방법__ 에 대해 논의합니다.

### 경사 하강법

각 반복 $m$에서, 우리는 $\theta$와 관련하여 $\Phi(\theta)$의 __음의 기울기__ 를 따라 한 단계 나아갑니다. 이러한 단계는 가장 가파른 하강 방향이며, 합리적인 단계 길이를 고려할 때 비용을 절감할 수 있습니다. 반복 $m$에서 갱신하기 전에 음의 기울기는 다음과 같습니다.

<center>
    $-g_m = -\frac{\partial \Phi(\theta)}{\partial \theta} \mid_{\theta = \theta^{(m-1)}}$
</center>

step 길이 $\rho_m$는 일반적으로 <font color='blue'>line search</font>을 통해 결정됩니다.

<center>
    $\rho_m =  \underset{\rho}{\operatorname{argmin}} \Phi(\theta^{(m-1)}-\rho g_m)$
</center>

따라서 반복 $m$에서 수행되는 단계는 다음과 같습니다.

<center>
    $\theta_m = -\rho_m g_m$
</center>

경사 하강법은 1차 함수 형태이며, $\Phi(\theta)$만 미분 가능하면 됩니다.

### 뉴턴 방법

뉴턴의 방법은 다음과 같이 이해할 수 있습니다.
반복 $m$에서 업데이트하기 전에 $\theta = \theta^{(m-1)}$가 있습니다. $\theta$의 _이상적인_ 업데이트는 다음을 충족해야 합니다.

<center>
    $\frac{\partial \Phi(\theta)}{\partial \theta} \mid _{\theta = \theta^{(m-1)}+\theta_m} = 0$
</center>

즉, 업데이트는 $\Phi(\theta)$를 직접 최소화합니다. 말하자면, 아래와 같이 서술할 수 있습니다.

<center>
    $\frac{\partial \Phi(\theta)}{\partial \theta} \mid _{\theta = \theta^{(m-1)}+\theta_m} = \frac{\partial \Phi(\theta^{(m-1)}+\theta_m)}{\partial (\theta^{(m-1)}+\theta_m)} = \frac{\partial \Phi(\theta^{(m-1)}+\theta_m)}{\partial (\theta_m)}$
</center>

따라서 각 단계에서 해결하고자 하는 사항은 다음과 같습니다.

<center>
    $\frac{\partial \Phi(\theta^{(m-1)}+\theta_m)}{\partial (\theta_m)} = 0$
</center>

우리는 2차 [Taylor expansion](https://en.wikipedia.org/wiki/Taylor_series)을 사용하여 $\Phi(\theta^{(m-1)}+\theta_m)$를 확장함으로써 위 방정식을 _대략적으로_ 해결할 수 있습니다.

<center>
    $\Phi(\theta^{(m-1)}+\theta_m) \approx \Phi(\theta^{(m-1)})+  \frac{\partial \Phi(\theta)}{\partial \theta}\mid_{\theta=\theta^{(m-1)}} \theta_m + \frac{1}{2!} \frac{\partial^2 \Phi(\theta)}{\partial \theta ^2}\mid_{\theta=\theta^{(m-1)}}\theta_m^2$
</center>

$g_m = \frac{\partial \Phi(\theta)}{\partial \theta}\mid_{\theta=\theta^{(m-1)}}$, $h_m = \frac{\partial^2 \Phi(\theta)}{\partial \theta ^2}\mid_{\theta=\theta^{(m-1)}}$로 각각 $g_m, h_m$을 나타내자. 그러면 아래와 같이 서술할 수 있다:

<center>
    $\frac{\partial \Phi(\theta^{(m-1)}+\theta_m)}{\partial (\theta_m)} \approx g_m + h_m\theta_m= 0$
</center>

따라서 반복 $m$에서의 업데이트는 다음과 같습니다.

<center>
    $\theta_m = - g_m/h_m$
</center>

뉴턴의 방법은 2차 방정식 형태이며, $\Phi(\theta)$가 두 번 미분 가능해야 합니다.

## 함수 공간에서의 수치 최적화

함수 공간에서 __각 $x$에서 예측된 $F(x)$를 최적화할 "파라미터"로 간주합니다.__ 그러면 다음과 같이 서술 할 수 있습니다.

<center>
 $\Phi(F) = E_{x,y}[L(y, F(x))] \equiv \int_x \int_y L(y, F(x)) f(x,y)dx dy = \int_x \left[ \int_y L(y, F(x)) f(y|x)dy \right]f(x)dx \equiv E_{x}\left[E_y \left(L(y, F(x))\mid x \right)\right]$    
</center>

따라서 $\Phi(F)$를 최소화하는 것은 다음을 최소화하는 것과 같습니다.

<center>
 $\phi(F) = E_y \left[L(y, F(x)) \mid x\right]$  
</center>

__각각의 $x$에 대해서__.  

파라미터 공간의 최적화와 유사하게 다음을 정의합니다.

<center>
 $F(x)=\sum_{m=0}^{M}f_m(x)$  
</center>

and

<center>
 $F^{(m)}(x) = F^{(m-1)}(x)+f_m(x)$  
</center>

**각각의 $x$에 대해서**. 여기서 $f_0(x)$는 초기 추측입니다.

### 경사 하강법

이제 "파라미터"는 각 $x$에 대한 $F(x)$가 됩니다. $F$에 대한 $\phi(F)$의 기울기는 다음과 같습니다.

<center>
 $g_m(x) = \left[ \frac{\partial \phi(F(x))}{\partial F(x)} \right]_{F(x)=F^{(m-1)}(x)}= \left[ \frac{\partial E_y \left[L(y, F(x)) \mid x\right]}{\partial F(x)} \right]_{F(x)=F^{(m-1)}(x)} =  E_y \left[ \frac{\partial L(y, F(x))}{\partial F(x)}\mid x  \right]_{F(x)=F^{(m-1)}(x)}$  
</center>

미분과 적분 사이의 상호작용은 충분한 규칙성이 있다는 가정에 기초합니다.손실 함수는 일반적으로 이 요구 사항을 충족하므로 예외에 대해 크게 걱정할 필요가 없습니다.

각 반복 $m$에서 업데이트는 다음과 같습니다.

<center>
$f_m(x) = -\rho_m g_m(x)$. 
</center>

step 길이 $\rho_m$는 라인 탐색(line search)을 통해 제공됩니다.

<center>
$\rho_m = \underset{\rho}{\operatorname{argmin}} E_{x,y} \left[ L(y, F^{(m-1)}(x)-\rho g_m(x)) \right]$
</center>

### 뉴턴 방법

파라미터 공간에서 수행한 것과 유사하게 각 반복 $m$ 및 각 $x$에서 다음에 대해 해결하고자 합니다.

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


따라서, $\frac{\partial}{\partial (f_m(x))} E_y \left[ L(y, F^{(m-1)}(x) + f_m(x)) \mid x \right]= 0$ 에 대한 해는 다음과 같이 근사합니다.

<center>
    $f_m(x) = - g_m(x)/h_m(x)$
</center>

## Finite data (유한 데이터)
간단한 요약으로서, 함수 공간의 수치 최적화는 다음과 같이 표현됩니다.

<center>
 $F(x)=\sum_{m=0}^{M}f_m(x) = f_0(x) + \sum_{m=1}^{M}f_m(x)$  
</center>

그러나, 위의 최적화 논의는 모두 분포 $f(x,y)$가 연속 데이터 $(x,y)$가 있다는 가정에 기초합니다. __실제 데이터에서는 유한 샘플 ${x_i, y_i}$, $i=1,2,...,n$만 있습니다.__ ${x_i}$ 이외의 $x$ 값에 대해서는 $-\rho_m g_m(x)$ 또는 $-g_m(x)/h_m(x)$를 사용하여 $F(x)$를 직접 업데이트할 수 없습니다. ($E_y \left[...\mid x \right]$는 훈련 포인트 외부의 $x$ 값으로 추정할 수 없기 때문) __최적화가 작동하려면 모델에 대한 기본적인 가정이 필요합니다.__ 일반적으로 파라미터화된 형태를 $f_m(x)$로 가정할 수 있습니다.

<center>
 $F(x)=\sum_{m=0}^{M}f_m(x) = f_0(x) + \sum_{m=1}^{M}\theta_m \phi(x;a_m)$
</center>

부스팅의 맥락에서 $\phi(x;a_m)$는 $a_m$로 매개 변수화된 약한 학습자 클래스이며, $\theta_m$는 약한 학습자 앞에 있는 해당 계수입니다. 약한 학습자의 클래스 가정은 특정 공간에서 $F(x)$를 제한합니다. 약한 학습이 결정 트리로 선택되면 $a_m$는 트리의 구조와 리프 노드 $j$의 가중치 $w_j$를 설명합니다. 결정 트리에 대한 자세한 내용은 [튜토리얼 5](https://www.kaggle.com/code/cosmosjung/machine-learning-5-random-forests) 부록을 참조하십시오. **<font color='blue'>부스팅 알고리즘은 특정 클래스의 약한 학습자로 제한된 단계와 함께 함수 공간의 최적화 문제로 간주될 수 있습니다</font>**. 목표 함수는 다음과 같습니다.

각 $x_i$, $i=1,2,...,n$에 대해

<center>
    $\hat{\Phi}(F) = \sum_{i=1}^n L(y_i, F(x_i))$
</center>

또는 다음과 같이 표현합니다.

<center>
    $\hat{\phi}(F(x_i)) = L(y_i, F(x_i))$
</center>

### FSAM (Foward stage-wise additive modeling)

대부분의 부스팅 알고리즘은 단계별(stage-wise) 방식으로 해결됩니다. 각 반복 $m$에서  다음을 해결합니다.

<center>
$\{\theta_m, a_m\} = \underset{\{\theta, a\}}{\operatorname{argmin}} \sum_{i=1}^n L(y_i, F^{(m-1)}(x_i)+\theta \phi(x_i; a))$
</center>

약한 학습자 계층을 고려할 때, 이 방법을 __FSAM__ 이라고 합니다. <font color='blue'>AdaBoost 알고리즘은 <font color='blue'>$\phi$가 출력 $-1$ 또는 $1$인 분류기라는 제약 하에 지수 손실 함수 $L(y, F) = exp(-yF)$</font>([Friedman et al. 2000](https://web.stanford.edu/~hastie/Papers/AdditiveLogisticRegression/alr.pdf))<font color='blue'>에 대해 위의 방정식을 정확하게 푸는 것과 동등하다는 것이 입증되었습니다. 반면에, 경사 부스팅과 뉴턴 부스팅은 각각 함수 공간에서 경사 하강법과 뉴턴의 방법을 통해 위의 방정식을 대략적으로 해결합니다. 그레이디언트 부스팅 및 뉴턴 부스팅의 경우 미분 가능(그라디언트 부스팅) 또는 두 번 미분 가능(뉴턴 부스팅)한 한 다양한 손실 함수를 사용할 수 있습니다.</font>

# Gradient boosting machine (GBM)
## 이론

그레이디언트 부스팅은 [1999년 프리드먼](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)에 의해 제안되었습니다. __이 방법은 함수 공간의 경사 하강에 해당합니다.__ 유한한 훈련 세트 ${x_i, y_i}$가 주어지면, 각 샘플 포인트와 반복 $m$에서 $F$에 대한 목표 함수의 기울기는 다음과 같습니다.

<center>
    $g_m(x_i) = \left[ \frac{\partial \hat{\phi}(F(x_i)) }{\partial F(x_i)} \right]_{F(x_i)=F^{(m-1)}(x_i)} = \left[ \frac{\partial L(y_i, F(x_i)) }{\partial F(x_i)} \right]_{F(x_i)=F^{(m-1)}(x_i)}$
</center>

여기서 $i=1,2,…,n$. 다시 말하자면 FSAM에서, 우리는 목적 함수를 최소화하는 함수 공간 $f_m(x_i) = \theta_m \phi(x_i;a_m)$의 단계를 찾고자 한다는 것을 기억하십시오.

<center>
$\{\theta_m, a_m\} = \underset{\{\theta, a\}}{\operatorname{argmin}} \sum_{i=1}^n L(y_i, F^{(m-1)}(x_i)+\theta \phi(x_i; a))$
</center>

함수 공간에서의 경사 하강법을 사용하여 __벡터 ${ \phi(x_i; a_m)}{i=1}^n$의 방향이 가능한 한 음의 기울기 ${-g_m(x_i)}{i=1}^n$과 정렬되기를 원합니다.__ 물론, 이상적으로는 ${ \phi(x_i; a_m)}{i=1}^n$이 ${-g_m(x_i)}{i=1}^n$과 정확히 같은 방향에 있기를 원합니다. 그러나 $\phi(x_i;a_m)$는 약한 학습자 모델에 의해 제한되기 때문에 ${\phi(x_i;a_m)}_{i=1}^n$을 음의 기울기에 가장 잘 맞추는 파라미터 $a_m$만 찾을 수 있습니다. 이는 다음을 통해 얻을 수 있습니다.
 
<center>
$a_m =  \underset{\{\beta, a\}}{\operatorname{argmin}} \sum_{i=1}^{n} \left[ \left( -g_m(x_i) \right) - \beta \phi(x_i;a) \right] ^2$
</center>

최소화 하는 단계는 경사 하강을 위한 제한된 단계 방향을 찾는 것으로 간주할 수 있습니다. step 길이 $\rho_m$은 _라인 탐색(line search)_ 을 통해 얻을 수 있습니다.

<center>
$\rho_m =  \underset{\rho}{\operatorname{argmin}} \sum_{i=1}^n L(y_i, F^{(m-1)}(x_i)+\rho \phi(x_i; a_m))$
</center>

[Friedman 1999](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)에서, 마지막 단계는 다음과 같습니다.

<center>
$f_m(x) =  \eta \rho_m \phi(x;a_m)$
</center>

여기서 $0 < \eta \leq 1$은 학습률입니다. 학습률은 각 반복에서 step 길이를 축소하여 알고리즘에 정규화를 도입할 수 있습니다.

### 알고리즘
일반적인 그레디언트 부스팅 알고리즘은 다음과 같습니다.

* 초기화 : $F^{(0)}(x) = f_0(x) = \underset{\theta}{\operatorname{argmin}}\sum_{i=1}^{n}L(y_i, \theta)$
* $m=1,2,...,M$에 대해 다음을 수행합니다.

    * $g_m(x_i) = \left[ \frac{\partial L(y_i, F(x_i)) }{\partial F(x_i)} \right]_{F(x_i)=F^{(m-1)}(x_i)}$, $i=1,2,...,n$
    * $a_m =  \underset{\{\beta, a\}}{\operatorname{argmin}} \sum_{i=1}^{n} \left[ \left( -g_m(x_i) \right) - \beta \phi(x_i;a) \right] ^2$
    * $\rho_m =  \underset{\rho}{\operatorname{argmin}} \sum_{i=1}^n L(y_i, F^{(m-1)}(x_i)+\rho \phi(x_i; a_m))$
    * $f_m(x) =  \eta \rho_m \phi(x;a_m)$
    * $F^{(m)}(x) = F^{(m-1)}(x) + f_m(x)$
    
* 예측 :
    * $x$가 주어지면 $F(x) = F^{(M)}(x) = \sum_{m=0}^{M}f_m(x)$를 예측합니다.

## 그레디언트 트리 부스팅
### 소개

__그레디언트 트리 부스팅은 결정 트리를 기본 모델로 하는 그레디언트 부스팅의 특정 사례입니다.__ 그것은 현재 여러 문제에서 매우 일반적으로 사용되며 보통 만족스러운 성능을 제공합니다. 계속하기 전에 [튜토리얼 5](https://www.kaggle.com/code/cosmosjung/machine-learning-5-random-forests)의 부록을 읽어야 합니다. [튜토리얼 5](https://www.kaggle.com/code/cosmosjung/machine-learning-5-random-forests)의 부록에서 결정 트리는 다음과 같이 표현될 수 있다는 것을 배웠습니다.

<center>
$f(x) = \sum_{j=1}^{T} w_j {\mathrm I}(x\in R_j)$
</center>

여기서 $w_j$는 j번째 리프 노드의 __가중치__ 라고 하며, ${R_j}(j=1,2,...,T)$는 트리의 __구조__ 라고 합니다. $T$는 이 트리의 리프 노드 수입니다. ${\mathrm I}(x\in R_j)$는 샘플 $x$가 영역 $R_j$에 속하면 1이고, 그렇지 않으면 0입니다. 결정 트리를 기본 모델로 사용하여 $F(x)$의 가산 모델을 다음과 같이 표현할 수 있습니다.

<center>
 $F(x)=f_0(x) + \sum_{m=1}^{M}f_m(x) = f_0(x) + \sum_{m=1}^{M} \sum_{j=1}^{T_m}w_{jm}{\mathrm I}(x\in R_{jm})$  
</center>

__일반 GBM 알고리즘에서 수행한 것과 마찬가지로 각 반복 $m$에서 기본 훈련 모델의 트리를 음의 그레이디언트에 맞추려고 합니다. 이러한 정렬 비용은 다시 손실 제곱 합입니다.__

<center>
$J_m = \sum_{i=1}^{n} \left[ \left(-g_m(x_i)\right) - \left( \sum_{j=1}^{T_m} w_{jm}{\mathrm I}(x\in R_{jm}) \right) \right]^2 = const +  \sum_{i=1}^{n} \left[ 2g_m(x_i) \sum_{j=1}^{T_m} w_{jm}{\mathrm I}(x\in R_{jm}) +  \sum_{j=1}^{T_m} w_{jm}^2{\mathrm I}(x\in R_{jm}) \right]$
</center>

위의 합계 오버 샘플(sum over samples)은 리프 노드 위의 합계로 대체할 수 있습니다.

<center>
$J_m = const +  \sum_{j=1}^{T_m} \sum_{i; \ x_i \in R_{jm}}\left[ 2g_m(x_i) w_{jm} +w_{jm}^2 \right] = const +  \sum_{j=1}^{T_m} \left[ 2 \left( \sum_{i; \ x_i \in R_{jm}} g_m(x_i) \right) w_{jm} +  \sum_{i; \ x_i \in R_{jm}}w_{jm}^2\right]$
</center>

$G_{jm} = \sum_{i; \x_i \in R_jm} g_m(x_i)$로 정의한다면, 영역 $R_{jm}$(리프 노드 $j$)에 속하는 샘플 수를 $n_{jm}$로 정의하고, $J_m$의 상수를 버리고, $J_m$를 2로 나눈다면(이 중 하나를 선택하면 최적화 결과가 변경됩니다), 비용은 다음과 같이 표현할 수 있습니다.

<center>
$J_m =  \sum_{j=1}^{T_m} \left[  G_{jm} w_{jm} +  \frac{1}{2}n_{jm}w_{jm}^2\right]$
</center>

이 비용 함수는 결정 트리를 훈련하는 데 사용됩니다.

트리를 훈련하기 위해, 우리는 [튜토리얼 5](https://www.kaggle.com/code/cosmosjung/machine-learning-5-random-forests)의 부록에 소개된 CART 알고리즘을 따릅니다. __알고리즘은 먼저 고정된 구조로 주어진 무게를 우선 결정하고, 그 다음에 주어진 무게로 구조를 학습합니다.__ 고정 구조에서 $G_{jm}$ 및 $n_{jm}$가 고정되므로, 가중치는 비용을 최소화하는 값으로 제공됩니다.

<center>
    $w_{jm}^{*} = -\frac{G_{jm}}{n_{jm}}$
</center>
여기서 $j=1,2,...,T_m$..

비용 함수에 가중치를 다시 연결하면 다음과 같은 이점을 얻을 수 있습니다.

<center>
    $J_m^* = -\frac{1}{2} \sum_{j=1}^{T_m} \frac{G_{jm}^2}{n_{jm}}$
</center>

따라서 트리 훈련 중에 발생할 수 있는 분할을 고려할 때 분할 가능성은 다음과 같습니다.

<center>
${\mathrm {Gain}}=J_{\mathrm {before}} - J_{\mathrm {after}} = L_k^* - (L_L^*+L_R^*) = \frac{1}{2} \left[ \frac{G_L^2}{n_L} + \frac{G_R^2}{n_R} - \frac{G_{km}^2}{n_{km}} \right]$ 
</center>

여기서 $k$는 분할되는 노드입니다.

위의 단계 이후, 트리는 반복 $m$의 기본 학습자가 되도록 훈련되었습니다.일반적인 GBM 알고리즘에서 최적의 step 길이 $\rho_m$를 탐색하는 라인 탐색 단계가 있음을 기억하십시오. __그레이디언트 _트리_ 부스팅의 경우, [Friedman 1999](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)는 하나의 일반적인 라인 탐색을 수행하는 대신 각 리프 노드에 하나씩 $T_m$ 라인 탐색 단계를 수행할 수 있다고 제안했습니다. 이는 가중치 $w_{jm}$에 대한 최종 업데이트와 같습니다.__

<center>
$\{ w_{jm} \}_{j=1}^{T_m} =  \underset{\{ w_{j} \}_{j=1}^{T_m}}{\operatorname{argmin}} \sum_{i=1}^{n} L\left(y_i, F^{(m-1)}(x_i) + \sum_{j=1}^{T_m}w_j {\mathrm I}(x_i\in R_{jm})\right)$
</center>

<font color='blue'>라인 탐색 단계는 $m$번째 반복에서 예측 비용을 최소화하고 있으며 결정 트리를 음의 기울기에 맞추는 비용과는 별개입니다.</font>
위의 최소화 문제는 $T$ 독립적인 최소화 문제로 나눌 수 있습니다.

<center>
$w_{jm} =  \underset{w_{j}}{\operatorname{argmin}} \sum_{i;\ x_i \in R_{jm}} L\left(y_i, F^{(m-1)}(x_i) + w_j\right)$
</center>
for $j=1,2,...,T_m$.

그레이디언트 트리 부스팅 알고리즘은 아래에 요약되어 있습니다.



### 알고리즘
* 초기화 : $F^{(0)}(x) = f_0(x) = \underset{\theta}{\operatorname{argmin}}\sum_{i=1}^{n}L(y_i, \theta)$
* $m=1,2,...,M$에 대해 다음을 수행합니다.
    * $g_m(x_i) = \left[ \frac{\partial L(y_i, F(x_i)) }{\partial F(x_i)} \right]_{F(x_i)=F^{(m-1)}(x_i)}$, $i=1,2,...,n$
    * 결정 트리를 음의 기울기에 맞게 훈련합니다. 이득이 최대가 되도록 하는 분할 구간을 선택하는 트리 구조를 $\{R_{jm}\}_{j=1}^{T_m}$ 트리에서 결정합니다. $\frac{1}{2} \left[ \frac{G_L^2}{n_L} + \frac{G_R^2}{n_R} - \frac{G_{km}^2}{n_{km}} \right]$.
    * 학습된 구조의 잎 가중치를 결정합니다: $w_{jm} =  \underset{w_{j}}{\operatorname{argmin}} \sum_{i;\ x_i \in R_{jm}} L\left(y_i, F^{(m-1)}(x_i) + w_j\right)$, 여기서$j=1,2,...,T_m$.
    * $f_m(x) = \eta \sum_{j=1}^{T_m}w_{jm} {\mathrm I}(x\in R_{jm})$, 여기서 $\eta$는 학습률 입니다.
    * $F^{(m)}(x) = F^{(m-1)}(x) + f_m(x)$
* 예측 :
    * $x$가 주어지면 $F(x) = F^{(M)}(x) = \sum_{m=0}^{M}f_m(x)$를 예측합니다.

### 정규화

그레디언트 트리 부스팅 모델을 정규화하는 두 가지 일반적인 방법이 있습니다.
1. __트리를 규제합니다__. 예를 들어 허용되는 최대 리프 노드 수를 설정할 수 있습니다. $M$ 앙상블의 총 트리 수를 제한할 수도 있습니다.
2. __랜덤 서브 샘플링__. [지난 튜토리얼](https://www.kaggle.com/code/cosmosjung/machine-learning-6-basic-ensemble-learning)에서 소개한 배깅 방법과 유사하게, 우리는 각 트리를 훈련시키기 위해 훈련 세트에서 무작위로 서브 샘플링을 할 수 있습니다. 즉, 훈련 세트의 다른 서브 샘플이 앙상블의 다른 트리를 훈련하는 데 사용됩니다. 여기서 배깅과 랜덤 서브 샘플링의 차이점은 중복 _없이_ 샘플을 추출한다는 것입니다. Sklearn의 그레이디언트 부스팅 함수에서 서브샘플링 분율은 "subsample"이라고 하는 파라미터입니다.서브샘플링 비율이 $1.0$보다 작을 때 모델을 <font color='blue'>확률적 그레이디언트 부스팅</font>이라고 합니다.

### Sklearn 함수 및 예제

Sklearn은 그레이디언트 트리 부스팅을 위해 다음과 같은 기능을 제공합니다.

* **GradientBoostingClassifier()**
* **GradientBoostingRegressor()**

함수에 대해 매우 중요한 파리미터 중 하나는 __손실 함수__ $L(y, F)$(sklearn의 파라미터 "loss")의 정의입니다. 사실, __그레이디언트 부스팅 분류 및 회귀 알고리즘은 사용하는 손실 함수만 다릅니다.__ Sklearn [문서](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)에는 분류 및 회귀에 대한 손실 함수의 전체 목록이 섹션 "1.11.4.5.1. 손실 함수"에 나와 있습니다. <font color='green'>GradientBoostingClassifier()</font>의 기본 손실 함수는 이항 분류에 대한 음의 이항 로그 우도 손실 함수("deviance")입니다. <font color='green'>GradientBoostingRegressor()</font>의 기본 손실 함수는 손실 제곱("ls")입니다. 손실 함수가 지수 손실("exponential")로 설정된 경우, 그레디언트 부스팅이 AdaBoost 알고리즘을 복구합니다.

**learning_rate** 파리미터는 위 알고리즘의 $\eta$에 해당합니다. **n_estimator**는 앙상블의 트리 수이며 위 알고리즘의 $M$에 해당합니다. **subsample**은 확률적 그레디언트 부스팅을 위한 서브 샘플링 비율입니다. max_features 및 max_leaf_nodes와 같은 대부분의 다른 파라미터는 각 트리에 대한 것입니다. 이러한 트리 파리미터는 [튜토리얼 5](https://www.kaggle.com/code/cosmosjung/machine-learning-5-random-forests)에서 소개되었습니다.

#### 예제 7.3

이제 Red Wine Quality 데이터 세트에서 <font color='green'>GradientBoostingClassifier()</font>를 사용해 보겠습니다.


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

<font color='green'>GradientBoostingClassifier</font>의 기본 파라미터는 0.90의 AUC를 생성했습니다. 이것은 꽤 좋은 결과입니다. 이제 <font color='green'>GridSearchCV</font>([tutorial 4](https://www.kaggle.com/fengdanye/machine-learning-4-support-vector-machine)에 소개됨)를 사용하여 모델을 더욱 향상시킬 수 있는지 살펴보겠습니다.


```python
tuned_parameters = {'learning_rate':[0.05,0.1,0.5,1.0], 'subsample':[0.4,0.6,0.8,1.0]}

clf = GridSearchCV(GradientBoostingClassifier(random_state = 27), tuned_parameters, cv=5, scoring = 'roc_auc')
clf.fit(X_train, y_train)
```


```python
print('The best model is: ', clf.best_params_)
```

이전 튜토리얼에서는 여기서 멈추고 최고의 모델의 성능에 대해 살펴보았습니다. 하지만 실제로 위의 그리드 검색은 대략적인 탐색일 뿐입니다. __거친 탐색에서 발견된 최상의 모델을 중심으로 항상 세부적인 검색을 수행할 수 있습니다.__


```python
tuned_parameters = {'learning_rate':[0.09,0.1,0.11], 'subsample':[0.7,0.75,0.8,0.85,0.9]}

clf = GridSearchCV(GradientBoostingClassifier(random_state = 27), tuned_parameters, cv=5, scoring = 'roc_auc')
clf.fit(X_train, y_train)
```


```python
print('The best model is: ', clf.best_params_)
```

원하는 경우, 이후에 더 세부적인 그리드 탐색을 수행할 수 있습니다. 하지만 이 예제에서는 여기까지 하겠습니다. ROC 곡선 및 AUC 값 측면에서 모델의 성능을 확인해 보겠습니다.


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

[튜토리얼 5](https://www.kaggle.com/code/cosmosjung/machine-learning-5-random-forests)에서 GridSearchCV가 있는 500개의 트리 랜덤 포레스트를 사용하여 동일한 데이터 세트에서 0.909의 AUC를 얻었습니다. random_state에 의한 변동을 고려하면 그레이디언트 트리 부스팅 모델의 AUC(0.904)는 500 트리 랜덤 포레스트의 AUC(0.909)만큼 우수한 것으로 간주할 수 있습니다. 비교를 엄격하게 하려면 분류기를 여러 번 실행하여 그레이디언트 트리 부스팅 및 랜덤 포레스트에 대한 AUC 벡터를 각각 얻고 t-test를 실행하여 두 분류기의 평균 AUC 차이의 유의성을 결정할 수 있습니다.

__take-away는 앙상블에 100개의 트리만 있는 경우 그레이디언트 부스팅 모델이 500개의 트리 랜덤 포레스트와 동일한 성능을 발휘합니다. 이것은 부스팅의 힘을 보여줍니다. 각 트리는 이전 트리와 동일한 훈련을 반복하는 대신 이전 반복의 성능을 기반으로 비용을 절감하도록 설계되었습니다.__ 그레이디언트 트리 부스팅 앙상블에 더 많은 트리가 있는 GridSearchCV를 사용해 보십시오. 그레이디언트 부스팅 모델이 랜덤 포레스트 모델보다 성능이 우수하다는 것을 알 수 있습니다.

#### 예제 7.4

이제 <font color='green'>GradientBoostingRegressor()</font>를 사용해 보겠습니다. 목표는 11가지 와인 특성을 고려하여 와인 품질을 예측하는 것입니다.


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

r2_score는 기본 <font color='green'>AdaBoostRegressor()</font>와 비교하여 약간 개선됩니다. 그레이디언트 부스팅 모델을 더 잘 만들 수 있는지 확인하기 위해 하이퍼 파라미터를 자유롭게 사용하십시오. 힌트: <font color='green'>GridSearchCV</font>를 사용합니다.

# Newton Boosting
## 이론

__Newton boosting은 함수 공간에서 뉴턴 최적화 방법에 해당합니다.__ XGBoost는 실제로 Newton boosting이 [Nielsen 2016](https://brage.bibsys.no/xmlui/bitstream/handle/11250/2433761/16128_FULLTEXT.pdf)에게로 부터 제안되기 전에 만들어졌지만, XGBoost는 이러한 클래스의 부스팅에 속합니다. Newton Boost를 통해 XGBoost를 소개하는 것은 많은 의미가 있습니다. 그래서 이번 튜토리얼에서는 Newton Boost를 먼저 소개하고 XGBoost를 소개하겠습니다.

유한한 훈련 집합 ${x_i, y_i}$가 주어지면, $i=1,2,..,n$의 목적 함수는 다음과 같습니다.

<center>
    $\hat{\Phi}(F) = \sum_{i=1}^n L(y_i, F(x_i))$
</center>

$F(x)$는 추가 모델링을 통해 추정됩니다.

<center>
 $F(x)=\sum_{m=0}^{M}f_m(x) = f_0(x) + \sum_{m=1}^{M}f_m(x)$  
</center>

각 반복 $m$에서 "뉴턴" 단계를 $\phi(x;a_m)$로 표시합니다 .이것은 $a_m$로 매개 변수화된 약한 학습자임을 기억하십시오. 나중에 $f_m(x)=\eta \phi(x;a_m)$를 볼 수 있습니다. 이 단계를 통해 다음과 같은 목표 함수를 최소화할 수 있습니다.

<center>
$\sum_{i=1}^{n} L(y_i, F^{(m-1)}(x_i) + \phi(x_i;a_m)) \approx \sum_{i=1}^n \left[ L(y_i, F^{(m-1)}(x_i)) + \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\mid _{F(x_i)=F^{(m-1)}(x_i)}\phi(x_i;a_m) + \frac{1}{2} \frac{\partial ^2 L(y_i, F(x_i))}{\partial F(x_i)^2} \mid _{F(x_i)=F^{(m-1)}(x_i)} \phi^2(x_i;a_m) \right]$
</center>

두 번째 단계는 "함수 공간에서의 수치 최적화" 단원에서 했던 것과 마찬가지로 테일러 확장입니다. 

Define

<center>
    $g_m(x_i) = \left[ \frac{\partial L(y_i, F(x_i)) }{\partial F(x_i)} \right]_{F(x_i)=F^{(m-1)}(x_i)}$
</center>

and

<center>
    $h_m(x_i) = \left[ \frac{\partial^2 L(y_i, F(x_i)) }{\partial F(x_i)^2} \right]_{F(x_i)=F^{(m-1)}(x_i)}$
</center>

그러면 $a_m$은 다음과 같이 평가할 수 있습니다.

<center>
    $a_m = \underset{a}{\operatorname{argmin}} \sum_{i=1}^n \left[ g_m(x_i) \phi(x_i;a) + \frac{1}{2}h_m(x_i) \phi^2(x_i;a) \right]$
</center>

다음과 같은 상수를 추가하여 사각형을 완성할 수 있습니다.

<center>
    $a_m = \underset{a}{\operatorname{argmin}} \sum_{i=1}^n  \frac{1}{2}h_m(x_i) \left[ \left( -\frac{g_m(x_i)}{h_m(x_i)} \right) - \phi(x_i; a) \right]^2$
</center>

따라서 약한 학습자를 해결하는 것은 가중 최소 제곱 회귀 문제를 해결하는 문제입니다.

그런 다음 $f_m(x)$는 $f_m(x)=\eta \phi(x;a_m)$로 정의되며, 여기서 $\eta$는 학습률(임시)입니다.

### 알고리즘
* 초기화 : $F^{(0)}(x) = f_0(x) = \underset{\theta}{\operatorname{argmin}}\sum_{i=1}^{n}L(y_i, \theta)$
* $m=1,2,...,M$에 대해 다음을 수행합니다:
    * $g_m(x_i) = \left[ \frac{\partial L(y_i, F(x_i)) }{\partial F(x_i)} \right]_{F(x_i)=F^{(m-1)}(x_i)}$, $i=1,2,...,n$
    * $h_m(x_i) = \left[ \frac{\partial^2 L(y_i, F(x_i)) }{\partial F(x_i)^2} \right]_{F(x_i)=F^{(m-1)}(x_i)}$, $i=1,2,...,n$
    * $a_m =   \underset{a}{\operatorname{argmin}} \sum_{i=1}^n  \frac{1}{2}h_m(x_i) \left[ \left( -\frac{g_m(x_i)}{h_m(x_i)} \right) - \phi(x_i; a) \right]^2$
    * $f_m(x) =  \eta \phi(x;a_m)$
    * $F^{(m)}(x) = F^{(m-1)}(x) + f_m(x)$
* 예측:
    * $x$가 주어지면 $F(x) = F^{(M)}(x) = \sum_{m=0}^{M}f_m(x)$를 예측합니다.

## XGBoost
### 소개

__XGBoost는 뉴턴 트리 부스팅의 하위 클래스로 간주할 수 있습니다.__ 이전 단원에서 각 반복 $m$에서 다음 비용 함수를 최소화하고자 한다는 것을 보여주었습니다.

<center>
    $J_m = \sum_{i=1}^n \left[ g_m(x_i)\phi(x_i; a_m) + \frac{1}{2} h_m(x_i) \phi ^2(x_i;a_m) \right]$
</center>

여기서 $g_m(x_i)$ 및 $h_m(x_i)$는 이전 단원에서 정의되었습니다. __뉴턴 트리 부스팅의 경우 $\phi(x_i;a_m) = \sum_{j=1}^{T_m}w_{jm}{\mathrm I}(x_i\in R_{jm})$를 $J_m$로 변환합니다:__

<center>
    $J_m = \sum_{i=1}^n \left[ g_m(x_i)\sum_{j=1}^{T_m}w_{jm}{\mathrm I}(x_i\in R_{jm}) + \frac{1}{2} h_m(x_i) \left( \sum_{j=1}^{T_m}w_{jm}{\mathrm I}(x_i\in R_{jm}) \right)^2 \right]= \sum_{i=1}^n \left[ g_m(x_i)\sum_{j=1}^{T_m}w_{jm}{\mathrm I}(x_i\in R_{jm}) + \frac{1}{2} h_m(x_i) \sum_{j=1}^{T_m}w_{jm}^2{\mathrm I}(x_i\in R_{jm})  \right]$
</center>

${R_{jm}}$의 분리된 특성으로 인해 $J_m$를 다음과 같이 표현할 수 있습니다.

<center>
    $J_m = \sum_{j=1}^{T_m} \sum_{i; \ x_i \in R_{jm}} \left[ g_m(x_i)w_{jm} + \frac{1}{2}h_m(x_i)w_{jm}^2 \right]$
</center>

이제, 정의하자면,

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

그레디언트 트리 부스팅 단원에서 수행한 작업과 마찬가지로 트리는 다음과 같은 단계로 훈련됩니다.

* 주어진 구조에 대한 가중치 학습
* 구조 학습
* 최종 가중치 학습

결정 트리의 고정된 구조에서 가중치는 다음과 같습니다. (가중치는 주어진 트리 구조에 대해 $J_m$을 최소화해야 함)

$j=1,2,…,T_m$의 경우,
<center>
    $w_{jm}^* = -\frac{G_{jm}}{H_{jm}}$
</center>

비용 함수에 가중치를 다시 연결하면 다음과 같은 이점을 얻을 수 있습니다.

<center>
    $J_m^* = -\frac{1}{2} \sum_{j=1}^{T_m} \frac{G_{jm}^2}{H_{jm}} = \sum_{j=1}^{T_m} L_j^*$
</center>

따라서 트리 훈련 중에 발생할 수 있는 분할을 고려할 때 분할 가능성은 다음과 같습니다.

<center>
${\mathrm {Gain}}=J_{\mathrm {before}} - J_{\mathrm {after}} = L_k^* - (L_L^*+L_R^*) = \frac{1}{2} \left[ \frac{G_L^2}{H_L} + \frac{G_R^2}{H_R} - \frac{G_{km}^2}{H_{km}} \right]$ 
</center>

여기서 $k$는 분할되는 노드입니다.

뉴턴 부스팅의 경우 트리 구조가 결정된 후에는 라인 탐색이 없습니다. 따라서 $w_{jm}^*$가 최종 가중치입니다.

### 정규화

그레이디언트 트리 부스팅 모델과 마찬가지로 __개별 트리와 랜덤 서브 샘플링을 제어하여__ 뉴턴 트리 부스팅 모델을 정규화할 수 있습니다. 그러나 다른 정규화 기법도 존재합니다. 특히 XGBoost는 다음과 같은 정규화 기술을 구현합니다.

* **Random subspace**: 각 반복에서 트리를 훈련하기 위해 (중복 없이 샘플링된) 특성의 랜덤 서브 샘플이 사용됩니다. 이 프로세스를 제어하는 파리미터를 "column subsampling fraction", $w_c$라고 부르기도 합니다.
* **Extra penalization term** : 반복 $m$에서 비용 함수 $J_m$

<center>
$\Omega_m = \gamma T_m + \frac{1}{2}\lambda \| w_m \|^2_2 + \alpha \| w_m \|_1$. 
</center>

여기서, $\| w_m \|^2_2 = \sum_{j=1}^{T_m}w_{jm}^2$이고, $\| w_m \|_1 = \sum_{j=1}^{T_m}|w_{jm}|$입니다. 첫번째 인자 $\gamma T_m$는 **잎 노드 갯수에 대한 규제** 이고, 두번째 인자 $\frac{1}{2}\lambda \| w_m \|^2_2$는 **잎 노드 가중치에 대한 l2 규제**, 세번째 인자 $\alpha \| w_m \|_1$ 는 잎 노드 가중치에 대한 **l1 규제** 입니다.

In practice, you can tune the hyperparameters $w_c$, $\gamma$, $\lambda$, and $\alpha$ to regularize an XGBoost model. Now let's take a look at how each of the penailzation term affects $J_m$ and consequently the weights $w_{jm}^*$ and tree structure $\{ R_{jm} \}$.

실제로 하이퍼 파라미터 $w_c$, $\gamma$, $\lambda$ 및 $\alpha$를 조정하여 XGBoost 모델을 정규화할 수 있습니다. 이제 각 규제 인자가 결과적으로 $J_m$, 가중치 $w_{jm}^*$와 트리 구조 ${R_{jm}}$에 어떤 영향을 미치는지에 대해 살펴보겠습니다.

#### 잎 노드 개수 규제

반복 $m$의 비용 함수는 다음과 같습니다.

<center>
    $J_m = \sum_{j=1}^{T_m} \sum_{i; \ x_i \in R_{jm}} \left[ g_m(x_i)w_{jm} + \frac{1}{2}h_m(x_i)w_{jm}^2 \right] + \gamma T_m = \sum_{j=1}^{T_m} \left[ G_{jm}w_{jm} + \frac{1}{2}H_{jm}w_{jm}^2 + \gamma \right]$
</center>

가중치는 그대로 입니다.

$j=1,2,…,T_m$의 경우,
<center>
    $w_{jm}^* = -\frac{G_{jm}}{H_{jm}}$
</center> 

비용 함수에 가중치를 다시 연결하면 다음과 같은 이점을 얻을 수 있습니다.

<center>
    $J_m^* = \sum_{j=1}^{T_m} \left[ -\frac{1}{2}\frac{G_{jm}^2}{H_{jm}} + \gamma \right]$
</center>

잠재적 분할의 이득은 다음과 같습니다.

<center>
${\mathrm {Gain}}=J_{\mathrm {before}} - J_{\mathrm {after}} = L_k^* - (L_L^*+L_R^*) = \frac{1}{2} \left[ \frac{G_L^2}{H_L} + \frac{G_R^2}{H_R} - \frac{G_{km}^2}{H_{km}} \right] - \gamma$ 
</center>

여기서 $k$는 분할되는 노드입니다.

**$\gamma$가 0이 아닌 경우, 노드를 분할할 때 양의 이득을 얻기가 더 어려워집니다. 튜토리얼 5에서 설명한 것처럼 트리를 보다 엄격하게 가지치기할 수 있습니다.**

#### 잎 가중치의 l2 규제

반복 $m$의 비용 함수는 다음과 같습니다.

<center>
    $J_m = \sum_{j=1}^{T_m} \sum_{i; \ x_i \in R_{jm}} \left[ g_m(x_i)w_{jm} + \frac{1}{2}h_m(x_i)w_{jm}^2 \right] + \frac{\lambda}{2}  \sum_{j=1}^{T_m} w_{jm}^2= \sum_{j=1}^{T_m} \left[ G_{jm}w_{jm} + \frac{1}{2}\left( H_{jm} + \lambda \right) w_{jm}^2 \right]$
</center>

따라서 가중치는 다음과 같습니다.

$j=1,2,…,T_m$의 경우,
<center>
    $w_{jm}^* = -\frac{G_{jm}}{H_{jm} + \lambda }$
</center>

비용 함수에 가중치를 다시 연결하면 다음과 같은 이점을 얻을 수 있습니다.

<center>
    $J_m^* = \sum_{j=1}^{T_m} \left[ -\frac{1}{2}\frac{G_{jm}^2}{H_{jm}+\lambda} \right]$
</center>

잠재적 분할의 이득은 다음과 같습니다.

<center>
${\mathrm {Gain}}=J_{\mathrm {before}} - J_{\mathrm {after}} = L_k^* - (L_L^*+L_R^*) = \frac{1}{2} \left[ \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{G_{km}^2}{H_{km}+\lambda} \right]$ 
</center>

여기서 $k$는 분할되는 노드입니다.

#### 잎 가중치의 l1 규제

반복 $m$의 비용 함수는 다음과 같습니다.

<center>
    $J_m = \sum_{j=1}^{T_m} \sum_{i; \ x_i \in R_{jm}} \left[ g_m(x_i)w_{jm} + \frac{1}{2}h_m(x_i)w_{jm}^2 \right] + \alpha \sum_{j=1}^{T_m} |w_{jm}| = \sum_{j=1}^{T_m} \left[ \left( G_{jm} + \alpha \cdot \mathrm{sign}(w_{jm}) \right)w_{jm}+ \frac{1}{2}H_{jm}w_{jm}^2 \right] \equiv  \sum_{j=1}^{T_m} \left[ T_{\alpha}\left( G_{jm}\right)w_{jm}+ \frac{1}{2}H_{jm}w_{jm}^2 \right]$
</center>

where

<center>
$T_{\alpha}(G) \equiv G_{jm} + \alpha \cdot \mathrm{sign}(w_{jm})  = \mathrm{sign}(G)\mathrm{max}(0, |G| - \alpha)$
</center>

따라서 가중치는 다음과 같습니다.

$j=1,2,…,T_m$의 경우,
<center>
    $w_{jm}^* = -\frac{T_{\alpha}(G_{jm})}{H_{jm}}$
</center> 

비용 함수에 가중치를 다시 연결하면 다음과 같은 이점을 얻을 수 있습니다.

<center>
    $J_m^* = \sum_{j=1}^{T_m} \left[ -\frac{1}{2}\frac{T_{\alpha(}G_{jm})^2}{H_{jm}} \right]$
</center>

잠재적 분할의 이득은 다음과 같습니다.

<center>
${\mathrm {Gain}}=J_{\mathrm {before}} - J_{\mathrm {after}} = L_k^* - (L_L^*+L_R^*) = \frac{1}{2} \left[ \frac{T_{\alpha}(G_L)^2}{H_L} + \frac{T_{\alpha}(G_R)^2}{H_R} - \frac{T_{\alpha}(G_{km})^2}{H_{km}} \right]$ 
</center>

여기서 $k$는 분할되는 노드입니다.

**동시에 두 가지 이상의 정규화 유형이 있는 경우 앞서 설명한 트리 훈련 단계를 거치면 $w_{jm}^*$와 유사하게 이득을 얻을 수 있습니다.**

### 알고리즘
* 초기화 : $F^{(0)}(x) = f_0(x) = \underset{\theta}{\operatorname{argmin}}\sum_{i=1}^{n}L(y_i, \theta)$
* $m=1,2,...,M$에 대해 다음을 수행합니다.
    * $g_m(x_i) = \left[ \frac{\partial L(y_i, F(x_i)) }{\partial F(x_i)} \right]_{F(x_i)=F^{(m-1)}(x_i)}$, $i=1,2,...,n$
    * $h_m(x_i) = \left[ \frac{\partial^2 L(y_i, F(x_i)) }{\partial F(x_i)^2} \right]_{F(x_i)=F^{(m-1)}(x_i)}$, $i=1,2,...,n$
    * 이득을 최대화하는 분할을 선택하여 트리$\{R_{jm}\}_{j=1}^{T_m}$의 구조를 결정합니다. 이득은 어떤 유형의 정규화가 존재하는지에 따라 달라집니다. 위 단원을 참조하십시오.
    * 학습된 구조의 리프 가중치 $\{ w_{jm}^* \}_{j=1}^{T_m}$를 결정합니다. 특정 식은 어떤 정규화 유형이 있는지에 따라 달라집니다. 위 단원을 참조하십시오.
    * $f_m(x) =  \eta \sum_{j=1}^{T_m} w_{jm}^*{\mathrm I}(x\in R_{jm})$
    * $F^{(m)}(x) = F^{(m-1)}(x) + f_m(x)$
* 예측 :
    * $x$가 주어지면 $F(x) = F^{(M)}(x) = \sum_{m=0}^{M}f_m(x)$를 예측합니다.

### XGBoost 함수 및 예제

XGBoost 패키지는 sklearn 함수에 익숙한 사용자를 위해 Scikit-learn API를 제공합니다. 이에 대한 자세한 내용은 [공식 문서](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn)를 참조하십시오.

사용할 수 있는 두 가지 함수는 다음과 같습니다.

* XGBoostClassifier()
* XGBoostRegressor()

두 함수 모두 기본 학습률 $\eta=0.1$과 트리 수 $M=100$을 가지고 있습니다. 기본 정규화 파라미터는 $\gamma=0$, $\lambda=1$(_reg_lambda_) 및 $\alpha=0$(_reg_alpha_)입니다. 즉, 기본적으로 l2 정규화만 존재합니다. 훈련 인스턴스를 랜덤 서브 샘플링하려면 *subsample*을 1보다 작게 설정합니다. 입력 특성을 랜덤 서브 샘플링("random subsapce")하려면 _colsample_bytree_ 또는 _colsample_bylevel_ 을 1보다 작게 설정합니다. *colsample_bytree*는 각 트리의 훈련에 대한 특성을 서브 샘플링합니다. 여기서 *colsample_bylevel*은 각 분할에 대한 특성을 서브 샘플링합니다. $L(y, F)$를 정의하려면 *objective* 파리미터를 설정합니다.

#### 예제 7.5

이 예에서는 XGBoost 분류기를 사용하여 레드 와인을 분류합니다. 먼저 XGBoost 모듈을 가져와 데이터를 읽어 보겠습니다.


```python
from xgboost import XGBClassifier
```


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

그런 다음 기본 XGBoost 분류기를 구성하고 교육 세트에서 이를 교육합니다.


```python
clf = XGBClassifier(random_state = 2)
clf.fit(X_train, y_train)
```

테스트 세트에서 분류기의 성능을 평가합니다.


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

AUC는 우리가 그레이디언트 트리 부스팅에서 얻은 것보다 높습니다. 또한 하이퍼 파라미터를 사용하여 최상의 XGBoost 모델을 찾을 수 있습니다. 다음은 정규화 파리미터 공간에서 탐색하는 간단한 예입니다.


```python
tuned_parameters = {'gamma':[0,1,5],'reg_alpha':[0,1,5], 'reg_lambda':[0,1,5]}

clf = GridSearchCV(XGBClassifier(random_state = 2), tuned_parameters, cv=5, scoring='roc_auc')
clf.fit(X_train, y_train)
```


```python
print('The best model is: ', clf.best_params_)
```

이 경우 최상의 모델은 잎 노드의 수에 규제를 주지만 잎 가중치에는 규제를 주지 않습니다. 그런 다음 테스트 세트에 최상의 모델 성능을 출력할 수 있습니다.


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

#### 예제 7.6

이 예제에서는 와인 품질 데이터 세트에 대해 XGBoost 회귀 모델을 실행합니다. 다시, 먼저 모듈을 가져와서 데이터를 읽어 보겠습니다.


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

다시, 회귀 분석기의 r2 점수를 향상시키기 위해 하이퍼 파라미터 공간을 탐색할 수 있습니다. 제 시도는 다음과 같습니다.


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

전반적으로 랜덤 포레스트와 엑스트라 트리가 이 회귀 문제를 더 잘 해결하는 것으로 보입니다.([튜토리얼 6](https://www.kaggle.com/code/cosmosjung/machine-learning-6-basic-ensemble-learning) 참조).

해당 튜토리얼이 완료되었습니다.

해당 튜토리얼은 아래의 링크에 해당하는 내용을 바탕으로 정리하였습니다.

- [Machine Learning 7 Boosting](https://www.kaggle.com/code/fengdanye/machine-learning-7-boosting)

# 참고 문헌
1. Y. Freund and R.E. Schapire. "[Experiments with a New Boosting Algorithm](https://cseweb.ucsd.edu/~yfreund/papers/boostingexperiments.pdf)", *Machine Learning: Proceedings of the Thirteenth International Conference* (1996).
2. J. Zhu, S. Rosset, H. Zou and T. Hastie. "[Multi-class AdaBoost](https://web.stanford.edu/~hastie/Papers/samme.pdf)", 2006.
3. H. Drucker. "[Improving Regressors using Boosting Techniques](https://pdfs.semanticscholar.org/8d49/e2dedb817f2c3330e74b63c5fc86d2399ce3.pdf)", 1997.
4. J. H. Friedman. "[Greedy Function Approximation: A Gradient Boosting Machine](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)", 1999.
5. T. Chen and C. Guestrin. "[XGBoost: A Scalable Tree Boosting System](https://arxiv.org/pdf/1603.02754.pdf)", 2016.
6. D. Nielsen. "[Tree Boosting with XGBoost](https://brage.bibsys.no/xmlui/bitstream/handle/11250/2433761/16128_FULLTEXT.pdf)", 2016.
7. J. H. Friedman, T. Hastie, and R. Tibshirani. ["Additive Logistive Regression: A Statistical View of Boosting](https://web.stanford.edu/~hastie/Papers/AdditiveLogisticRegression/alr.pdf)", *The Annals of Statistics* 28 (2000), p337-407.
