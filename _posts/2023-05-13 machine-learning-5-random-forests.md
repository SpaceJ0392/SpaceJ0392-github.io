## Machine Learing 5. Random Forests

__목차__

- 결정 트리 모델
    - 결정 트리 분류 모델
        - 소개
        - 지니 불순도(Gini impurity)
        - 훈련 알고리즘
        - 과대적합 및 정규화
        - 레드 와인 품질 데이터 세트에 대한 결정 트리
    - 결정 트리 회귀 모델
        - 소개
        - 예제
- 랜덤 포레스트 모델
    - 랜덤 포레스트란?
    - 예제 - 레드 와인 품질 데이터 세트의 랜덤 포레스트 분류
    - 예제 - 와인 인식 데이터 세트의 랜덤 포레스트 회귀
- 부록: CART 알고리즘
    - 고정된 구조의 가중치 학습
    - 구조 학습
    - 가지치기(Pruning)


```python
import numpy as np
import graphviz
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import os
print(os.listdir("../input")) # 입력 데이터 확인
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Cell In[1], line 2
          1 import numpy as np
    ----> 2 import graphviz
          3 import pandas as pd
          4 import matplotlib.pyplot as plt
    

    ModuleNotFoundError: No module named 'graphviz'



```python
# plot 에 대한 초기 설정
plt.rc('axes', lw = 1.5)
plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('xtick.major', size = 5, width = 3)
plt.rc('ytick.major', size = 5, width = 3)
```

# 결정 트리
## 결정 트리 분류 모델
### 소개

$n$개의 특성(특성 1,2,…,n)이 있는 입력 데이터 세트 $X$가 있고 데이터 세트의 각 인스턴스가 분류된다고 가정할 때,(예: 클래스 A, B 및 C로 분류) 데이터 세트에 대해 훈련된 깊이 = 2인 결정 트리 분류기는 _대략_ 다음과 같습니다:

<img src="https://imgur.com/aJLCV2J.png" width="500px"/>

예측값을 구하기 위해, 결정 트리는 먼저 인스턴스의 특성 값 $k_1$을 임계값 $t_{k1}$과 비교합니다. $k_1 \leq_{k1}$인 경우 인스턴스는 "A 클래스"로 분류하며, $k_1 > t_{k1}$인 경우 결정 트리는 이 인스턴스의 특성 값 $k_2$가 임계값 $t_{k2}$보다 작거나 같은지 확인합니다. 그 결과가 "Yes"인 경우 인스턴스는 "클래스 B"로 분류되고, 그렇지 않은 나머지는 "클래스 C"로 분류됩니다. 맨 위에 있는 노드를 _루트 노드_(깊이 = 0)라고 하며, 하위 노드가 없는 노드를 *잎 노드*라고 합니다.  
결정 트리 분류기를 sklearn 모듈에서 제공하는 와인 인식 데이터 세트에 적용해 보겠습니다.


```python
from sklearn.datasets import load_wine
wine = load_wine()
X = wine.data # 훈련 셋
y = wine.target #레이블 셋
```

저희는 결정 트리 모델을 훈련시키기 위해 sklearn의 **DecisionTreeClassifier** 모델을 사용할 것입니다. 


```python
tree = DecisionTreeClassifier(max_depth = 2, random_state = 0)
tree.fit(X,y)
```

트리를 시각화하려면 python의 __graphviz__ 모듈이 필요합니다. 이 예제에서는 이미 모듈을 아래의 명령어로 불러왔습니다.

> import graphviz

다음 코드를 실행하여 트리 그래프를 표시합니다.


```python
dot_data = export_graphviz(tree,
                out_file = None,
                feature_names = wine.feature_names,
                class_names=wine.target_names,
                rounded = True,
                filled = True)

graph = graphviz.Source(dot_data)
graph.render() 
graph
```

위에 보여지는 그림을 보면, 루트 노드(깊이=0)에서는 분할을 위한 특성으로 "proline"을 선택하고 임계값으로 755.0을 선택했습니다. 2 개의 깊이가 1인 노드는 각각 "od280/od315_of_dilated_wine"과 "flavanoids"를 분할을 위한 특성으로 선택했습니다. 4 개의 잎 노드(깊이=2)에서는 각각 클래스를 예측합니다. 그 중 2 개는 class_2로 예측하고 1 개는 class_1으로 나머지 1 개는 class_0으로 예측합니다. 그래프를 통해 "gini" 및 "value"과 같은 노드의 다른 속성도 확인할 수 있습니다. 이러한 특성은 다음과 같은 의미를 가집니다.

- 각 노드에서 __"samples"__ 는 해당 노드에 속하는 훈련 인스턴스(표본)의 수를 나타냅니다. 예를 들어 proline > 755.0인 표본은 67개 있습니다.
- __"value"__ 속성은 각 노드에 대해 각 클래스에 속하는 인스턴스(표본) 수를 의미합니다. 예를 들어 가장 왼쪽의 잎 노드에서는 class_0에 0개, class_1에 6개, class_2에 40개의 인스턴스가 있습니다.
- 각 노드에서는 해당 노드에서 가장 주요한 인스턴스(표본)을 가진 클래스를 **"class"** 에 할당합니다. 따라서 루트 노드가 3 개의 클래스 각각에 유사한 수의 인스턴스를 가지고 있더라도 루트 노드의 클래스 레이블은 71 > 59 > 48이므로 "class_1"입니다.
- __예측값을 구하기 위해서, 잎 노드에 도달할 때까지 트리를 계속 따라 갑니다. 각 클래스의 확률은 잎 노드에서 각 클래스의 훈련 샘플의 비율입니다.__   
예를 들어, 왼쪽 하단 잎 노드는 $\frac{40}{46}\approx$87.0%의 확률로 class_2로 결과를 예측합니다.
이 속성은 DecisionTreeClassifier의 *predict_proba()* 함수를 통해 구할 수 있습니다. __동일한 잎 노드에 속하는 모든 표본은 동일한 예측 확률을 공유합니다.(즉, 해당 노드에서 속하는 모든 표본의 예측 확률의 합은 100%여야 합니다.)__

### 지니 불순도(Gini impurity)

위 그림을 통해 `gini`로 표기된 __지니 불순도__(*Gini impurity*)라는 또 다른 노드 속성을 확인할 수 있습니다. 지니 불순도는 노드에서의 샘플들의 구성이 얼마나 "순수"한지를 의미합니다. 예를 들어, 노드에 class_0 및 class_1의 샘플의 수가 0이지만 class_2의 샘플의 수가 20이면 지니 불순도 0이 됩니다. 지니 불순도의 수학적 정의는 아래과 같습니다.

<center>
$G=1-\sum_{k=1}^{n}p_k^2$
</center>

여기서 $p_k$는 노드에서 클래스 k의 인스턴스의 비율입니다. 위 트리 그림의 왼쪽 아래 노드에 대해, 우리는 다음과 같이 지니 불순도를 정의할 수 있습니다.

<center>
$G = 1 - (\frac{0}{46})^2 - (\frac{6}{46})^2 - (\frac{40}{46})^2 \approx 0.227.$    
</center>

**지니 불순도는 결정 트리에서 노드를 분할하는 기준 역할**을 하며, 이에 대해 아래에서 더 자세히 설명할 것입니다.


### 훈련 알고리즘

CART(Classification and Regression Tree) 알고리즘은 **하나의 노드를 한 번 시행에 두 개의 하위 노드로 분할**하는 방식으로 작동합니다. 분할 기준은 불순도 비용 함수를 최소화하는 특징 $k$와 이 특징의 임계값$t_k$으로 구성됩니다.

<center>
    $J(k, t_k) = G_{\mathrm {left}}\times{m_{\mathrm {left}}}/{m}+ G_{\mathrm {right}}\times{m_{\mathrm {right}}}/{m}$
</center>

이것은 **왼쪽 과 오른쪽 자식 노드의 지니 불순도의 가중 합계입니다.** 여기서 $G_{\mathrm {left}}$($G_{\mathrm {right}}$)는 왼쪽(오른쪽) 자식 노드의 지니 불순도이고, $m_{\mathrm {right}}$($m_{\mathrm {right}}$)는 왼쪽(오른쪽) 자식 노드의 샘플 수이고, $m$는 부모 노드 $m=m_{\mathrm {left}} + m_{\mathrm {right}}$의 샘플 수입니다.
알고리즘은 우선 루트 노드를 두 개로 분할하고, 각각의 자식 노드를 다시 두 개로 분할하는 과정을 반복하는 구조입니다. 이 과정은 트리가 사용자가 정의한 최대 깊이에 도달하거나, 분할 후 불순도 $G_{\mathrm {left}}\times{m_{\mathrm {left}}}/{m}+ G_{\mathrm {right}}\times{m_{\mathrm {right}}}/{m}$가 부모 노드의 불순도 $G_{\mathrmathrm{parent}}$보다 크지만 다른 종료 조건이 존재하는 경우 중지됩니다.

이 알고리즘과 sklearn의 결정 트리 분류기에 대해 몇 가지 지적하고 싶은 것이 있습니다:

- CART 알고리즘은 **국소적으로 최적의 결정**을 탐색하는 탐욕 알고리즘입니다. 따라서 이 최적 분할은 트리 하단에서 전체적으로 가장 작은 불순도로 이어지는지 _여부를 고려하지 않고_ 각 노드에서 탐색됩니다. 이는 DecisionTreeClassifier's [documentation](https://scikit-learn.org/stable/modules/tree.html#tree) 문서에도 언급되어 있습니다.
- 알고리즘을 고려할 때, 어떤 DecisionTreeClassifier는 기본 설정을 사용하는 경우 특정 교육 세트(즉, 임의성 없음)가 주어지면 결정되어 있는 결과를 예측합니다. 그러나, __DecisionTreeClassifier는 기본 설정에서도 랜덤성을 표시합니다.(아마도 여기서의 DecisionTreeClassifier는 skilearn의 분류기를 의미하는 것으로 보임.)__ 이는 분류기가 모든 특성을 랜덤하게 다시 정렬한 다음 각 특성을 테스트하기 때문입니다. 만약, 두 개의 분할이 동률일 경우 먼저 발생한 분할이 선택됩니다. 자세한 내용은 이 [github page](https://github.com/scikit-learn/scikit-learn/issues/8443)에서 확인할 수 있습니다.
- 훈련을 가속하기 위해 DecisionTreeClassifier의 max_features of Decision 하이퍼 파리미터를 설정할 수 있습니다. 이 하이퍼 파리미터는 총 특성 수보다 작아야 합니다. 이 경우 알고리즘은 랜덤하게 샘플링된 특성의 집합에서 최적 분할을 실행합니다.


### 과대적합 및 정규화

규제이 없으면 __결정 트리 분류기는 쉽게 과대적합됩니다.__ 규제가 없다면, 분류기는 분류를 100% 정확하게 하기 위해 노력할 것이며, 이는 매우 적은 샘플 수를 갖는 잎 노드로 이루어진 결과를 초래합니다. 이 단원에서는 결정 트리의 과대적합 상태를 살피고 어떻게 수정 할지를 알아보겠습니다. 시각화를 돕기 위해 flavanoids와 proline levels이라는 두 가지 기능을 분류기에 훈련시킬 것입니다.


```python
wine = load_wine()
X = wine.data[:,[6,12]] # flavanoids and proline
y = wine.target

tree1 = DecisionTreeClassifier(random_state=5) 
tree1.fit(X,y)
```

다음과 같이 결정 경계를 그릴 수 있습니다.


```python
# 결정 경계를 그리기 위한 초기 작업
x0min, x0max = X[:,0].min()-1, X[:,0].max()+1
x1min, x1max = X[:,1].min()-10, X[:,1].max()+10
xx0, xx1 = np.meshgrid(np.arange(x0min,x0max,0.02),np.arange(x1min, x1max,0.2))
Z = tree1.predict(np.c_[xx0.ravel(), xx1.ravel()])
Z = Z.reshape(xx0.shape)
```


```python
plt.subplots(figsize=(12,10))
plt.contourf(xx0, xx1, Z, cmap=plt.cm.RdYlBu)
plot_colors = "ryb"
n_classes = 3
for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, label=wine.target_names[i],
                cmap=plt.cm.RdYlBu, edgecolor='black', s=30)
plt.legend(fontsize=18)
plt.xlabel('flavanoids', fontsize = 18)
plt.ylabel('proline', fontsize = 18)
plt.show()
```

보다시피, 이 결정 트리 분류기는 과대적합입니다. 훈련 세트에는 완벽하게 들어맞지만, 일반화는 잘 되지 않을 것입니다. 만약, 트리를 그린다면, 트리가 매우 깊다는 것을 알 수 있을 것입니다(깊이 = 8). 따라서 규제를 통해 과대적합 문제를 조정할 필요가 있습니다. DecisionTreeClassifier에는 분류기를 정규화하도록 조정할 수 있는 몇 가지 매개변수가 있습니다.

* max_depth: 트리 깊이를 줄입니다.
* min_samples_split: 분할된 하나의 노드가 가져야 하는 최소 샘플 수를 늘립니다.
* min_samples_leaf: 잎 노드가 가져야 하는 최소 샘플 수 늘립니다.
* max_leaf_nodes: 최대  노드 수를 줄입니다.
* max_features: 각 분할 시 탐색할 최대 특성 수를 줄입니다. (기본값은 모든 특성을 탐색하는 것입니다).
* ...

여기서는 각각 **max_depth** 와 **max_leaf_nodes**를 정규화한 결과를 보여드리겠습니다.


```python
# limit maximum tree depth
tree1 = DecisionTreeClassifier(max_depth=3,random_state=5) 
tree1.fit(X,y)

# limit maximum number of leaf nodes
tree2 = DecisionTreeClassifier(max_leaf_nodes=4,random_state=5) 
tree2.fit(X,y)

x0min, x0max = X[:,0].min()-1, X[:,0].max()+1
x1min, x1max = X[:,1].min()-10, X[:,1].max()+10
xx0, xx1 = np.meshgrid(np.arange(x0min,x0max,0.02),np.arange(x1min, x1max,0.2))

Z1 = tree1.predict(np.c_[xx0.ravel(), xx1.ravel()])
Z1 = Z1.reshape(xx0.shape)
Z2 = tree2.predict(np.c_[xx0.ravel(), xx1.ravel()])
Z2 = Z2.reshape(xx0.shape)

fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(15,6))
ax[0].contourf(xx0, xx1, Z1, cmap=plt.cm.RdYlBu)
ax[1].contourf(xx0, xx1, Z2, cmap=plt.cm.RdYlBu)
plot_colors = "ryb"
n_classes = 3
for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(y == i)
    ax[0].scatter(X[idx, 0], X[idx, 1], c=color, label=wine.target_names[i],
                cmap=plt.cm.RdYlBu, edgecolor='black', s=30)
    ax[1].scatter(X[idx, 0], X[idx, 1], c=color, label=wine.target_names[i],
                cmap=plt.cm.RdYlBu, edgecolor='black', s=30)
ax[0].legend(fontsize=14)
ax[0].set_xlabel('flavanoids', fontsize = 18)
ax[0].set_ylabel('proline', fontsize = 18)
ax[0].set_ylim(260,1690)
ax[0].set_title('max_depth = 3', fontsize = 14)
ax[1].legend(fontsize=14)
ax[1].set_xlabel('flavanoids', fontsize = 18)
ax[1].set_ylabel('proline', fontsize = 18)
ax[1].set_ylim(260,1690)
ax[1].set_title('max_leaf_nodes = 4', fontsize = 14)
plt.show()
```

분류기 _tree1_(최대 트리 깊이 제한) 과 _tree2_(최대 리프 노드 수 제한)는 모두 아무런 규정이 없는 분류기에 비해 보다 정규화된 동작을 갖습니다. 정규화된 분류기는 기본 분류기보다 __낮은 분산(과대적합이 적음)과 보다 높은 편향(예측 오차가 더 높음)을 갖습니다.__ 따라서 __항상 결정 트리 분류기를 사용하고자 할 때에는 정규화하여 테스트 세트로 잘 일반화되었는지 확인해야 합니다.__ 그러나 정규화할 파라미터와 범위는 분석의 특정 목표에 따라 달라집니다.
추가적으로 알고리즘이 한 번에 하나의 특성에 초점을 맞추기 때문에 __모델 훈련을 진행하기 전에 데이터를 표준화할 필요가 없습니다.__

### 레드 와인 품질 데이터 세트에 대한 결정 트리
이제 이전에 로지스틱 회귀 분석 튜토리얼([logistic regression tutorial](https://www.kaggle.com/fengdanye/machine-learning-3-logistic-and-softmax-regression))에서 사용했던 레드 와인 품질 데이터 세트에 대한 결정 트리 분류기를 훈련하겠습니다. (이 데이터 세트는 위 그림에서 사용한 와인과 관련이 있는 두 데이터 세트와는 다른 데이터 세트입니다).

먼저 데이터를 읽어 보겠습니다.


```python
wineData = pd.read_csv('../input/winequality-red.csv')
wineData.head()
```

로지스틱 회귀 분석 튜토리얼에서 수행한 작업과 마찬가지로 **와인의 품질이 7 이상이면 '좋음'(1)으로, 그렇지 않으면 '좋지 않음'(0)으로 정의합니다.**


```python
wineData['category'] = wineData['quality'] >= 7
wineData.head()
```


```python
X = wineData[wineData.columns[0:11]].values
y = wineData['category'].values.astype(np.int)
```

그런 다음 전체 데이터 세트에서 30%의 샘플을 테스트 세트로, 70%의 샘플을 훈련 세트로 훈련 세트와 테스트 세트를 분할합니다. 고정된 결과를 위해 'random_state' 파리미터를 고정하였으며, 직접 코드를 실행할 때 원한다면 제거할 수 있습니다.


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print('X train size: ', X_train.shape)
print('y train size: ', y_train.shape)
print('X test size: ', X_test.shape)
print('y test size: ', y_test.shape)
```

앞서 언급했듯이 분류기를 조절할 수 있는 여러 파라미터가 있습니다. 여기서는 그 중 세 가지에 초점을 맞춥니다.

* max_features: 각 노드에서 고려할 최대 특성 수입니다. 부동 소수는 고려 중인 특성의 비율을 의미합니다. max_features가 < 1.0인 경우, 알고리즘은 각 노드에서 임의의 랜덤한 특성 집합 중에서 가장 적합한 특성을 탐색합니다.
- max_depth: 최대 트리 깊이. [설명서](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)에 따르면 기본적으로 max_depth=None으로 설정되며, "모든 잎 노드가 순수할 때까지 또는 모든 잎 노드가 min_samples_split 샘플 개수보다 작은 값을 포함할 때까지 확장됩니다."라는 의미입니다. 하지만 여기서는 그것을 특정한 숫자로 제한하고 싶습니다.
- min_samples_leaf: 각 잎 노드의 최소 샘플 수입니다. 기본적으로 min_delay_leaf=1 입니다.

To compare classifiers with different combinations of values of the above parameters, we use sklearn's GridSearchCV function, which was introduced in the last section of my previous tutorial. Here I will repeat what I typed in that tutorial:
위 파라미터의 값 조합이 다른 분류기를 비교하기 위해 [이전 튜토리얼](https://www.kaggle.com/code/fengdanye/machine-learning-4-support-vector-machine/notebook)의 마지막 단원에 소개된 sklearn의 __GridSearchCV__ 함수를 사용합니다. 이전 튜토리얼에 있던 내용을 이용하여 타이핑합니다.

> 모델을 선택하기 위해 Scikit-Learn의 GridSearchCV 함수를 사용합니다. 이 함수는 가능한 모든 하이퍼 파라미터 조합을 통해 반복되며 각 조합(즉, 모델)에 대해 교차 검증을 실행합니다. 최상의 모델은 교차 검증 중에 최상의 점수를 산출하는 모델입니다. 교차 검증 중에 GridSearchCV는 (X_train, y_train)을 더 분할하여 훈련 세트와 테스트 세트로 만듭니다. 최상의 모델이 발견되면 GridSearchCV는 전체 훈련 데이터(X_train, y_train)을 최상의 모델에 대해 훈련시키며, 추가 예측에 사용될 것은 이 훈련된 최상의 모델입니다.

여기서 관심 점수로 AUC(ROC 곡선 아래 영역)를 선택합니다. [로지스틱 회귀](https://www.kaggle.com/code/fengdanye/machine-learning-3-logistic-and-softmax-regression/notebook) 튜토리얼로 이동하여 ROC 곡선 및 AUC에 대해 알아볼 수 있습니다.


```python
from sklearn.model_selection import GridSearchCV
```


```python
tuned_parameters = {'max_features': [0.5,0.6,0.7,0.8,0.9,1.0], 'max_depth': [2,3,4,5,6,7],'min_samples_leaf':[1,10,100],'random_state':[14]} 

clf = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=5, scoring='roc_auc')
clf.fit(X_train, y_train)
```

print(clf.cv _property_)를 사용하여 각 분류기의 성능에 대한 자세한 보고서를 생성할 수 있습니다. $6\times6\times3=108$ 분류기의 테스트의 결과 보고서가 너무 길기 때문에 보고서는 여기에 표시하지 않습니다. 훈련 세트에 최상의 모델과 그 성능을 인쇄해 보겠습니다.


```python
print('The best model is: ', clf.best_params_)
print('This model produces a mean cross-validated score (auc) of', clf.best_score_)
```

이제 테스트 세트(X_test, y_test)에서의 분류기의 성능을 살펴보겠습니다.


```python
from sklearn.metrics import precision_score, accuracy_score
y_true, y_pred = y_test, clf.predict(X_test)
print('precision on the evaluation set: ', precision_score(y_true, y_pred))
print('accuracy on the evaluation set: ', accuracy_score(y_true, y_pred))
```

* 좋은 것으로 예측되는 와인 중 55.8%가 실제로 좋은 와인입니다.
* 87.3%의 와인에 대해 품질을 정확하게 예측하고 있습니다.

이제 ROC 곡선을 그리고 테스트 세트에서 AUC를 계산합니다.


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

## 결정 트리 회귀 모델
### 소개

결정 트리는 회귀를 수행할 수도 있습니다. 알고리즘은 여전히 특성 $k$와 임계값 $t_k$를 사용하여 노드를 한 번에 두 개의 자식 노드로 나누는 방식으로 작동하지만 비용 함수는 다릅니다.

<center>
    $J(k, t_k) = \mathrm {MSE_{left}}\times{m_{\mathrm {left}}}/{m}+ \mathrm {MSE_{right}}\times{m_{\mathrm {right}}}/{m}$
</center>

where

<center>
    $\mathrm {MSE_{left (right)}} =\frac{1}{m_{left(right)}} \sum_{i\in \mathrm{left (right)}}(\hat{y}_{\mathrm {left (right)}}-y^{(i)})^2$    
</center>
<center>
    $\hat{y}_{\mathrm{left(right)}} = \frac{1}{m_{\mathrm{left(right)}}} \sum_{i \in \mathrm{left(right)}} y^{(i)}$
</center>

훈련 과정은 위의 방정식을 이해하는 데 도움이 되도록 아래에서 단계별로 설명합니다.

- 알고리즘은 특성 $k$와 임계값 $t_k$를 선택하고 부모 노드의 샘플을 왼쪽 노드($k$ 값 <= $t_k$)와 오른쪽 노드($k$ 값 > $t_k$)로 분할합니다.
- 왼쪽 노드 $\hat{y}{\mathrm{left}}$에 대해 예측된 $y$ 값은 왼쪽 노드에 있는 모든 샘플의 $y$ 값의 평균입니다. 마찬가지로 오른쪽 노드 $\hat{y}{\mathrm{right}}$에 대해 예측된 $y$ 값은 오른쪽 노드에 있는 모든 샘플의 $y$ 값의 평균입니다.
- 이 알고리즘은 각 노드 내의 평균 제곱 오차를 계산합니다. $\mathrm{MSE_{left}}$ 및 $\mathrm{MSE_{right}}$.
- 이 알고리즘은 비용 함수 $J(k, t_k)$를 최소화하는 분할을 찾기 위해 가능한 모든 $k, t_k$를 검색합니다.
* 알고리즘은 분할로 자식 노드를 생성합니다.
* ...

기본적으로 결정 트리 회귀 알고리즘은 불순도을 사용하는 대신 평균 제곱 오차(MSE)를 사용합니다. 간단한 예를 살펴보겠습니다.

### 예제

sklearn에서 제공하는 와인 인식 데이터 세트를 재사용해 보겠습니다. 여기서 우리는 flavanoid 값을 입력 "$x$"로, proline 값을 출력 "$y$"로 취할 것입니다.


```python
wine = load_wine()
x = wine.data[:,6] # flavanoids
y = wine.data[:,12] # proline
```


```python
plt.scatter(x,y)
plt.xlabel('flavanoids',fontsize=16)
plt.ylabel('proline',fontsize=16)
plt.show()
```

결정 트리 회귀 모델을 실행하기 위해 sklearn의 **DecisionTreeRegressor**을 사용할 수 있습니다.


```python
x = x.reshape(-1,1)
tree = DecisionTreeRegressor(max_depth = 2, random_state = 5) # max tree depth is limited to 2
tree.fit(x,y)
```


```python
dot_data = export_graphviz(tree,
                out_file = None,
                feature_names = ['flavanoids'],
                rounded = True,
                filled = True)

graph = graphviz.Source(dot_data)
graph.render() 
graph
```

이 훈련된 회귀 분석기의 예측을 그림으로 그려 보겠습니다.


```python
xx = np.arange(0,5.3, step = 0.01).reshape(-1,1)
yy = tree.predict(xx)

plt.scatter(x,y)
plt.plot(xx,yy,color='r')
plt.xlabel('flavanoids',fontsize=16)
plt.ylabel('proline',fontsize=16)
plt.show()
```

이 회귀 분석기는 데이터를 대략적으로 적합시킬 뿐입니다. 만약 우리가 트리 깊이에 제한이 없는 DecisionTreeRegressor을 시도한다면요?



```python
x = x.reshape(-1,1)
tree = DecisionTreeRegressor(random_state = 5)
tree.fit(x,y)

xx = np.arange(0,5.3, step = 0.01).reshape(-1,1)
yy = tree.predict(xx)

plt.scatter(x,y)
plt.plot(xx,yy,color='r')
plt.xlabel('flavanoids',fontsize=16)
plt.ylabel('proline',fontsize=16)
plt.show()
```

예상대로 회귀 분석기가 과대적합됩니다. 또한 모든 값을 완벽하게 예측하려고 하기 때문에 특이치에 매우 민감합니다. 트리 최대 깊이를 3으로 제한해 보겠습니다.


```python
x = x.reshape(-1,1)
tree = DecisionTreeRegressor(max_depth = 3, random_state = 5)
tree.fit(x,y)

xx = np.arange(0,5.3, step = 0.01).reshape(-1,1)
yy = tree.predict(xx)

plt.scatter(x,y)
plt.plot(xx,yy,color='r')
plt.xlabel('flavanoids',fontsize=16)
plt.ylabel('proline',fontsize=16)
plt.show()
```

이것이 과대적합 회귀 분석기보다 확실히 낫지만, 적합하다고 생각됩니까?__결정 트리의 주요 특징 중 하나는 데이터에 대해 어떠한 가정도 하지 않는다는 것입니다.__ 이것은 때때로 문제가 될 수 있습니다. 예를 들어, 위 그림에서 예측 선이 맨 오른쪽 데이터 점에 맞도록 아래로 떨어지는 것을 볼 수 있듯이 __결정 트리는 특이치에 민감할 수 있습니다.__ 데이터에 노이즈가 많은 경우, 회귀 분석을 안내하는 기본 모델(예: 선형 모델, 다항식 모델)이 없기 때문에 __결정 트리가 노이즈와 신호를 구별하지 못할 수 있습니다.__

트리를 정규화하면 위의 문제에 도움이 될 수 있지만 경우에 따라 결정 트리 회귀 모델이 제대로 작동하지 않을 수 있습니다. 따라서 실제로는 여러 가능한 회귀 분석(결정 트리, 다항식 등)을 시도하고 가장 잘 수행되는 회귀 분석을 선택하는 것이 좋습니다. 결정 트리(랜덤 포레스트)의 앙상블도 개별 트리보다 더 잘 작동하는 경향이 있는데, 이는 다음에 설명할 것입니다.

__트리 훈련을 위한 CART 알고리즘에 대한 자세한 내용을 알아보고 싶다면, 새로 추가된 부록을 참조하십시오.__

# 랜덤 포레스트 모델
## 랜덤 포레스트란?

**랜덤 포레스트 모델은 결정 트리의 집합입니다. 랜덤 포레스트에서 각 결정 트리는 일반적으로 훈련 세트 크기와 동일한 대체 샘플로 샘플링되는 훈련 세트의 무작위로 선택된 집합으로 훈련됩니다. 분류의 경우 모든 트리에서 평균 확률이 가장 높은 클래스가 선택됩니다. 회귀의 경우 예측된 $y$ 값은 모든 트리에서 예측된 $y$ 값의 평균입니다.**

Scikit-learn에서 랜덤 포레스트 분류기([RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html))는 랜덤 포레스트 분류에 사용되고 랜덤 포레스트 회귀기([RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html))는 랜덤 포레스트 회귀에 사용됩니다. [문서](https://scikit-learn.org/stable/modules/ensemble.html#forest) 에서 언급했듯이, "(훈련에서 무작위 서브 샘플링으로 인해) 포레스트의 편향은 보통 (단일 비랜덤 트리의 편향과 관련하여) 약간 증가하지만 평균화로 인해 분산도 감소하며, 일반적으로 편향의 증가를 보상하는 것보다 더 많이 감소하므로 전반적으로 더 나은 모델을 산출합니다." __따라서 일반적으로 단일 트리보다 랜덤 포레스트가 선호됩니다.__ __문서는 또한 DecisionTreeClassifier는 선호되지 않고, 기본적으로 RandomForestClassifier는  최적 분할을 위해 모든 특성을 탐색하지 않고 임의의 집합만 탐색한다고 언급하고 있습니다.__ RandomForestClassifier의 설명서([RandomForestClassifier's documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html))에 따르면 max_features는 기본 값으로 sqrt(n_features)입니다.

이제 레드 와인 품질 데이터 세트에서 랜덤 포레스트 분류기를 사용해 보겠습니다.

## 예제 - 레드 와인 품질 데이터 세트의 랜덤 포레스트 분류

이전 단원에서 설명한 것처럼 데이터를 읽고 이진화된 와인 품질을 정의합니다.


```python
wineData = pd.read_csv('../input/winequality-red.csv')

wineData['category'] = wineData['quality'] >= 7

X = wineData[wineData.columns[0:11]].values
y = wineData['category'].values.astype(np.int)
```

그런 다음 "레드 와인 품질 데이터 세트의 의사 결정 트리" 단원에 사용된 것과 동일한 random_state를 사용하여 데이터 세트를 훈련 세트와 테스트 세트로 분할합니다.


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print('X train size: ', X_train.shape)
print('y train size: ', y_train.shape)
print('X test size: ', X_test.shape)
print('y test size: ', y_test.shape)
```

그런 다음 scikit-learn의 __RandomForestClassfier__를 사용하여 랜덤 포레스트 분류기를 교육합니다. RandomForestClassfier에는 DecisionTreeClassifier의 모든 하이퍼 파라미터가 있습니다. "레드 와인 품질 데이터 세트에 대한 결정 트리" 단원에서 수행한 작업과 마찬가지로 __GridSearchCV__를 사용하여 max_features 및 max_depth와 같은 다양한 하이퍼 파라미터 조합을 테스트합니다.

또한 R**RandomForestClassfier**에는 n_estimator 및 n_jobs와 같은 고유한 하이퍼 파리미터가 있습니다. n_estimators는 포레스트의 트리 모델의 개수이고, n_jobs는 병렬로 실행할 작업 수입니다. 여기서는 n_estimators를 500으로, n_jobs를 -1(가능한 모든 프로세서를 사용함)로 설정합니다.


```python
tuned_parameters = {'n_estimators':[500],'n_jobs':[-1], 'max_features': [0.5,0.7,0.9], 'max_depth': [3,5,7],'min_samples_leaf':[1,10],'random_state':[14]} 
# random_state is only to ensure repeatable results. You can remove it when running your own code.

clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='roc_auc')
clf.fit(X_train, y_train)
```

훈련에 시간이 좀 걸릴 수 있습니다. 작업이 완료된 후 최적의 모델을 출력하면 그 것은 교차 검증된 AUC를 의미합니다.


```python
print('The best model is: ', clf.best_params_)
print('This model produces a mean cross-validated score (auc) of', clf.best_score_)
```

이제 이 모델이 테스트 세트에서 어떤 성능을 발휘하는지 살펴보겠습니다.


```python
from sklearn.metrics import precision_score, accuracy_score
y_true, y_pred = y_test, clf.predict(X_test)
print('precision on the evaluation set: ', precision_score(y_true, y_pred))
print('accuracy on the evaluation set: ', accuracy_score(y_true, y_pred))
```


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

결과는 결정 트리보다 훨씬 좋습니다.

## 예제 - 와인 인식 데이터 세트의 랜덤 포레스트 회귀 분석

진행 함에 앞서, 해당 데이터 세트에 대해 결정 트리 회귀를 테스트한 것을 기억해 두어야 합니다.


```python
wine = load_wine()
x = wine.data[:,6] # flavanoids
y = wine.data[:,12] # proline

plt.scatter(x,y)
plt.xlabel('flavanoids',fontsize=16)
plt.ylabel('proline',fontsize=16)
plt.show()
```

이제 동일한 데이터 세트에서 랜덤 포레스트 회귀 모델을 사용해 보겠습니다. 우리는 다시 500개의 트리 모델을 사용할 것입니다. 각 트리의 최대 깊이는 2입니다.


```python
reg = RandomForestRegressor(n_estimators=500, n_jobs=-1, max_depth=2, random_state = 5)
x = x.reshape(-1,1)
reg.fit(x,y)
```


```python
xx = np.arange(0,5,0.02).reshape(-1,1)
yhat = reg.predict(xx)
plt.plot(xx,yhat,color='red')
plt.scatter(x,y)
plt.xlabel('flavanoids',fontsize=16)
plt.ylabel('proline',fontsize=16)
plt.show()
```

예측이 훨씬 매끄럽지만 여전히 특이치의 영향을 받음을 확인할 수 있습니다.

----------------------------------------------------
# 부록: CART 알고리즘

앞서 결정 트리 분류 모델(지니 불순도 사용)와 회귀 모델(MSE 사용)를 훈련하기 위해 CART 알고리즘을 도입했습니다. 여기서는 좀 더 일반적인 의미의 CART 알고리즘을 소개하겠습니다. <font color='blue'>이 알고리즘에 대한 이해는 튜토리얼 7에서 소개할 부스팅 알고리즘을 이해하는 데 중요합니다</font>. 이 부록의 내용은 Didrik Nielsen의 마스터 논문([Master thesis](https://brage.bibsys.no/xmlui/handle/11250/2433761))에 기초합니다.

__CART 알고리즘은 노드를 한 step에 두 개의 자식 노드로 분할하여 트리를 훈련시킵니다. 각 단계에서 가능한 모든 분할(분할 할 노드, 분할 할 특성, 특성 임계값) 중에서 훈련 오차를 최소화하는 것이 선택됩니다.__

트리는 다음과 같이 표현할 수 있습니다.
<center>
$f(x) = \sum_{j=1}^{T} w_j {\mathrm I}(x\in R_j)$
</center>

여기서 $w_j$는 j번째 잎 노드의 __가중치__라고 하며, ${R_j}(j=1,2,…,T)$는 트리의 __구조__라고 합니다. $T$는 이 트리의 잎 노드 수입니다. ${\mathrm I}(x\in R_j)$는 샘플 $x$가 영역 $R_j$에 속하면 1이고, 그렇지 않으면 0입니다. 따라서 샘플 $x$가 영역 $j$에 해당하는 경우 $f(x)$는 $w_j$의 값으로 예측합니다. 아래 그림에는 $T=4$가 있습니다. 샘플 공간은 $R_1={x:x\leq 1.21}$, $R_2={x:1.21<x\leq 2.31}$, $R_3={x:2.31<x \leq 3.165}$, $R_4={x:x>3.165}$의 네 가지 분리된 영역으로 나뉩니다. 여기서 $x$는 flavanoids 값입니다. 네 가지 영역의 가중치는 $w_1=647.889$, $w_2=525.357$, $w_3=882.209$ 및 $w_4=1204.35$입니다.

<img src="https://imgur.com/YDuPS3X.png" width="500px"/>

이제 훈련 세트에 $n$개의 샘플이 있다고 가정하면, 우리의 목표는 비용 함수를 최소화하도록 트리를 훈련시키는 것입니다.

<center>
$J(f)=\sum_{i=1}^{n}L(y_i, f(x_i))=\sum_{i=1}^{n}L(y_i, \sum_{j=1}^{T}w_j {\mathrm I}(x\in R_j))$
</center>

여기서 $y_i$는 샘플 $x_i$의 실제 응답입니다. __일반적으로 말해서, 훈련은 세 단계로 구성됩니다. (1) 고정된 구조에 주어진 가중치를 학습하는 것, (2) 구조를 배우는 것, (3) 가지치기__ 우리는 그것들을 하나씩 살펴볼 것입니다.

## 고정된 구조의 가중치 학습

영역 $R_j$의 분리된 특성과 모든 영역의 결합이 완전한 표본 공간을 제공한다는 사실을 고려할 때, 위의 비용 함수는 다음과 같이 추가적으로 쓸 수 있습니다.

<center>
$J(f)=\sum_{j=1}^{T}\sum_{x_i \in R_j}L(y_i, w_j)$
</center>

고정 트리 구조(즉, 모든 $R_j$가 정의 및 고정되고 $T$가 고정됨)를 고려할 때, $J(f)$를 최소화하는 것은 각 $j$에 대해 $\sum_{x_i \in R_j}L(y_i, w_j)$를 최소화하는 것과 같습니다. 그러므로, 고정된 구조를 고려할 때, 우리는 다음과 같이 서술할 수 있습니다.

<center>
$w_j^* = \underset{w}{\operatorname{argmin}}\sum_{x_i \in R_j}L(y_i, w)$
</center>


<font color='blue'>제곱 손실 함수</font>의 경우, 우리는 $w_j^* = \underset{w}{\operatorname{argmin}}\sum_{x_i \in R_j}(y_i-w)^2$를 가지고 있습니다. $\frac{\partial \sum_{x_i \in R_j}(y_i-w)^2}{\partial w}$를 $0$로 설정하면 다음과 같습니다.

<center>
$w_j^* = \frac{\sum_{x_i \in R_j}y_i}{n_j}$
</center>

여기서 $n_j$는 영역 $R_j$에 속하는 샘플 수입니다. __즉, 제곱 손실 함수의 경우 추정 가중치 $w_j$는 단순히 영역 $R_j$의 반응 평균입니다.__ __이것이 바로 우리가 결정 트리 회귀 모델 단원에서 설명한 것입니다.__ 손실 함수 $L$이 다르게 정의되면 $w_j^*$의 표현식이 달라집니다. 예를 들어, 우리가 절대적인 손실을 사용한다면, 여기서 $L(y_i,w)= |y_i-w|$, $w_j^*$는 영역 $R_j$에서 반응의 중위수여야 합니다.


## 구조 학습
이제 구조에 어떤 가중치가 주어져야 하는지 알았으므로 비용 함수는 다음과 같이 쓸 수 있습니다.

<center>
$J(f)=\sum_{j=1}^{T}\sum_{x_i \in R_j}L(y_i, w_j^*)  \equiv \sum_{j=1}^{T} L_j^*$
</center>

여기서 $L_j^*$는 잎 노드 $j$에서 집계된 손실이라고 합니다. __추정 가중치 $w_j^*$가 사용됩니다.__ 트리를 훈련하고 잎 노드 $k$의 다음 분할을 고려하고 있다고 가정한다면, 이 분할이 발생하기 전에 전체 비용은 다음과 같습니다.

<center>
$J_{\mathrm {before}}= \sum_{j\neq k} L_j^* + L_k^*$
</center>

잎 노드 $k$에서 분할 후, 우리는 왼쪽 자식 노드("L")와 오른쪽 자식 노드("R")를 얻습니다. 새로운 비용은 다음과 같습니다.

<center>
$J_{\mathrm {after}}= \sum_{j\neq k} L_j^* + L_L^*+L_R^*$
</center>

고려된 분할의 <font color='blue'>이득</font>은 다음과 같이 정의됩니다.

<center>
${\mathrm {Gain}}=J_{\mathrm {before}} - J_{\mathrm {after}} = L_k^* - (L_L^*+L_R^*)$ 
</center>

이득이 클수록 비용 함수 $J$의 감소가 커집니다. __트리 훈련의 모든 단계에서 가능한 모든 분할(분할할 노드, 분할할 특성, 특성 임계값)에 대한 이득을 계산하고 이득을 최대화하는 분할을 선택합니다.__ 다른 분할을 고려할 때 $L_k^*$는 상수이므로 $(L_L^+L_R^*)$를 최소화하면 됩니다. <font color='blue'>손실 제곱</font>의 경우 다음과 같은 이점이 있습니다.

<center>
$L_L^*+L_R^* = \sum_{x_i \in \mathrm{left}}(y_i-\bar{y}_{\mathrm{left}})^2 + \sum_{x_i \in \mathrm{right}}(y_i-\bar{y}_{\mathrm{right}})^2 = n_{\mathrm{left}}\mathrm{MSE}_{\mathrm{left}} +  n_{\mathrm{right}}\mathrm{MSE}_{\mathrm{right}}$ 
</center>

여기서 $n_{\mathrm{left(right)}}$는 왼쪽(오른쪽) 노드의 교육 샘플 수입니다. $\bar{y}_{\mathrm{left(right)}}$는 이전 단계에서 보여드린 것처럼 제곱 손실 하에서 왼쪽(오른쪽) 노드에 대한 추정 가중치입니다. __최소화할 이 양은 결정 트리 회귀 모델 단원에서 정의한 비용 함수 $J(k,t_k)$에 비례합니다. 즉, 이 단원에서 정의한 모든 것은 일반적인 CART 알고리즘에 제곱 손실이 적용되는 특정한 경우입니다.__

## 가지치기(Pruning)

트리 훈련 중에 가능한 모든 이득이 음수이면 분할을 포기할 수 있습니다. 그러나 현재 분할 후 추가 분할은 긍정적인 이득을 얻을 수 있고 전체적으로 비용을 절감할 수 있기에 이결정은 너무 국지적일 수 있습니다. 이 문제를 해결하기 위해 트리는 일반적으로 특정 정지 기준(예: 최대 잎 노드 수에 도달)이 충족될 때까지 성장합니다. 성장 기간 동안, 이득이 음수인 경우에도 각 단계에서 이득이 최대화된 최적화된 분할이 수행됩니다. __트리가 성장한 후에는 상향식 방식으로 음의 이득을 갖는 노드가 제거되므로 전체 비용이 더욱 절감됩니다.__ 트리를 자르는 이 기술은 __가지치기__라고 불립니다.

튜토리얼 7에서는 트리 부스팅 알고리즘(그라디언트 부스팅 머신 및 XGBoost)에 대해 설명합니다. 위의 알고리즘이 어떻게 작동하는지 이해하는 것은 트리 부스팅을 이해하는 데 중요합니다.

이 노트북은 아래의 링크에 대하여 정리한 문서입니다.

- [Machine Learning 5 Random Forests](https://www.kaggle.com/code/fengdanye/machine-learning-5-random-forests)
