---
layout: post
title: Machine_Learning_For_Beginners
subtitle: Machine_Learning_Assignment7
categories: Machine_Learning
tags: Machine_Learning
use_math: true
---

## **분류 소개**
[Getting started with classification](https://github.com/codingalzi/ML-For-Beginners/tree/main/4-Classification) 내용 정리

분류는  지도 학습의 한 형태이며 일반적으로 분류는 이진 분류와 다중 클래스 분류의 두 그룹으로 나뉜다.


*   **선형 회귀** 를 사용하면 변수 사이 관계를 예측하고 새로운 데이터 포인트로 라인과 엮인 위치에 대한 정확한 예측 가능      ex) 9월과 
12월의 호박 가격 예측 가능
*   **로지스틱 회귀** 는 "이진 범주"를 찾는 데 유용함             ex) 이 가격대에서 이 호박은 주황색인지 아닌지

분류는 데이터 포인트의 레이블 또는 클래스를 결정하는 다른 방법을 결정하기 위해 다양한 알고리즘을 사용한다. 이 요리 데이터를 사용하여 재료 그룹을 관찰하여 원산지 요리를 결정할 수 있는지 알아보자.

## **chap1**

### **데이터 불러오기**
---

```python
# imblearn 설치하기 위해 pip install

pip install imblearn
```

```python
# 데이터를 가져오는 데 필요한 패키지를 가져오고 시각화할 수 있으며, imblearn에서 SMOT도 가져올 수 있다.
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from imblearn.over_sampling import SMOTE
```
```python
# 데이터 가져와서 읽기
# read_csv()를 사용하여 해당 csv파일의 내용을 읽고 변수 df에 저장
df  = pd.read_csv('cuisines.csv')
# 앞에서부터 5개의 데이터를 확인
df.head()
```

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini | 
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- | 
| 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        | 
| 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        | 
| 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        | 
| 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        | 
| 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        | 

5 rows × 385 columns
```python
# info()를 호출하여 데이터에 대한 정보를 가져오기
df.info()
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2448 entries, 0 to 2447
Columns: 385 entries, Unnamed: 0 to zucchini
dtypes: int64(384), object(1)
memory usage: 7.2+ MB
```

**요리 당 데이터의 분포를 알아보기**

```python
# barh() 호출하여 데이터를 막대 그래프로 출력
df.cuisine.value_counts().plot.barh() # 데이터 분포가 고르지 않음 
```


![res_1](http://jjhcom.github.io/assets/images/banners/res_1.png)

```python
# 요리 당 얼마나 많은 데이터 사용할 수 있는지 확인
thai_df = df[(df.cuisine == "thai")]
japanese_df = df[(df.cuisine == "japanese")]
chinese_df = df[(df.cuisine == "chinese")]
indian_df = df[(df.cuisine == "indian")]
korean_df = df[(df.cuisine == "korean")]

print(f'thai df: {thai_df.shape}')
print(f'japanese df: {japanese_df.shape}')
print(f'chinese df: {chinese_df.shape}')
print(f'indian df: {indian_df.shape}')
print(f'korean df: {korean_df.shape}')
```
```
thai df: (289, 385)
japanese df: (320, 385)
chinese df: (442, 385)
indian df: (598, 385)
korean df: (799, 385)
```

이제 데이터를 더 깊이 파고들어 요리 당 전형적인 재료가 무엇인지 배울 수 있다.

음식 사이에 혼란을 일으키는 반복적인 데이터를 지워야 하는데, 이 문제에 대해 알아볼 것이다.

파이썬에서 `creat_ingredient()` 함수를 만들어 **성분 데이터 프레임을 생성**한다.

이 기능은 **도움이 되지 않는 열을 삭제**하는 것부터 시작하여 **성분을 개수에 따라 정렬**한다.

```python
# 도움이 되지 않는 열을 삭제하고 성분을 개수에 따라 정렬하혀 성분 데이터 프레임을 생성하는 함수 작성
def create_ingredient_df(df):
    ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
    ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
    ingredient_df = ingredient_df.sort_values(by='value', ascending=False, inplace=False)
    return ingredient_df
```

```python
# 요리별로 가장 인기 있는 10대 식재료에 대한 아이디어 얻기(Thai)
thai_ingredient_df = create_ingredient_df(thai_df)
thai_ingredient_df.head(10).plot.barh()
```


![res_2](http://jjhcom.github.io/assets/images/banners/res_2.png)

```python
# 요리별로 가장 인기 있는 10대 식재료에 대한 아이디어 얻기(Japanese)
japanese_ingredient_df = create_ingredient_df(japanese_df)
japanese_ingredient_df.head(10).plot.barh()
```

![res_3](http://jjhcom.github.io/assets/images/banners/res_3.png)

```python
# 요리별로 가장 인기 있는 10대 식재료에 대한 아이디어 얻기(Chinese)
chinese_ingredient_df = create_ingredient_df(chinese_df)
chinese_ingredient_df.head(10).plot.barh()
```

![res_4](http://jjhcom.github.io/assets/images/banners/res_4.png)


```python
# 요리별로 가장 인기 있는 10대 식재료에 대한 아이디어 얻기(Indian)
indian_ingredient_df = create_ingredient_df(indian_df)
indian_ingredient_df.head(10).plot.barh()
```

![res_5](http://jjhcom.github.io/assets/images/banners/res_5.png)


```python
# 요리별로 가장 인기 있는 10대 식재료에 대한 아이디어 얻기(Korean)
korean_ingredient_df = create_ingredient_df(korean_df)
korean_ingredient_df.head(10).plot.barh()
```

![res_6](http://jjhcom.github.io/assets/images/banners/res_6.png)



```python
# drop()을 호출하여 구별되는 요리 사이에 혼란을 일으키는 가장 일반적인 재료 삭제 -> 'rice', 'garlic', 'ginger'와 같은 일반적인 재료
feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
labels_df = df.cuisine #.unique()
feature_df.head()
```


|    | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0 |	0 |	0 |	0 |	0 |	0 |	0 |	0 |	0 |	0 |	0 |	... |	0 |	0 |	0 |	0 |	0 |	0 |	0 |	0 |	0 |	0 |
| 1 |	1 | 	0 | 	0 |	0 |	0 |	0 | 	0 |	0 |	0 |	0 |	... |	0 |	0 |	0 |	0 |	0 | 	0 |	0 |	0 |	0 |	0 |
| 2 |	0 |	0 |	0 |	0 | 	0 |	0 |	0 |	0 |	0 |	0 |	... |	0 |	0 |	0 |	0 |	0 |	0 |	0 |	0 |	0 |	0 | 
| 3 |	0 |	0 |	0 |	0 |	0 |	0 |	0 |	0 |	0 |	0 |	... |	0 |	0 |	0 |	0 |	0 |	0 |	0 |	0 |	0 |	0 |
| 4 |	0 |	0 |	0 |	0 |	0 |	0 |	0 |	0 |	0 |	0 |	... |	0 |	0 |	0 |	0 |	0 |	0 |	0 |	0 |	1 |	0 |

5 rows × 380 columns

이제 데이터를 정리했으므로 `SMOTE`, 즉 "Synthetic Minority Over-sampling Technique(합성 소수 과표본 기법)"을 사용하여 **균형**을 잡을 것이다.

```python
# fit_resample()을 호출하면 이 전략은 보간을 통해 새 샘플을 생성
oversample = SMOTE()
transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)

# 성분 당 레이블 수 확인 -> 정제된 균형이 맞는 데이터가 출력
print(f'new label count: {transformed_label_df.value_counts()}')
print(f'old label count: {df.cuisine.value_counts()}')
```
**데이터의 균형을 유지**함으로써 *데이터를 분류할 때 더 나은 결과*를 얻을 수 있다.

**이진분류기**를 생각해보면, 대부분의 **데이터가 `하나`의 클래스**인 경우 *ML 모델은 단지 더 많은 데이터가 있다*는 이유로 **해당 클래스를 더 자주 예측**한다.

**데이터의 균형**을 맞추는 것은 왜곡된 데이터가 있고, 이러한 *왜곡 데이터와 같은* **불균형을 제거**하는 데 도움이 된다.

```
new label count: thai        799
indian      799
japanese    799
chinese     799
korean      799
Name: cuisine, dtype: int64
old label count: korean      799
indian      598
chinese     442
japanese    320
thai        289
Name: cuisine, dtype: int64
```

```python
# 레이블 및 기능을 포함한 균형 잡힌 데이터를 파일로 내보낼 수 있는 새 데이터 프레임에 저장
transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
# 데이터를 한 번 더 살펴보기(맨 앞 5가지의 데이터 확인)
transformed_df.head()
```

|  |	cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot	| armagnac | artemisia | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast |	yogurt |	 zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
|0|	indian	|0	|0	|0	|0	|0	|0	|0	|0	|0	|...	|0	|0	|0	|0	|0	|0	|0|	0|	0|	0|
|1|	indian	|1	|0	|0	|0	|0	|0	|0	|0	|0	|...	|0	|0	|0	|0	|0	|0	|0|	0|	0|	0|
|2|	indian	|0	|0	|0	|0	|0	|0	|0	|0	|0	|...	|0	|0	|0	|0	|0	|0	|0|	0|	0|	0|
|3|	indian	|0	|0	|0	|0	|0	|0	|0	|0	|0	|...	|0	|0	|0	|0	|0	|0	|0|	0|	0|	0|
|4|	indian	|0	|0	|0	|0	|0	|0	|0	|0	|0	|...	|0	|0	|0	|0	|0	|0	|0|	0|	1|	0|
|...|	...	|...	|...	|...	|...	|...	|...	|...	|...	|...	|...	|...	|...	|...	|...	|...	|...	|...|	...|	...|	...|
|3990|	thai	|0	|0	|0	|0	|0	|0	|0	|0	|0	|...	|0	|0	|0	|0	|0	|0	|0|	0|	0|	0|
|3991|	thai	|0	|0	|0	|0	|0	|0	|0	|0	|0	|...	|0	|0	|0	|0	|0	|0	|0|	0|	0|	0|
|3992|	thai	|0	|0	|0	|0	|0	|0	|0	|0	|0	|...	|0	|0	|0	|0	|0	|0	|0|	0|	0|	0|
|3993|	thai	|0	|0	|0	|0	|0	|0	|0	|0	|0	|...	|0	|0	|0	|0	|0	|0	|0|	0|	0|	0|
|3994|	thai	|0	|0	|0	|0	|0	|0	|0	|0	|0	|...	|0	|0	|0	|0	|0	|0	|0|	0|	0|	0|

3995 rows × 381 columns

```python
# 데이터 정보 확인
transformed_df.info()
# 다음 교육에서 사용할 수 있도록 데이터 복사본을 저장
transformed_df.to_csv("../data/cleaned_cuisines.csv")
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3995 entries, 0 to 3994
Columns: 381 entries, cuisine to zucchini
dtypes: int64(380), object(1)
memory usage: 11.6+ MB
```

🎈 **데이터 폴더를 살펴보고 이진 또는 다중 클래스 분류에 적합한 데이터 셋이 있는지 확인하고 해당 데이터 세트에 대해 어떤 질문을 할지 생각해보기**

## Classifiers 1

### 요리 분류기
- "Introduction"에서 저장한 모든 음식에 대한 **균형 잡힌 깨끗한 데이터로 가득 찬 데이터 세트**를 사용할 것이다.
- 이 데이터 세트를 **다양한 분류기**와 함께 사용하여 *재료 그룹을 기반*으로 **특정 국가 음식을 예측**할 수 있다.
- 이렇게 하는 동안, **분류 작업**에 알고리즘을 활용할 수 있는 몇 가지 방법에 대해 자세히 알아볼 수 있다.

### 연습
**국민 요리 예언**
```python
# 파일 불러오기 
import pandas as pd
cuisines_df = pd.read_csv("cleaned_cuisines.csv")

# 맨 앞 5개의 데이터 확인
cuisines_df.head()
```


|     | Unnamed: 0| cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0      | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1        | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2        | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3        | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4       | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |


```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
from sklearn.svm import SVC
import numpy as np

# 훈련을 위해 두개의 데이터프레임으로 X와 Y좌표를 나눈다.
# 요리는 라벨 데이터프레임이 될 수 있다.
cuisines_label_df = cuisines_df['cuisine']
# 데이터 확인
cuisines_label_df.head()
```
```
0    indian
1    indian
2    indian
3    indian
4    indian
Name: cuisine, dtype: object
```


```python
# 'Unnamed: 0'의 열과 'cuisine' 열을 'drop()'을 호출하여 삭제 -> 나머지 데이터를 교육 가능한 형상으로 저장
cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
cuisines_feature_df.head() # 'Unnamed: 0'과 'cuisine' 
```

|     | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
|------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2    | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3    | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |


### 분류기 선택

데이터가 깨끗해지고, 훈련 준비가 되었으므로 작업에 사용할 **알고리즘**을 결정해야 한다.

`Scikit-learn` 그룹은 **지도 학습**에서 분류되며, 이 범주에서 분류할 수 있는 다양한 방법을 찾을 수 있다.

**모든 분류 기법이 포함된 방법**
- Linear Models (선형 모델)
- Support Vector Machines (서포트 벡터 머신)
- Stochastic Gradient Descent (확률적 경사 하강법)
- Nearest Neighbors (NN)
- Gaussian Processes (가우시안 프로세스)
- Decision Trees (의사 결정 트리)
- Ensemble methods (voting Classifier) (앙상블 기법)
- Multiclass and multioutput algorithms (multiclass and multilabel classification, multiclass-multioutput classification) (다중 클래스 및 다중 출력 알고리즘)
  - 신경망을 사용하여 데이터를 분류할 수도 있지만, 이 범위를 벗어난다.


종종 여러 개를 훑어보고 좋은 결과는 찾는 것이 **테스트하는 방법**이다.

`Scikit-learn`은 `KNeighbors`, `SVC`, `GaussianProcessClassifier`, `DecisionTreeClassifier`, `RandomForestClassifier`, `MLPClassifier`, `AdaBoostClassifier`, `GaussianNB` 그리고 `QuadraticDiscrinationAnalysis`를 **비교**하고 **시각화된 결과**를 보면서, **생성된 데이터세트**에 대해 나란히 비교한다.

![classifiers1](http://jjhcom.github.io/assets/images/banners/classifiers1.png)


    - Scikit-learn의 설명서에서 생성된 그림
    - AutoML은 이러한 비교를 클라우드에서 실행하여 데이터에 가장 적합한 알고리즘을 선택할 수 있도록 함으로써 이 문제를 해결

하지만, 넓게 추축하는 것보다 더 나은 방법은 **다운로드 가능한 `ML Cheat sheet`의 아이디어**를 따르는 것이다.

이때, **다중 클래스 문제**에 대해 몇 가지 **선택사항**이 있다는 것을 발견한다.

![classifiers2](http://jjhcom.github.io/assets/images/banners/classifiers2.png)
  > 다중 클래스 분류 옵션을 자세히 설명하는 Microsoft 알고리즘 차트 시트의 분류

### 추론

제약 조건들을 고려할 때, 다른 접근 방식들을 통해 추론할 수 있는지 알아볼 것이다.
- **신경망은 너무 무겁다**
  - 깨끗하지만, 최소한의 데이터 세트와 노트북을 통해 로컬로 교육을 실행하고 있다는 사실을 감안할 때 신경망은 이 작업에 너무 무겁다
- **2-클래스 분류기가 없다**
  - OvA를 배제하기 위해, 2-클래스 분류기를 사용하지 않는다.
- **의사 결정 트리 또는 로지스틱 회귀 분석이 작동할 수 있다**
  - 의사 결정 트리가 작동하거나 다중 클래스 데이터에 대해 로지스틱 회귀 분석을 수행할 수 있다.
- **멀티클래스 부스트 결정 트리는 다른 문제를 해결한다**
  - 멀티클래스 부스트 결정 트리는 순위를 구축하도록 설계된 작업과 같은 비모수 작업에 가장 적합하므로 유용하지 않다.

### Scikit-learn 사용하기
- `Scikit-learn`을 사용하여 데이터를 분석할 것이다.
- `Scikit-learn`에서는 `로지스틱 회귀 분석`을 사용하는 여러 가지 방법이 있다.
-  전달할 매개 변수를 살펴봐야 한다.
- 기본적으로 `Scikit-learn`에게 `로지스틱 회귀 분석`을 수행하도록 요청할때 지정해야 하는 중요한 두가지의 매개 변수가 있다.  
  - `multi_class` : 특정 동작을 적용 
  - `solver` : 사용할 알고리즘
  - 모든 `solver`와 `multi_class의 값`를 쌍으로 구성할 수 있는 것은 아니다.

**멀티 클래스**사례에서의 **훈련 알고리즘**
- `multi_class` 옵션이 `ovr`로 지정되었을 때 **OVR 체계 사용**
- `multi_class` 옵션이 `multinominal`로 지정되면 **교차 엔트로피 손실을 사용**
  - 현재 `multinominal` 옵션은 `lbfgs`, `sag`, `saga`, `newton-cg` solvers에만 지원된다.


📌 여기서 **'scheme'** 은 `ovr`이나 `multinominal`일 수 있다.
- 로지스틱 회귀는 실제로 이진 분류기를 지원하도록 설계되었기 때문에, 이러한 체계를 통해 다중 클래스 분류 작업을 더 잘 처리할 수 있다.

📌 **'solver'** 는 "*최적화 문제에 사용할 알고리즘*"으로 정의된다.

📌 `Scikit-learn`은 `solvers`가 다양한 종류의 데이터 구조에서 나타나는 **다양한 문제를 처리하는 방법**을 설명하는 **표를 제공**한다.

![classifiers3](http://jjhcom.github.io/assets/images/banners/classifiers3.png)

### 연습 
**데이터 나누기**

이전 수업에서 후자에 대해 알게 된 이후, 첫번째 교육 시행을 위해 **로지스틱 회귀 분석**에 초점을 맞출 수 있다.

```python
# train_test_split()을 호출하여 데이터를 훈련, 테스트 그룹으로 나눈다,
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

**로지스틱 회귀 분석 적용**
다중 클래스 케이스를 사용 중이므로, 사용할 `scheme`과 설정할 `solver`를 선택해야 한다.

**훈련**을 위해, `다중 클래스 설정`과 `liblinear solver`을 함께 **로지스틱 회귀 분석**을 사용한다.

```python
# multi_class를 'ovr'로 지정하고, solvr를 'linbear'로 설정한 "로지스틱 회귀 분석" 생성
lr = LogisticRegression(multi_class='ovr',solver='liblinear')
model = lr.fit(X_train, np.ravel(y_train))

accuracy = model.score(X_test, y_test)
print ("Accuracy is {}".format(accuracy))
```
*종종 기본값으로 설정된 `lbfgs`와 같은 다른 solver를 사용해도 된다*

```
Accuracy is 0.8065054211843202
```
**정확도**가 약 80%를 넘는다.

```python
# 하나의 데이터 행을 테스트하면서 모델이 작동하는 것을 확인
print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
print(f'cuisine: {y_test.iloc[50]}')
```
```
ingredients: Index(['chicken', 'cilantro'], dtype='object')
cuisine: thai
```
*다른 행 번호를 사용해서도 결과 확인해봐도 된다.*

```python
# 예측의 정확성 확인
test= X_test.iloc[50].values.reshape(-1, 1).T
proba = model.predict_proba(test)
classes = model.classes_
resultdf = pd.DataFrame(data=proba, columns=classes)

topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
topPrediction.head()
```

|              |                0 |
| -------- | ------------ |
| indian   |  0.715851   |
| chinese |  0.229475   |
| japanese |  0.029763 |
| korean   |  0.017277  |
| thai      |  0.007634   |

**인도 요리**가 가장 좋은 추측이며, 그럴 확률이 약 71%로 높다.

*모델이 왜 인도 요리가 가장 좋다고 확신하는지 설명할 수 있는지 생각해봐야 한다.*

```python
# 회귀 분석 수업에서 했던 것처럼 분류 보고서를 인쇄하여 더 자세히 확인 
y_pred = model.predict(X_test)
print(classification_report(y_test,y_pred))
```
```
              precision    recall  f1-score   support

     chinese       0.78      0.70      0.73       252
      indian       0.91      0.93      0.92       242
    japanese       0.75      0.78      0.76       225
      korean       0.83      0.81      0.82       243
        thai       0.77      0.83      0.80       237

    accuracy                           0.81      1199
   macro avg       0.81      0.81      0.81      1199
weighted avg       0.81      0.81      0.81      1199
```

🎈 **정리된 데이터를 사용하여 일련의 재료를 기반으로 국가 요리를 예측할 수 있는 기계 학습 모델을 구축했다. Scikit-learn이 제공하는 다양한 데이터 분류 옵션을 읽어보는 것도 좋다. 'solver'의 개념을 더 깊이 파고들어서 이면에서 무슨일이 벌어지는지 이해할 수 있을 것이다.**




## Classifiers 2 

**숫자 데이터를 분류**하는 더 많은 방법을 살펴볼 것이다.

또한, 한 분류기를 **다른 분류기로 선택하는 데 미치는 영향**에 대해서도 배울 수 있을 것이다.

### 분류 지도 

이전에는, `Microsoft`의 차트시트를 사용하여 **데이를 분류할 때 사용할 수 있는 다양한 옵션**에 대해서 배웠다.

`Scikit-learn`은 유사하지만, *추정기(분류기)의 범위를 더 좁히는*데 도움이 될 수 있는 **세분화된 시트(granular cheat sheet)**를 제공한다.

![classifiers4](http://jjhcom.github.io/assets/images/banners/classifiers4.png)

*데이터를 사용해 볼 추정기와 관련된 문제에 접근하는 방법에 대한 대략적인 지침을 사용자에게 제공하기 위해 설계*

위의 지도는 **데이터를 명확하게 파악**하면 **의사 결정**에 이르는 경로로 나아갈 수 있으므로 유용하다.

따라가기에 매우 도움이 되는 길은 다음과 같다.
- 50개 이상의 샘플을 가지고 있다
- 범주를 예측하고 싶다
- 레이블이 지정된 데이터가 있다
- 10만개 미만의 샘플을 가지고 있다
- 선형 SVC를 선택할 수 있다
- 수치 데이터를 가지고 있어서 작동이 되지 않는다면, KNeighbors Classifier을 사용해볼 수 있다.
- 그래도 작동이 되지 않는다면, SVC 및 앙상블 분류기를 사용해 볼 수 있다.


### 연습
**데이터 나누기**
```python
# 사용할 라이브러리 불러오기
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
import numpy as np
```

```python
# 데이터를 훈련과 테스트로 분리
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

### 선형 SVC 분류기
- `SVC`(Support-Vector Clustering)는 ML 기술인 `Support-Vector Macine` 제품군의 하위 제품이다.
- 이 방법에서 **'kernel'** 을 선택하여, **레이블을 군집화하는 밥법을 결정**할 수 있다.
- `C` 인자는 **파라미터의 영향을 조절**하는 **"정규화"** 를 의미한다.
- **커널**은 여러 가지 중 하나일 수 있다.
  - 여기서는 `선형 SVC`를 활용하도록 `linear` 설정
- **확률**은 기본적으로 `'false'`로 설정된다.
  - 여기서는 **확률 추정치를 수집하기 위해** `true`로 설정
- **확률값을 얻기** 위해, **데이터를 섞는** `무작위 상태`를 '0'으로 설정한다.

### 연습
**선형 SVC 적용**
```python
C = 10

# 다양한 분류기 생성
classifiers = {'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)}
```

```python
# 선형 SVC를 사용하여 모델을 교육하고 확인 -> 대략 81%로의 정확성
n_classifiers = len(classifiers)

for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(X_train, np.ravel(y_train))

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
    print(classification_report(y_test,y_pred))
```
```
Accuracy (train) for Linear SVC: 81.1% 
              precision    recall  f1-score   support

     chinese       0.72      0.78      0.75       246
      indian       0.90      0.87      0.89       259
    japanese       0.77      0.74      0.76       226
      korean       0.86      0.81      0.84       236
        thai       0.80      0.84      0.82       232

    accuracy                           0.81      1199
   macro avg       0.81      0.81      0.81      1199
weighted avg       0.81      0.81      0.81      1199
```
### K-Neighbors 분류기
`K-Neighbors`는 ML 방법의 "neighbors" 계열의 일부로, **지도 학습**과 **비지도 학습** 모두에 사용할 수 있다.

이 방법에서는, 일반화된 레이블이 그 데이터에 대해 예측할 수 있도록 미리 정의된 점이 생성되고 데이터는  이 점 주변에 수집된다.

### 연습
**K-Neighbors 분류기 적용**

*이전의 분류기도 좋았고, 데이터에 잘 작동했지만, 더 좋은 정확도를 얻을 수 있으므로 `K-Neighbors 분류기`를 시도해볼 것이다.*

```python
# 분류기 배열에 줄을 추가하여 훈련하고 확인 -> 71%로 좋지 않은 결과가 나온다
classifiers = {'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0), 'KNN classifier': KNeighborsClassifier(C)}
```

```
Accuracy (train) for KNN classifier: 73.1% 
              precision    recall  f1-score   support

     chinese       0.66      0.73      0.69       246
      indian       0.90      0.75      0.81       259
    japanese       0.62      0.81      0.70       226
      korean       0.92      0.56      0.69       236
        thai       0.69      0.81      0.75       232

    accuracy                           0.73      1199
   macro avg       0.76      0.73      0.73      1199
weighted avg       0.76      0.73      0.73      1199
```

### Support Vector 분류기

`Support-Vector 분류기`는 **분류** 및 **회귀 작업**에 사용되는 ML 메서드의 `Support-Vector Machine` 제품군의 일부이다.

`SVM`은 **두 범주간의 거리를 최대화**하기 위해 **"공간 내 지점에 훈련 예제를 매핑**한다.

후속 데이터는 *해당 범주를 예측*할 수 있도록 이 공간에 매핑된다.

### 연습
**Support Vector Classifier 적용**

```python
# Support Vector Classifier 적용하여 훈련하고 확인-> 약 84%로 좋은 결과
classifiers = {'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0), 'KNN classifier': KNeighborsClassifier(C), 'SVC': SVC()}
```
```
Accuracy (train) for SVC: 84.2% 
              precision    recall  f1-score   support

     chinese       0.80      0.79      0.80       246
      indian       0.90      0.91      0.91       259
    japanese       0.84      0.77      0.80       226
      korean       0.90      0.84      0.87       236
        thai       0.78      0.90      0.84       232

    accuracy                           0.84      1199
   macro avg       0.84      0.84      0.84      1199
weighted avg       0.84      0.84      0.84      1199
```

### 앙상블 분류기

**앙상블 분류기**는 특히, `Random Forest`와 `AdaBoost`를 사용하여 성능을 확인해볼 것이다.

```python
# 랜덤 포레스트의 경우 성능이 좋다
classifiers = {'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0), 'KNN classifier': KNeighborsClassifier(C), 'SVC': SVC(), 'RFST': RandomForestClassifier(n_estimators=100), 'ADA': AdaBoostClassifier(n_estimators=100)}
```

```
Accuracy (train) for RFST: 84.9% 
              precision    recall  f1-score   support

     chinese       0.83      0.81      0.82       246
      indian       0.91      0.90      0.90       259
    japanese       0.83      0.78      0.81       226
      korean       0.87      0.86      0.87       236
        thai       0.80      0.89      0.84       232

    accuracy                           0.85      1199
   macro avg       0.85      0.85      0.85      1199
weighted avg       0.85      0.85      0.85      1199

Accuracy (train) for ADA: 71.4% 
              precision    recall  f1-score   support

     chinese       0.66      0.43      0.53       246
      indian       0.87      0.86      0.86       259
    japanese       0.56      0.66      0.61       226
      korean       0.76      0.80      0.78       236
        thai       0.70      0.82      0.75       232

    accuracy                           0.71      1199
   macro avg       0.71      0.71      0.71      1199
weighted avg       0.72      0.71      0.71      1199
```

이 머신 러닝의 방법은 **모델의 품질을 향상**시키기 위해 **"여러 기본 추정기의 예측을 결합"** 하는데, 이 예에서는 `랜덤 트리`와 `AdaBoost`를 사용해봤다.
- **`Random Forest`**
  - 평균화 방법
  - 과적합을 피하기 위해 무작위성이 주입된 "결정 트리"의 "숲"을 구축
  - n_estimators 매개변수는 트리 수로 설정
- **`AdaBoost`**
  - 분류기를 데이터 집합에 적합시킨 다음 해당 분류기의 복사본을 동일한 데이터 집합에 적합
  - 잘못 분류된 항목의 가중치에 초점을 맞추고 다음 분류자가 수정할 적합도를 조정

🎈**각 기술에는 조정할 수 있는 매개변수가 있는데, 각 인자의 기본 인자를 조사하고 모델의 품질에 대해 이러한 인자를 조정하는 것이 무엇을 의미하는지 생각해 봐야 한다**

## Applied

- 이전 교육에서 배운 몇가지 기술과 이 시리즈에서 사용된 맛있는 요리 데이터 세트를 사용하여 **분류 모델을 구축**한다.
- 또한, `Onnx의 웹 런타임`을 활용하여 **저장된 모델을 사용**할 수 있는 **작은 웹 앱**을 만들 것이다.
- 머신러닝의 가장 유용한 실용 중 하나는 **추천 시스템을 구축**하는 것이고, 이 방향으로 가는 첫 단계를 학습해볼 것이다.
  - [적용 참고 영상](https://www.youtube.com/watch?v=17wdM9AHMfg)

배우게 될 방법
- 모델을 작성하고 이를 Onnx 모델로 저장하는 방법
- Netron을 사용하여 모델을 검사하는 방법
- 추론을 위해 웹 앱에서 모델을 사용하는 방법

### 모델 구축하기
- 응용 ML 시스템을 구축하는 것은 비즈니스 시스템에 이러한 기술을 활용하는 데 있어 중요한 부분이다.
- Onnx를 사용하여 웹 응용 프로그램 내에서 모델을 사용할 수 있으므로 필요한 경우 오프라인 컨텍스트 모델을 사용할 수 있다.
- 이전 수업에서, UFO 목격에 대한 회귀 모형을 만들고 그것을 "pickled"하여 플라스크 앱에 사용했다.
  - 이 구조는 매우 유용하지만, Full-Stack 파이썬 앱이며, 요구사항은 자바스크립트 애플리케이션 사용을 포함할 수 있다.


### 연습
**분류 모델 훈련**

우리가 사용했던 정제된 요리 데이터를 사용하여 **분류 모델을 훈련**시킨다.
```python
# 필요한 라이브러리 불러오기
# skl2onnx는 Scikit-learn 모델을 Onnx 형식으로 변환하기 위해 사용
!pip install skl2onnx
import pandas as pd 
```

```python
# CSV 파일을 읽어와서 데이터 확인
data = pd.read_csv('cleaned_cuisines.csv')
data.head()
```

|     | Unnamed: 0| cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0      | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1        | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2        | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3        | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4       | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |

```python
# 처음 두개의 불필요한 열을 삭제하고, 남아있는 데이터를 'X'로 저장
X = data.iloc[:,2:]
X.head()
```


|        | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| ---- | -------- | ---------- | ----- | ------------ | ------ | --------------- | -------- | ----------- | - | --------- | ------------- | ------------ | ---------------------------- | ----- | ------ | ----- | ----- | -------- | ---------- | 
| 0   | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2    | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3    | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |

```python
# 레이블을 'y'로 저장
y = data[['cuisine']]
y.head()
```

|  | cuisine |
|---|--------|
|0 |	indian |
|1 |	indian |
|2 |	indian |
|3 |	indian |
|4 |	indian |

```python
# 정확도가 좋았던 'SVC 라이브러리'를 사용하여 훈련
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
```

```python
# 훈련과 테스트 셋으로 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
```

```python
# SVC분류기 모델을 구축
model = SVC(kernel='linear', C=10, probability=True,random_state=0)
model.fit(X_train,y_train.values.ravel())
```

```
SVC(C=10, kernel='linear', probability=True, random_state=0)
```

```python
# predict()를 호출하면서 모델을 테스트
y_pred = model.predict(X_test)

# 모델의 품질 확인하기 위해 확인
print(classification_report(y_test,y_pred))
```

```
              precision    recall  f1-score   support

     chinese       0.67      0.70      0.68       243
      indian       0.89      0.85      0.87       238
    japanese       0.83      0.72      0.77       237
      korean       0.81      0.75      0.78       226
        thai       0.71      0.84      0.77       255

    accuracy                           0.77      1199
   macro avg       0.78      0.77      0.78      1199
weighted avg       0.78      0.77      0.77      1199
```

### Onnx로 모델 전환 

**적절한 텐서 수**로 변환해야 한다.

이 데이터 집합에는 380개의 성분이 나열되어 있으므로 `FloatTensorType`에 이 숫자를 기록해야 한다.

```python
# 380개의 수를 tensor를 사용하여 변환 
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, 380]))]
options = {id(model): {'nocl': True, 'zipmap': False}}
```

```python
# onx를 생성하고 'model.onnx'로 저장
onx = convert_sklearn(model, initial_types=initial_type, options=options)
with open("./model.onnx", "wb") as f:
    f.write(onx.SerializeToString())
```

*변환 스크립트에서 옵션을 전달할 수 있다.*
  - *이 경우, 'nocl'을 True로, 'zipmap'을 False로 전달*
- *이 모델은 분류 모델이기 때문에 필수는 아니지만, 사전 목록을 생성하는 ZipMap을 제거할 수 있다.*
- *'nocl'은 모델에 포함된 클래스 정보를 참조한다.*
- *'nocl'을 True로 지정하면, 모델 크기를 줄인다.*


### 모델 확인

`Onnx 모델`은 비쥬얼 스튜디오 코드에서 잘 보이지 않지만, 많은 연구자들이 **모델을 시각화**하기 위해 사용하는 매우 좋은 무료 소프트웨어이다.
- `Netron`을 다운로드하고 `model.onnx`파일을 연다.
  - *380개의 입력과 분류기가 나열된 단순 모델을 시각화 할 수 있다*

![applied1](http://jjhcom.github.io/assets/images/banners/applied1.png)

`Netron`은 **모델을 보는 데 유용한 도구**이다.

이제 웹 앱에서 이 깔끔한 모델을 사용할 준비가 되었다. 냉장고 안을 볼 때 유용하게 사용할 수 있는 앱을 만들고, 모델에 의해 결정되는 대로, 주어진 요리를 요리하기 위해 어떤 남은 재료의 조합을 사용할 수 있는지 알아볼 것이다.

### 추천 웹 앱 구축

웹 앱에서 직접 모델을 사용할 수 있다.

또한, 이 구조는 필요한 경우 **로컬** 및 **오프라인**에서도 실행될 수 있다.

`model.onnx` 파일을 저장한 폴더와 동일한 폴더에 `index.html`를 생성하면서 시작해볼 것이다.


**index.html 파일에 마크업 추가**
```html
<!DOCTYPE html>
<html>
    <header>
        <title>Cuisine Matcher</title>
    </header>
    <body>
        ...
    </body>
</html>
```

**본문 태그 내에서 작업하면서, 일부 성분을 반영하는 확인 목록을 보여주기 위해 약간의 마크업 추가**
```html
<h1>Check your refrigerator. What can you create?</h1>
        <div id="wrapper">
            <div class="boxCont">
                <input type="checkbox" value="4" class="checkbox">
                <label>apple</label>
            </div>
        
            <div class="boxCont">
                <input type="checkbox" value="247" class="checkbox">
                <label>pear</label>
            </div>
        
            <div class="boxCont">
                <input type="checkbox" value="77" class="checkbox">
                <label>cherry</label>
            </div>

            <div class="boxCont">
                <input type="checkbox" value="126" class="checkbox">
                <label>fenugreek</label>
            </div>

            <div class="boxCont">
                <input type="checkbox" value="302" class="checkbox">
                <label>sake</label>
            </div>

            <div class="boxCont">
                <input type="checkbox" value="327" class="checkbox">
                <label>soy sauce</label>
            </div>

            <div class="boxCont">
                <input type="checkbox" value="112" class="checkbox">
                <label>cumin</label>
            </div>
        </div>
        <div style="padding-top:10px">
            <button onClick="startInference()">What kind of cuisine can you make?</button>
        </div> 
```
- 각 확인란에는 값이 지정
  - 이는 데이터 세트에 따라 성분이 발견되는 인덱스를 반영
  - 예를 들어, 알파벳 목록에서 'Apple'은 다섯 번째 열을 차지하기 때문에, 0에서 숫자를 시작할 때 값은 '4'
- 성분 스프레드 시트를 참조하여 특정 성분의 색인을 찾기 가능
- `index.html`에서 작업을 계속하고, 최종 종료 `</div>` 뒤에 모델이 호출되는 스크립트 블록 추가


**Onnx Runtime 실행**
```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
```
*Onnx Runtime은 **최적화와 사용할 API를 포함한** 광범위한 `하드웨어 플랫폼`에서 **Onnx 모델을 실행할 수 있도록**하는데 사용*

**Runtime이 설치되면, 호출 가능**
```html
<script>
    const ingredients = Array(380).fill(0);
    
    const checks = [...document.querySelectorAll('.checkbox')];
    
    checks.forEach(check => {
        check.addEventListener('change', function() {
            // toggle the state of the ingredient
            // based on the checkbox's value (1 or 0)
            ingredients[check.value] = check.checked ? 1 : 0;
        });
    });

    function testCheckboxes() {
        // validate if at least one checkbox is checked
        return checks.some(check => check.checked);
    }

    async function startInference() {

        let atLeastOneChecked = testCheckboxes()

        if (!atLeastOneChecked) {
            alert('Please select at least one ingredient.');
            return;
        }
        try {
            // create a new session and load the model.
            
            const session = await ort.InferenceSession.create('./model.onnx');

            const input = new ort.Tensor(new Float32Array(ingredients), [1, 380]);
            const feeds = { float_input: input };

            // feed inputs and run
            const results = await session.run(feeds);

            // read from results
            alert('You can enjoy ' + results.label.data[0] + ' cuisine today!')

        } catch (e) {
            console.log(`failed to inference ONNX model`);
            console.error(e);
        }
    }
           
</script>
```
**위의 코드의 의미**
1. 성분 확인란이 선택되었는지 여부에 따라 380개의 가능한 값(0 또는 1)을 설정하여 모델로 전송하여 추론했다.
2. 응용프로그램이 시작될 때 호출되는 init 함수로 확인되었는지 결정하는 방법과 확인란 배열을 만들었다.
    > 확인란을 선택하면 선택한 성분을 반영하도록 성분 배열이 변경된다.
3. 확인란이 선택되었는지 확인하는 testCheckboxes 함수를 만들었다.
4. 버튼을 눌렀을 때, 체크된 체크박스가 있다면, startInference 함수를 사용하여 추론을 시작한다.
5. 추론 루틴에는 다음과 같이 포함되어 있다.
    > 모델의 미동기 로드 설정
    > 
    > 모델에 보낼 텐서 구조 생성
    > 
    > 모델을 교육할 때 만든 float_input 입력을 반영하는 'feeds' 생성(Netron을 사용하여 해당 이름 확인 가능)
    > 
    > 이러한 'feeds'를 모델에 보내고 응답을 기다림


**앱 테스트**

`index.html` 파일이 있는 폴더에서 Visul Studio Code에서 터미널 세션을 연다.

`http-server`가 전체적으로 설치되어 있는지 확인하고 프롬포트에 `http-server`을 입력한다.

로컬 호스트가 열리면 웹 앱을 볼 수 있다.

다양한 재료에 따라 어떤 요리를 추천하는지 확인한다.

위의 간단한 테스트로 **몇 개의 필드가 있는 추천 웹 앱을 구축했다는 것**을 확인했다.

🎈 **만든 웹 앱은 매우 작으므로 ingredient_indexes 데이터에서 성분과 해당 인덱스를 사용하여 계속 구축할 수 있다. 어떤 맛의 조합이 주어진 국민 요리를 만드는 데 효과가 있을지 확인해본다.**




___

## 참고 :
💥 **위의 모든 내용은 [ML-For-Beginners의 자료](https://github.com/codingalzi/ML-For-Beginners/tree/main/4-Classification)를 참고하여 작성했습니다**
