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

```python
pip install imblearn
```

```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from imblearn.over_sampling import SMOTE
```
```python
df  = pd.read_csv('cuisines.csv')
```
```python
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
df.info()
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2448 entries, 0 to 2447
Columns: 385 entries, Unnamed: 0 to zucchini
dtypes: int64(384), object(1)
memory usage: 7.2+ MB
```

### **연습 - 요리에 대해 배우기**

```python
df.cuisine.value_counts().plot.barh() # 데이터 분포가 고르지 않음 
```

![image](https://user-images.githubusercontent.com/62239143/167248334-e68e45b0-79c1-4aa1-901d-5ad8b49525ab.png)

```python
# 요리당 사용할 수 있는 데이터 크기
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
### **성분 발견하기**

지금부터 데이터를 깊게 파서 요리별 일반적인 재료가 무엇인지 배우기위해 요리 사이의 혼동을 일으킬 중복 데이터를 정리해보자.

- Python에서 성분 데이터프레임을 생성하기 위해서 create_ingredient() 함수를 만든다. 이 함수는 도움이 안되는 열을 삭하고 개수별로 재료를 정렬한다.

```python
def create_ingredient_df(df):
    ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
    ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
    ingredient_df = ingredient_df.sort_values(by='value', ascending=False, inplace=False)
    return ingredient_df
```

함수를 사용하여 요리별 가장 인기있는 10개 재료의 아이디어를 얻을 수 있다.

```python
# 타이
thai_ingredient_df = create_ingredient_df(thai_df)
thai_ingredient_df.head(10).plot.barh()
```

![image](https://user-images.githubusercontent.com/62239143/167248357-418b4789-6f54-436a-99c5-954a9ec293c2.png)

```python
# 일본
japanese_ingredient_df = create_ingredient_df(japanese_df)
japanese_ingredient_df.head(10).plot.barh()
```

![image](https://user-images.githubusercontent.com/62239143/167248364-446c08a5-1bd9-4461-a496-d68d424205db.png)

```python
# 중국
chinese_ingredient_df = create_ingredient_df(chinese_df)
chinese_ingredient_df.head(10).plot.barh()
```

![image](https://user-images.githubusercontent.com/62239143/167248370-205807ee-eec8-4a85-ab42-5fc4757feadd.png)


```python
# 인도
indian_ingredient_df = create_ingredient_df(indian_df)
indian_ingredient_df.head(10).plot.barh()
```

![image](https://user-images.githubusercontent.com/62239143/167248378-886e4508-aa65-42f4-8396-316ea62bdb08.png)


```python
# 한국
korean_ingredient_df = create_ingredient_df(korean_df)
korean_ingredient_df.head(10).plot.barh()
```

![image](https://user-images.githubusercontent.com/62239143/167248383-2029db17-23e0-43f5-9de4-9207098d55f3.png)



```python
# 구별되는 요리 사이에 혼란을 주는 가장 공통적인 재료를 삭제 
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

### **데이터셋 균형 맞추기**

데이터를 정리 했으므로 SMOTE ("Synthetic Minority Over-sampling Technique")를 사용하여 균형을 맞춘다.

```python
#fit_resample(): 보간으로 새로운 샘플을 생성함 
oversample = SMOTE()
transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
```
데이터의 균형을 맞추면 분류할 때 더 나은 결과를 얻을 수 있다. 데이터 균형을 맞추면 왜곡된 데이터를 가져와 이러한 불균형을 제거하는 데 도움이 된다.

```python
print(f'new label count: {transformed_label_df.value_counts()}')
print(f'old label count: {df.cuisine.value_counts()}')
```

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
# 레이블과 특성을 포함한 균형 잡힌 데이터를 파일로 내보낼 수 있는 새 데이터 프레임에 저장
transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
transformed_df
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

## **chap2**

### **연습 - 국가 요리 예측하기**

```python
# 파일 불러오기 
import pandas as pd
cuisines_df = pd.read_csv("cleaned_cuisines.csv")

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

# 훈련을 위해 두개의 데이터프레임으로 X와 Y좌표를 나눔
cuisines_label_df = cuisines_df['cuisine']
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
# 'Unnamed: 0'의 열과 'cuisine' 열을 'drop()'을 호출하여 삭제
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

데이터가 정리되고 학습할 준비가 되었으므로 작업에 사용할 알고리즘을 결정해야함

Scikit-learn은 지도 학습에서 분류를 그룹화하고 해당 범주에서 분류하는 다양한 방법을 찾을 수 있음
- 선형 모델
- 서포트 벡터 머신
- 확률적 경사하강법
- 가장 가까운 이웃
- 가우스 프로세스
- 의사결정나무
- 앙상블 방법(투표 분류기)
- 다중 클래스 및 다중 출력 알고리즘(다중 클래스 및 다중 레이블 분류, 다중 클래스 다중 출력 분류)
- 신경망 (이 강의의 범위를 벗어나므로 여기선 사용x)

분류기를 선택하기위해선 여러 가지를 실행하고 좋은 결과를 찾는 것이 테스트하는 방법이다.기본적으로 Scikit-learn에 로지스틱 회귀를 수행하도록 요청할 때 지정해야 하는 `multi_class` 와 `solver` 중요한 두 개의 파라미터가 있다. 
- `multi_class` 값은 특정 동작을 적용
- `solver`의 값은 사용할 알고리즘

#### **더 나은 접근법**
성급히 추측하기보다 더 나은 방법은 다운로드 가능한 ML Cheat sheet의 아이디어를 따르는 것이다.
![image](https://user-images.githubusercontent.com/62239143/167248760-1d3552f0-af22-4b65-a992-77c09aa63d20.png)
> 다중 클래스 분류 옵션을 자세히 설명하는 Microsoft의 알고리즘 치트 시트 섹션

기본적으로 Scikit-learn에 로지스틱 회귀를 수행하도록 요청할 때 지정해야 하는 `multi_class` 와 `solver` 중요한 두 개의 파라미터가 있다. 
- `multi_class` 값은 특정 동작을 적용
- `solver`의 값은 사용할 알고리즘



### **연습 - 데이터 나누기**
**데이터 나누기**

```python
# train_test_split()을 호출하여 데이터를 훈련, 테스트 그룹으로 나눈다,
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

### **연습 - 로지스틱 회귀 적용하기**

1. multi_class를 `ovr`로 설정하고 solver도 `liblinear`로 설정하여 로지스틱 회귀를 만든다.

```python
lr = LogisticRegression(multi_class='ovr',solver='liblinear')
model = lr.fit(X_train, np.ravel(y_train))

accuracy = model.score(X_test, y_test)
print ("Accuracy is {}".format(accuracy))
```
*종종 기본값으로 설정된 `lbfgs`와 같은 다른 solver를 사용해도 된다*

```
Accuracy is 0.8065054211843202
```

정확도 80%이상으로 좋음

```python
print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
print(f'cuisine: {y_test.iloc[50]}')
```
```
ingredients: Index(['chicken', 'cilantro'], dtype='object')
cuisine: thai
```
다른 행 번호 사용해서도 결과 확인 가능

```python
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

**인도 요리**가 가장 좋은 추측이다.

```python
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

## **chap3**

### **분류 지도 **

이전에, Microsoft 치트 시트를 사용해서 데이터를 분류할 때 다양한 옵션을 배웠다. Scikit-learn은 추정기(classifiers)를 좁히는 데 더 도움을 받을 수 있는 유사하지만 보다 세분화된 치트 시트를 비슷하게 제공한다.
![image](https://user-images.githubusercontent.com/62239143/167249053-138126c6-6dfc-45a4-b0ea-bb596424ef8c.png)

위의 지도는 데이터를 명확하게 파악하면 의사 결정에 이르는 경로로 나아갈 수 있으므로 유용하다.
- 50개 이상의 샘플을 가지고 있다
- 범주를 예측하고 싶다
- 레이블이 지정된 데이터가 있다
- 100K(10만)개 미만의 샘플을 가지고 있다
- 선형 SVC를 고를 수 있다
- 수치 데이터를 가지고 있어서 작동이 되지 않을 때
   -  KNeighbors Classifier을 사용해볼 수 있다.
   - 그래도 작동 되지 않는다면, SVC 및 앙상블 분류기를 사용해 볼 수 있다.

### **연습 - 데이터 나누기**

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
import numpy as np
```

```python
# 데이터를 훈련과 테스트로 나눔
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

### **선형 SVC 분류기**

`SVC`(Support-Vector Clustering)는 ML 기술의 `Support-Vector Macine` 제품군의 하위 항목이다. 이 방법에서 `kernel`을 선택하여 레이블을 군집화하는 밥법을 결정할 수 있다. `C` 인자는 파라미터의 영향을 규제하는 '정규화'를 나타낸다. `kernel`은 여러 개 중 하나일 수 있다. 여기서는 선형 SVC를 활용하도록 `linear`로 설정한다. 확률은 기본적으로 `false`로 설정된다. 여기서는 확률 추정치를 수집하기 위해 `true`로 설정한다. 확률값을 얻기 위해 무작위 상태를 '0'으로 설정한다.

#### **연습 - 선형 SVC 적용하기**
```python
C = 10

# 다양한 분류기 생성
classifiers = {'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)}
```

```python
# 선형 SVC로 모델을 훈련하고 확인 
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
결과 좋음

### **K-Neighbors 분류기**
`K-Neighbors`는 지도 학습과 비지도 학습 모두에 사용할 수 있는 ML 방법의 "neighbors" 계열의 일부이다. 이 방법에서는, 일반화된 레이블이 그 데이터에 대해 예측할 수 있도록 미리 정의된 점이 생성되고 데이터는 이 포인트 주변에 수집된다.

#### **연습 - K-Neighbors 분류기 적용**

이전의 분류기도 좋았고 잘 작동했지만, 더 나은 정확도를 얻을 수 있으므로 `K-Neighbors 분류기`를 시도해보자.

```python
# 결과가 73.1%로 조금 더 나쁨
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

### **Support 벡터 분류기**

`Support-Vector 분류기`는 분류 및 회귀 작업에 사용되는 ML 메서드의 `Support-Vector Machine` 제품군의 일부이다. `SVM`은 두 범주간의 거리를 최대화하기 위해 "공간 내 지점에 훈련 예제를 매핑"한다. 후속 데이터는 해당 범주를 예측할 수 있도록 이 공간에 매핑된다.

#### **연습 - Support 벡터 분류기 적용하기**
```python
# Support 벡터 분류기 적용하여 훈련하고 확인-> 약 83.2%로 좋은 결과
classifiers = {'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0), 'KNN classifier': KNeighborsClassifier(C), 'SVC': SVC()}
```
```
Accuracy (train) for SVC: 83.2% 
              precision    recall  f1-score   support

     chinese       0.79      0.74      0.76       242
      indian       0.88      0.90      0.89       234
    japanese       0.87      0.81      0.84       254
      korean       0.91      0.82      0.86       242
        thai       0.74      0.90      0.81       227

    accuracy                           0.83      1199
   macro avg       0.84      0.83      0.83      1199
weighted avg       0.84      0.83      0.83      1199
```

### **앙상블 분류기**

앙상블 분류기는 `Random Forest`와 `AdaBoost`를 사용하여 성능을 확인해볼 것이다.

```python
# 랜덤 포레스트 결과 매우 좋음
classifiers = {'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0), 'KNN classifier': KNeighborsClassifier(C), 'SVC': SVC(), 'RFST': RandomForestClassifier(n_estimators=100), 'ADA': AdaBoostClassifier(n_estimators=100)}
```

```
Accuracy (train) for RFST: 84.5% 
              precision    recall  f1-score   support

     chinese       0.80      0.77      0.78       242
      indian       0.89      0.92      0.90       234
    japanese       0.86      0.84      0.85       254
      korean       0.88      0.83      0.85       242
        thai       0.80      0.87      0.83       227

    accuracy                           0.85      1199
   macro avg       0.85      0.85      0.85      1199
weighted avg       0.85      0.85      0.85      1199

Accuracy (train) for ADA: 72.4% 
              precision    recall  f1-score   support

     chinese       0.64      0.49      0.56       246
      indian       0.91      0.83      0.87       259
    japanese       0.68      0.69      0.69       226
      korean       0.73      0.79      0.76       236
        thai       0.67      0.83      0.74       232

    accuracy                           0.72      1199
   macro avg       0.73      0.73      0.72      1199
weighted avg       0.73      0.72      0.72      1199
```

머신 러닝의 이 방법은 모델의 품질을 향상시키기 위해 "여러 기본 추정기의 예측을 결합"한다. 이 예시에서는 `Random Trees`와 `AdaBoost`를 사용해봤다.
- 평균화 방법인 **`Random Forest`** 는 과적합을 피하기 위해 무작위성이 주입된 'decision trees'의 'forest'를 만든다. `n_estimators` 파라미터는 트리 수로 설정한다.
- **`AdaBoost`** 는 분류기를 데이터 집합에 맞춘 다음 해당 분류기의 복사본을 동일한 데이터 집합에 맞춘다. 잘못 분류된 항목의 가중치에 초점을 맞추고 다음 분류자가 수정하도록 적합도를 조정한다.

## **chap4**

### **요리 추천 Web App 만들기**
이 단원에서는 이전 단원에서 배운 몇 가지 기술과 이 시리즈 전체에서 사용된 맛있는 요리 데이터 세트를 사용하여 분류 모델을 구축한다. 또한 Onnx의 웹 런타임을 활용하여 저장된 모델을 사용하는 작은 웹 앱을 만든다. 머신 러닝의 가장 유용한 실제 용도 중 하나는 추천 시스템을 구축하는 것이다.


**배우게 될 방법**
- 모델을 빌드하고 Onnx 모델로 저장하는 방법
- Netron을 사용하여 모델을 검사하는 방법
- 추론을 위해 웹 앱에서 모델을 사용하는 방법

### **모델 구축하기**
응용 ML 시스템을 구축하는 것은 비즈니스 시스템에 이러한 기술을 활용하는 데 중요한 부분이다. Onnx를 사용하여 웹 애플리케이션 내에서 모델을 사용할 수 있다.(필요한 경우 오프라인 컨텍스트에서 모델을 사용할 수 있음).

이전 단원 에서는 UFO 목격에 대한 회귀 모델을 만들고 "pickled"하고 Flask 앱에서 사용했다. 이 구조는 알고 있으면 매우 유용하지만 full-stack Python 앱이며 요구 사항에 JavaScript 응용 프로그램 사용이 포함할 수 있다.

이 단원에서는 추론을 위한 기본 JavaScript 기반 시스템을 구축할 수 있다. 그러나 먼저 모델을 훈련시키고 Onnx에서 사용할 수 있도록 변환해야 한다.

### **연습 - 훈련 분류 모델**

우리가 사용했던 정제된 요리 데이터를 사용하여 분류 모델을 훈련시킨다.

```python
!pip install skl2onnx
import pandas as pd 
```
Scikit-learn 모델을 Onnx 형식으로 변환 하려면 ' skl2onnx '가 필요

```python
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
# 처음 두개의 필요없는 열을 삭제하고, 나머지 데이터를 'X'로 저장
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

### **훈련 루틴 개시하기**
```python
# 정확도가 좋은 'SVC 라이브러리'를 사용하여 훈련
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
# SVC분류기 모델을 빌드
model = SVC(kernel='linear', C=10, probability=True,random_state=0)
model.fit(X_train,y_train.values.ravel())
```

```
SVC(C=10, kernel='linear', probability=True, random_state=0)
```

```python
# predict()를 호출하면서 모델을 테스트
y_pred = model.predict(X_test)

# 모델의 품질 확인
print(classification_report(y_test,y_pred))
```

```
              precision    recall  f1-score   support

     chinese       0.72      0.69      0.70       257
      indian       0.91      0.87      0.89       243
    japanese       0.79      0.77      0.78       239
      korean       0.83      0.79      0.81       236
        thai       0.72      0.84      0.78       224

    accuracy                           0.79      1199
   macro avg       0.79      0.79      0.79      1199
weighted avg       0.79      0.79      0.79      1199
```
정확도는 좋음

### **Onnx로 모델 변환**

적절한 Tensor 수로 변환해야 한다. 데이터 세트에는 380개의 성분이 나열되어 있으므로 `FloatTensorType`에 이 숫자를 표기해야 한다.

```python
# 380개의 수를 tensor를 사용하여 변환 
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, 380]))]
options = {id(model): {'nocl': True, 'zipmap': False}}
```

```python
# onx를 생성하고 'model.onnx' 파일로 저장
onx = convert_sklearn(model, initial_types=initial_type, options=options)
with open("./model.onnx", "wb") as f:
    f.write(onx.SerializeToString())
```

> 변환 스크립트에서 옵션을 전달할 수 있다. 이 경우, 'nocl'을 True로, 'zipmap'을 False로 전달했다. 이 모델은 분류 모델이므로 사전 목록을 생성하는 ZipMap을 제거하는 옵션이 있다.(필수 아님) 'nocl'은 모델에 포함된 클래스 정보를 나타낸다. 'nocl'을 True로 설정하면, 모델 크기를 줄인다.


### **모델 보기**

`Onnx 모델`은 Visual Studio code에서 잘 보이지 않지만, 많은 연구자들이 모델을 시각화하기 위해 사용하는 매우 우수한 무료 소프트웨어이다. `Netron`을 다운로드하고 `model.onnx`파일을 연다. 380개의 입력 및 분류기가 나열된 간단한 모델을 시각화한 것을 볼 수 있다.

![image](https://user-images.githubusercontent.com/62239143/167251056-d1e21c92-e809-45b5-98ef-e0257fd42199.png)

`Netron`은 모델을 보는 데 유용한 도구이다.

이제 웹 앱에서 이 깔끔한 모델을 사용할 준비가 되었다. 냉장고를 볼 때 유용하게 사용할 수 있는 앱을 만들고, 모델에 따라 주어진 요리를 요리하는 데 사용할 수 있는  남은 재료의 조합을 알아볼 것이다.

### **추천 웹 앱 구축**

웹 앱에서 직접 모델을 사용할 수 있다. 이 구조는 필요한 경우 로컬 및 오프라인에서도 실행 가능하다. `model.onnx` 파일을 저장한 동일한 폴더에서 `index.html`를 생성하여 시작해볼 것이다.

1. index.html 파일에 마크업 추가한다.
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

2. body 태그 내에서 작업하여 일부 구성 요소를 반영하는 확인 목록을 보여주기 위해 약간의 마크업 추가한다.
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
각 확인란에는 값이 지정된다. 이는 데이터 세트에 따라 성분이 발견된 인덱스를 반영한다.
  - ex) 알파벳 목록에서 'Apple'은 다섯 번째 열을 차지하므로 0에서 숫자를 시작할 때 값은 '4'
성분 스프레드 시트를 참조하여 특정 성분의 색인을 찾을 수 있다. `index.html`에서 작업을 계속하면서 최종 종료 `</div>` 뒤에 모델이 호출되는 스크립트 블록 추가


3. Onnx Runtime 가져온다.
```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
```
> Onnx Runtime은 최적화와 사용할 API를 포함하여 광범위한 하드웨어 플랫폼에서 Onnx 모델을 실행할 수 있도록 하는 데 사용

4. Runtime이 준비되면 다음과 같이 호출할 수 있다.
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

**이 코드에서는 몇 가지 일이 발생한다**
1. 성분 확인란이 선택되었는지 여부에 따라 380개의 가능한 값(0 또는 1)의 배열을 생성하여 추론을 위해 모델로 전송했다.
2. 성분의 배열과 애플리케이션을 시작하며 호출했던 `init` 함수에서 확인되었는 지 확인할 방법을 만들었다. 확인란을 선택하면, 선택한 성분을 반영하도록 `ingredients` 배열이 변경된다.
3. 모든 확인란을 선택했는지 확인하는 `testCheckboxes` 함수를 만들었다.
4. 버튼을 누르면 이 함수를 사용하고, 만약 선택된 확인란이 있다면, `startInference` 함수를 사용하여 추론을 시작한다.
5. 추론 루틴에는 다음이 포함된다.
    i. 모델의 비동기 로드 설정
   ii. 모델에 보낼 Tensor 구조 생성
  iii. 모델을 훈련할 때 만들었던 입력 `float_input`을 반영하는 'feeds' 생성 (Netron을 사용하여 해당 이름 확인 가능)
   iv. 이'feeds'를 모델에 보내고 응답 기다림

### **애플리케이션 테스트**

index.html 파일이 있는 폴더에서 Visul Studio Code로 터미널 세션을 연다. `http-server`가 전체적으로 설치되어 있는지 확인하고, 프롬포트에 http-server를 입력한다. 로컬 호스트가 열리면 웹 앱을 볼 수 있다. 다양한 재료에 따라 어떤 요리가 추천되는지 확인한다.

🚀 이 웹 앱은 매우 작으므로 ingredient_indexes 데이터에서 성분과 해당 인덱스를 사용하여 구축해라. 주어진 국가 요리를 만드려면 어떤 풍미 조합으로 작업해야 될까?





---

출처 : [Getting started with classification](https://github.com/codingalzi/ML-For-Beginners/tree/main/4-Classification)
