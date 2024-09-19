### 👨‍🏫 Bike_Sharing_Demand
kaggle에서 제공하는 Bike_Sharing_Demand를 이용해 EDA와 model 학습을 통해 고객 만족도를 예측하는 프로젝트

---
### ⏲️ 분석 기간
2024.09.14 - 2024.09.17

---

### 📝 소개
kaggle에서 진행한 Bike Sharing Demand는 2014년부터 2015년까지 1년간 진행된 kaggle의 대회이다. 특히, Bike Sharing Demand는 titanic data와 같이 머신러닝을 입문하는 사람들이 가장 먼저 시작하는 데이터 중 하나이다.

따라서 필자 역시 회귀 게시물 첫 시작을 Bike Sharing Demand으로 시작하려고 한다.

---

### 프로젝트 개요
##### 📌 목표
Kaggle에서 2014년부터 2015년까지 Bike_Sharing_Demand 대회가 진행되었다. Bike_Sharing_Demand 대회는 자전거 대여 수요를 정확히 예측하는 것이 목표이다.
Bike_Sharing_Demand는 머신러닝에 입문하는 사람들이 가장 먼저 접하는 데이터로 현재까지도 더 좋은 점수를 얻기 위한 분석과 코드가 계속 업로드되고 있다. 이에 필자 역시 정확한 자전거 수요 예측을 목표로 상위 10% 안에 진입하는 것을 목표로 이번 프로젝트를 진행했다.

##### 🖥️ 데이터셋 (Data Set)
이 프로젝트에서 사용한 데이터셋은 Kaggle에서 제공하는 다음 파일들로 구성되어 있습니다.
1. train.csv: 훈련 데이터셋, 특징들과 목표 변수를 포함.
2. test.csv: 테스트 데이터셋, 예측을 위해 사용될 데이터.
3. sampleSubmission.csv: 예측 결과를 제출하기 위한 샘플 파일.

---

##### 방법론
1. 문제에 대한 정보 수집
  * 문제 정의
  * 분석 대상에 대한 이해
2. Bike Sharing Demand을 이용한 EDA
  * 공통 코드
    * 오차행렬(Confusion matrix) 및 평가 지표
  * 분석
    * Bike Sharing Demand에 대한 기본적인 정보(구조 파악)
    * 시각화
    * Data cleaning
    * Feature Engineering
3. 모델 학습
  * RandomForest
  * CatBoost
  * XGBoost
  * Top Score
4. 결론
  * EDA 및 주요 인사이트
  * 모델 성능 비교
  * 로그 변환을 통한 성능 향상
  * 한계점
  * 향후 개선 방향

---

## 문제에 대한 정보 수집
### 1. 문제 정의

Bike Sharing Demand는 kaggle에서 진행한 자전거 수요 예측 대회이다. Kaggle은 이 대회를 머신 러닝 커뮤니티가 재미있게 연습할 수 있도록 개최한 것으로 Bike Sharing Demand data는 Capital Bikeshare의 데이터를 사용한 Hadi Fanaee Tork에 의해 제공되었다.

자전거 공유 시스템은 도시 곳곳에 설치된 키오스크 네트워크를 통해 회원 가입, 대여, 반납 과정이 자동화된 자전거 대여 수단으로 사람들은 한 장소에서 자전거를 빌리고 필요에 따라 다른 장소에 반납할 수 있습니다. 자전고 공유 시스템으로 생성되는 데이터는 여행 시간, 출발지, 도착지, 경과 시간 등이 명확하게 기록되기 때문에 자전거 공유 시스템은 도시 내 이동성을 연구할 수 있는 센서 네트워크 역할을 한다. 따라서 kaggle에서 진행한 대회는 과거의 사용 패턴과 날씨 데이터를 결합하여, 워싱턴 D.C.의 Capital Bikeshare 프로그램에서 자전거 대여 수요를 예측하는 과제를 수행하는 것이다.

수요 예측이 도움이 되는 이유는 다음고 같다. 몇몇 이유는 카카오 바이크, 에브리바이크 등 공공 자전거를 이용해 본 사람들은 공감할 수 있다고 생각한다. 수요 예측은 특히 도시 계획, 자원 배분, 운영 효율성 등을 개선하는 데 도움이 된다. 따라서 자전거 수요를 예측하는 것이 중요한 이유는 다음과 같다. 운영 효율성 향상, 고객 만족도 향상, 도시 계획 및 교통 관리, 비즈니스 및 정책 결정, 이벤트 및 날씨 대비 등 여러 상황에서 도움을 줄 수 있다.

#### 1. 운영 효율성 향상

* 자전거 배치 최적화

자전거 대여 시스템은 특정 시간대와 위치에서 자전거가 많이 대여되거나 반납되는 패턴이 존재한다. 수요 예측을 통해 자전거가 필요할 시간과 장소를 미리 예측할 수 있으므로, 자전거를 효율적으로 배치할 수 있다. 예를 들어, 아침 출근 시간에는 주거 지역에서 자전거 대여가 많고, 저녁 시간에는 직장 근처에서 반납이 많이 일어날 수 있다.

#### 2. 고객 만족도 향상

  * 사용자 편의성 증대
  
  수요를 예측하여 사용자들이 자전거를 대여하고 반납하는 데 불편함이 없도록 할 수 있다. 자전거가 부족하거나 반납할 공간이 없는 상황을 줄임으로써 서비스의 품질과 고객 만족도를 높일 수 있다.
  
  * 서비스 품질 향상
  
  대여소에 자전거가 부족하거나 반납할 공간이 없으면 사용자의 불편이 커진다. 수요 예측을 통해 이러한 문제를 미리 방지할 수 있다.

#### 3. 도시 계획 및 교통 관리

  * 교통 혼잡 완화
  
  자전거 대여 시스템은 대중교통의 보완 수단으로, 특히 교통 혼잡이 심한 지역에서 대중교통과 연계하여 효과적으로 사용할 수 있다. 수요 예측을 통해 교통 혼잡을 줄이고 자전거 이용을 활성화함으로써 도시의 교통 흐름을 개선할 수 있다.
  
  * 지속 가능한 도시 개발
  
  자전거 대여 시스템은 친환경적이고 지속 가능한 교통수단이다. 수요 예측을 통해 자전거를 더 효율적으로 활용하면, 도시의 환경 개선과 지속 가능한 개발에도 기여할 수 있다.

#### 4. 비즈니스 및 정책 결정

  * 비즈니스 전략 수립
  
  자전거 대여 시스템의 수익성은 수요에 따라 좌우된다. 수요 예측을 통해 대여 요금을 조정하거나 특정 시간대에 프로모션을 제공하는 등의 전략을 수립할 수 있다.
  
  * 정부 정책 지원
  
  자전거 대여 데이터를 통해 자전거 이용 활성화 정책을 평가하고, 이를 기반으로 정책을 조정할 수 있다. 예를 들어, 특정 지역에서 자전거 사용이 늘어나면 추가 인프라 구축이나 자전거 사용을 장려하는 정책을 시행할 수 있다.

#### 5. 이벤트 및 날씨 대비

  * 이벤트 대응
  
  특별한 이벤트나 행사가 열릴 때 자전거 수요가 급증할 수 있다. 수요 예측을 통해 이러한 이벤트를 대비하고, 자전거 대여 시스템의 준비를 할 수 있다.
  
  * 날씨 변화 대응
  
  날씨는 자전거 수요에 큰 영향을 미친다. 비가 오거나 더운 날씨에는 수요가 줄어들고, 쾌적한 날씨에는 수요가 늘어날 수 있다. 이러한 패턴을 예측하여 날씨 변화에 맞춰 자전거 배치를 조정할 수 있다.

  이런 이유로 자전거 수요 예측은 운영 효율성을 높이고, 고객 만족도를 개선하는 데 중요한 역할을 합니다. 데이터 기반의 수요 예측을 통해 자전거 대여 시스템의 성능을 극대화하고, 비즈니스와 정책 결정에 필요한 중요한 인사이트를 제공할 수 있다.

### 2. 분석 대상에 대한 이해
kaggle에서는 train.csv, test.csv, sampleSubmission.csv 총 3개의 파일을 제공해 준다. train.csv로 학습을 하고 test.csv의 count 즉, 수요를 예측하는 것이다. 이후 예측 값을 sampleSubmission.csv에 결합한 후 제출하는 것이다.
![image](https://github.com/user-attachments/assets/edcc13a4-a027-4df0-932d-058684ae048e)

데이터는 2011년 1월 ~ 2012년 12월까지 1시간 간격 동안 자전거 대여 횟수 기록한 것으로 datetime, season, holiday, workingday, weather, temp, atemp, humidity, windspeed, casual, registered, count 각각 다음과 같다.
  * datetime: hourly date + timestamp
  * season: 1 - 봄, 2 - 여름, 3- 가을, 4 - 겨울
  * holyday: 1 - 토, 일요일의 주말을 제외한 국경일 등의 휴일, 0 - 휴일이 아닌 날
  * workingday: 1 - 토, 일요일의 주말 및 휴일이 아닌 주중, 0 - 주말 및 휴일
  * weather: 1 = 맑음, 약간 구름 낀 흐림 / 2 = 안개, 안개 + 흐림 / 3 = 가벼운 눈, 가벼운 비 + 천둥 / 4 = 심한 눈/비, 천둥/번개
  * temp: 온도(섭씨)
  * atemp: 체감 온도(섭씨)
  * humidity: 상대 습도
  * windspeed: 풍속
  * casual: 사전에 등록되지 않은 사용자가 대여한 횟수
  * registered: 사전에 등록된 사용자가 대여한 횟수
  * count: 대여 횟수

---

## Bike Sharing Demand을 이용한 EDA
### 1. 공통 코드
Bike Sharing Demand의 평가 요소는 RMSLE이다. RMSLE는  Root Mean Squared Logarithmic Error로 RMSE에 로그를 적용한 것, 결정값이 클 수록 오류값도 커지기 때문에 일부 큰 오류값들로 인해 전체 오류값이 커지는 것을 막아준다. RMSE는 MSE 값은 로그를 적용한 것, 실제 오류 평균보다 더 커지는 특성이 있으므로 MSE에 루트를 씌운 것이다. MSE는 평균 제곱 오차로 실제 타깃값과 예측 타깃값 차의 제곱의 평균이다. 쉽게 말해 RMSLE는 예측값과 실제값의 로그 차이를 측정하는 지표로 예측값과 실제값이 큰 범위에 걸쳐 있거나, 상대적 오차에 더 관심이 있을 때 사용된다. 특히, 값의 크기가 매우 클 때 과도한 오류를 방지하고, 작은 값에 더 민감하게 반응하도록 하는 데 유리하다.
```
from sklearn.metrics import make_scorer

def rmsle(y, pred):
    log_y = np.log1p(y)
    log_pred = np.log1p(pred)
    squared_error = (log_y - log_pred) ** 2
    rmsle = np.sqrt(np.mean(squared_error))
    return rmsle

def rmse(y, pred):
    return np.sqrt(mean_squared_error(y,pred))

def evaluate_regr(y, pred):
    rmsle_val = rmsle(y, pred)
    rmse_val = rmse(y, pred)
    mae_val = mean_absolute_error(y, pred)
    print('RMSLE: {0:.3f}, RMSE: {1:.3F}, MAE: {2:.3F}'.format(rmsle_val, rmse_val, mae_val))
    return rmsle(y, pred)

scores = make_scorer(evaluate_regr)
```
각각 RMSLE, RMSE, MAE를 구하고 출력하고 RMSLE를 반환하는 함수이다. RMSLE는 위에서 이미 설명했으며, 나머지는 다음에 설명하겠다. 특히, make_scorer는 사용자 정의 평가 함수를 스코어링 함수로 변환하는 역할을 하는 것이다. 따라서 필자가 정의한 evaluate_regr 함수를 scikit-learn에서 사용할 수 있는 스코어링 함수로 변환하는 역할을 한다.
```
warnings.filterwarnings('ignore')
# 노트북 안에 그래프를 그리기 위해
%matplotlib inline
plt.style.use('ggplot') # 그래프에서 격자로 숫자 범위가 눈에 잘 띄도록 ggplot 스타일을 사용
mpl.rcParams['axes.unicode_minus'] = False # 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처

RANDOM_STATE = 110
pd.set_option('display.max_columns', None)

train_df = pd.read_csv("../../data/bike/train.csv", parse_dates=['datetime'])
test_df = pd.read_csv("../../data/bike/test.csv", parse_dates=['datetime'])
santander_submission_df = pd.read_csv("../../data/bike/sampleSubmission.csv")
```
데이터가 위치한 폴더는 각자 다르기 때문에 적절히 수정해서 사용하면 될 것이다.

### 2. 분석
#### 1. Bike Sharing Demand에 대한 기본적인 정보(구조 파악)
```
train_df.shape
```
(10886, 12)

총 10886개의 행이 있고 12개의 feature가 있다. 12개의 feature는 datetime, season, holiday, workingday, weather, temp, atemp, humidity, windspeed, casual, registered, count이며 설명은 위에서 했다.
```
train_df.info()
```
![image](https://github.com/user-attachments/assets/22430559-fab6-4393-b312-f12ce6fcb683)

NaN 값은 없으며 datetime은 type이 datetime으로 날짜를 기록한 feature임을 확인할 수 있다.
```
train_df.describe()
```
![image](https://github.com/user-attachments/assets/719398e4-9e13-4fbf-82e8-ec25d22cc967)

describe() 메서드를 통해 요약 통계를 확인할 수 있다. 요약 통계를 보면 humidity와 windspeed가 최소 값이 0이다. humidity 즉, 상대 습도가 0이라는 것은 공기 중에 수증가 존재하지 않는다는 것이며, 풍속의 경우 바람이 약해 풍향을 판단하기 어려울 시에는 '00'으로 표기하기도 한다. 따라서 humidity와 windspeed는 확인이 필요하다고 생각한다.
```
test_df.info()
```
![image](https://github.com/user-attachments/assets/4c866746-92c3-4ae8-a0b4-55cef18a541a)

test 데이터의 경우 9개의 feature를 가지고 있다. 즉, train 데이터에는 있는 casual, registered가 없다. count는 예측해야 하는 타겟 값이기 때문에 제외한다면 casual, registered는 train 데이터에만 있는 값이기 때문에 삭제할 필요가 있다. 또한, test 데이터 역시 NaN값이 없는 것을 확인할 수 있다.

#### 2. 시각화
시각화에 앞서 datatime을 년, 월, 일, 시간으로 구분하는 작업을 진행하려고 한다.
```
train_df['year'] = train_df.datetime.apply(lambda x: x.year)
train_df['month'] = train_df.datetime.apply(lambda x: x.month)
train_df['day'] = train_df.datetime.apply(lambda x: x.day)
train_df['hour'] = train_df.datetime.apply(lambda x: x.hour)
```
```
fig, axs = plt.subplots(figsize=(16, 8), ncols=4, nrows=2)
features = ['year', 'month', 'season', 'weather', 'day', 'hour', 'holiday', 'workingday']

for i, feature in enumerate(features):
    row = int(i/4)
    col = i%4
    sns.barplot(x=feature, y='count', data=train_df, ax=axs[row][col])
```
![image](https://github.com/user-attachments/assets/8917566d-b851-48af-902d-e8c34adb5c2d)
![image](https://github.com/user-attachments/assets/d4332099-cd95-42e9-86e9-83989277621c)

위와 같이 시각화를 할 수 있다.
  * year - 2011년보다 2012년에 더 많은 사람들이 자전거를 대여했다.
  * month - 6월이 가장 대여량이 많으며, 1월이 가장 적다.
  * season - 1: 봄에 대여량이 가장 적은 것을 확인할 수 있다. 자전거 대여 서비스가 시작한지 별로 안 된 시점이라 적을 수도 있지만 일반 적인 예상인 날씨가 좋으면 대여량이 많다는 것은 아닌 것을  확인할 수 있다.
  * weather - 날씨 역시 3: 가벼운 비/눈 보다 심한 비/눈이 올 때 대여량이 많다. 오히려 2: 안개 일 때와 비슷하다. 
  * day - 1일부터 19일까지 있으며, 20일부터는 test 데이터에만 있다. 따라서 day는 사용하면 안 되는 데이터이다.
  * hour - 시간대 별로 출근 시간대, 퇴근 시간대에 가장 많은 사람들이 이용하는 것을 확인할 수 있다. 하지만 주말과 비교해 볼 필요가 있다.
  * holiday - 1: 토, 일요일의 주말을 제외한 국경일 등의 휴일, 0: 휴일이 아닌 날로 비슷한 것을 확인할 수 있다. 즉, 출퇴근 용이 아니라도 많이 이용하는 것을 확인할 수 있다.
  * workingday - 1: 토, 일요일의 주말 및 휴일이 아닌 주중, 0: 주말 및 휴일로 holiday와 비슷하게 출퇴근 용이 아니라도 많이 이용하는 것을 확인할 수 있다.
  * minute, second는 모두 0으로 생략했다.

다음으로 시간별 대여량을 여러 지표를 기준으로 확인해보려고 한다.
```
fig,(ax1,ax2,ax3,ax4)= plt.subplots(nrows=4)
fig.set_size_inches(18,25)

sns.pointplot(data=train_df, x="hour", y="count", ax=ax1)

sns.pointplot(data=train_df, x="hour", y="count", hue="workingday", ax=ax2)

sns.pointplot(data=train_df, x="hour", y="count", hue="weather", ax=ax3)

sns.pointplot(data=train_df, x="hour", y="count", hue="season", ax=ax4)
```
![image](https://github.com/user-attachments/assets/5fd92ace-5465-4627-bed5-8f92949ee982)

위에서 확인했 듯 출퇴근 시간에 대여량이 급증한 것을 확인할 수 있다.

![image](https://github.com/user-attachments/assets/5f77b122-cf04-4ad8-8f64-bbd6a0766006)

파란 선이 주중, 빨간 선이 주말이다. 주중에는 출퇴근 시간에 많이 대여를 하며, 주말에는 점심 이후 저녁 전에 많이 이용하는 것을 확인할 수 있다.

![image](https://github.com/user-attachments/assets/aa8a4867-b75a-461a-811b-5fb63b8e691c)

weather의 경우 1 = 맑음, 약간 구름 낀 흐림 / 2 = 안개, 안개 + 흐림 / 3 = 가벼운 눈, 가벼운 비 + 천둥 / 4 = 심한 눈/비, 천둥/번개으로 1~3은 출퇴근 시간에 가장 많이 이용하며, 3은 평소보다 적은 것을 확인할 수 있다. 위에서 weather을 시각화했을 때 날씨가 많이 나쁠 때가 세 번째로 대여량이 높았는데 그 이유는 위 그래프를 통해 추측하면 4의 지표가 퇴근 시간에만 있다. 따라서 퇴근 후 갑작스런 기상 악화에 자전거를 대여해 빨리 집에 간 것으로 볼 수 있다.

![image](https://github.com/user-attachments/assets/e3f87236-581f-41d4-99dd-1318a0674fe5)

1 = 봄을 제외하면 출근 시간대에는 거의 같은 대여량을 보여주고 퇴근 시간대에 날씨가 나쁠 수록 대여량이 적은 것을 확인할 수 있다. 지금까지 확인해 본 결과 날씨는 자전거 대여량에 많은 영향을 주는 것을 확인할 수 있다.

날씨가 자전거 대여량에 많은 영향을 주는 것을 확인했지만 봄에 대여량이 적은 것을 확인했다. 따라서 이 부분에 대해서도 한 번 확인을 해보려고 한다.

```
def concatenate_year_month(datetime):
    return "{0}-{1}".format(datetime.year, datetime.month)

train_df["year_month"] = train_df["datetime"].apply(concatenate_year_month)

print(train_df.shape)
train_df[["datetime", "year_month"]].head()
```
```
fig, ax1 = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(18, 4)
sns.barplot(data=train_df, x="year_month", y="count", ax=ax1)
```
![image](https://github.com/user-attachments/assets/65e8c58e-0081-462f-bb6b-0181c4f7f480)

Bike Sharing Demand의 데이터는 워싱턴 D.C의 데이터이다. 따라서 3~5월이 봄이다. 위 그래프를 보면 2011년에는 3, 4월에 대여량이 2012년에 비해 많이 적다. 따라서 자전거 대여 서비스가 시작한지 별로 안 된 시점이라 적을 수 있다는 가설이 맞을 수도 있다. 또한, 날씨가 좋을 때 자전거 대여량이 많다는 것도 다시 한 번 확인할 수 있다.

다음으로 상관관계를 확인해 보겠다.
```
corr_matrix = train_df[["temp", "atemp", "humidity", "windspeed", 'season', 'weather', 'year', 'month', 'day', 'hour', "count"]].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.show()
```
![image](https://github.com/user-attachments/assets/5ab485a6-e95d-4eba-999b-b48a82fbb12f)

temp, atemp는 0.98로 상관관계가 높지만 온도와 체감온도로 피처로 사용하기에 적합하지 않을 수 있다. 따라서 확인 필요하다. month와 season 역시 0.97로 상관관계가 높으며 역시 확인이 필요하다.

0, 1, 2, 3과 같이 category type이 아닌 온도, 습도, 풍속을 시각화를 해보면 다음과 같다.
```
fig,(ax1,ax2,ax3,ax4) = plt.subplots(ncols=4)
fig.set_size_inches(20, 5)
sns.regplot(x="temp", y="count", data=train_df,ax=ax1)
sns.regplot(x="atemp", y="count", data=train_df,ax=ax2)
sns.regplot(x="windspeed", y="count", data=train_df,ax=ax3)
sns.regplot(x="humidity", y="count", data=train_df,ax=ax4)
```
![image](https://github.com/user-attachments/assets/41db1b25-1989-407f-b222-0b651108e984)

온도, 체감 온도는 0이 없으며 습도는 0이 많지 않다. 반면 풍속은 0이 굉장히 많은 것으로 알 수 있다. 풍속에 대해 시각화를 더 자세히 하면 다음과 같다.

```
fig, axes = plt.subplots(nrows=2)
fig.set_size_inches(18, 10)

plt.sca(axes[0])
plt.xticks(rotation=30, ha='right')
axes[0].set(ylabel='Count')
sns.countplot(data=train_df, x='windspeed', ax=axes[0])

plt.sca(axes[1])
plt.xticks(rotation=30, ha='right')
axes[0].set(ylabel='Count')
sns.countplot(data=test_df, x='windspeed', ax=axes[1])
```
![image](https://github.com/user-attachments/assets/f0d1d27a-fbeb-48e0-8458-e119451d4553)
![image](https://github.com/user-attachments/assets/1be3dd3d-edd0-4164-a0b6-495933a9ac09)

바람이 약해 풍향을 판단하기 어려울 시에는 '00'으로 표기하기도 하지만 지금 데이터의 경우엔 0에 굉장히 많은 값이 모여있는 것을 확인할 수 있다. 아마도 관측되지 않은 수치에 대해 0으로 기록된 것이 아닐까 추측해 볼 수 있다. 따라서 풍속의 0 값에 특정 값을 넣어줘야 하며, 평균을 구해 일괄적으로 넣어줄 수도 있지만, 예측의 정확도를 높이는 데 도움이 되지 않을 수도 있다. 따라서 풍속이 0인 것과 아닌 것의 세트를 나누어 예측을 통해 풍속을 구하는 방법을 이용하려고 한다.

#### Data cleaning
##### 1. 필요 없는 feature 삭제
위에서 확인했던 test 데이터에는 없고 train 데이터에만 있는 casual, registered 그리고 시각화를 위해 만들었던 year_month를 drop하고자 한다.
```
drop_feature = ['casual', 'registered']
train_df.drop(drop_feature, axis=1, inplace=True)

train_df.drop(['year_month'], axis=1, inplace=True)

train_df.head()
```
##### 2. 이상치
이상치를 확인해 보려고 한다. 이상치는 데이터 분포에서 벗어난 비정상적인 값으로, 분석과 모델링에 부정적인 영향을 줄 수 있다.
```
fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12, 10)
sns.boxplot(data=train_df,y="count",orient="v",ax=axes[0][0])
sns.boxplot(data=train_df,y="count",x="season",orient="v",ax=axes[0][1])
sns.boxplot(data=train_df,y="count",x="hour",orient="v",ax=axes[1][0])
sns.boxplot(data=train_df,y="count",x="workingday",orient="v",ax=axes[1][1])

axes[0][0].set(ylabel='Count',title="대여량")
axes[0][1].set(xlabel='Season', ylabel='Count',title="계절별 대여량")
axes[1][0].set(xlabel='Hour Of The Day', ylabel='Count',title="시간별 대여량")
axes[1][1].set(xlabel='Working Day', ylabel='Count',title="근무일 여부에 따른 대여량")
```
박스 플롯(box plot)을 사용해, 여러 피처(feature)와 자전거 대여량(count) 간의 관계를 한눈에 확인하기 위한 작업이다. 박스 플롯은 데이터의 분포와 이상치(outliers)를 확인하는 데 유용한 도구이다.
![image](https://github.com/user-attachments/assets/29e68337-38d2-439c-adfa-79146c515fd2)
![image](https://github.com/user-attachments/assets/e1ff4b2e-6031-4a09-8b5e-5bafb328dec9)

전체 대여량, 계절별, 시간별, 근무일별 각각 이상치가 있다. 하지만 이상치라 해서 무작정 제거하는 것은 좋지 않다. 특정 상황에서는 이상치가 중요한 의미를 가질 수 있기 때문에, 이상치가 중요한 정보일 경우 삭제가 아닌 수정하거나 그대로 이용하는 방법이 있다.
```
trainWithoutOutliers = train_df[np.abs(train_df["count"] - train_df["count"].mean()) <= (3*train_df["count"].std())]

print(train_df.shape)
print(trainWithoutOutliers.shape)
```
표준 편차를 이용하여 count 값이 평균에서 크게 벗어난 값(이상치)을 제거하는 코드로 각 count 값에서 평균을 뺀 후 절대값을 취한다. 이렇게 하면 각 데이터가 평균으로부터 얼마나 떨어져 있는지를 알 수 있다. 음수는 제거되고 모두 양수로 변환된다. 또한, 통계적으로, 정규분포에서는 약 99.7%의 데이터가 평균으로부터 3 표준 편차 이내에 위치. 즉, 이 범위를 벗어나는 값들은 이상치일 가능성이 크다. 따라서 평균에서 3 표준편차 이상 벗어난 값을 drop하는 것이다. 즉, 3 표준편차 이내에 있는 데이터만 남기는 것이다.
```
figure, axes = plt.subplots(ncols=2, nrows=2)
figure.set_size_inches(12, 10)

sns.distplot(train_df["count"], ax=axes[0][0])
stats.probplot(train_df["count"], dist='norm', fit=True, plot=axes[0][1])

sns.distplot(np.log(trainWithoutOutliers["count"]), ax=axes[1][0])
stats.probplot(np.log1p(trainWithoutOutliers["count"]), dist='norm', fit=True, plot=axes[1][1])
```
이상치를 제거하기 전과 후를 비교하면 다음과 같다.
![image](https://github.com/user-attachments/assets/ebfb1240-c3d0-47d1-9042-f1524d00914d)

대부분의 기계학습은 종속변수가 normal 이어야 하기에 정규분포를 갖는 것이 바람직하다. 대안으로 outlier data를 제거하고 "count"변수에 로그를 씌워 변경해 봐도 정규분포를 따르지는 않지만 이전 그래프보다는 좀 더 자세히 표현하고 있다.

#### Feature Engineering
##### 1. 새로운 피처 생성
앞에서 시각화를 위해 train 데이터에 대해서 datetime을 year, month, day, hour로 나눴었다. 따라서 test 데이터에도 같은 작업을 하려고 한다.
```
test_df['year'] = test_df.datetime.apply(lambda x: x.year)
test_df['month'] = test_df.datetime.apply(lambda x: x.month)
test_df['day'] = test_df.datetime.apply(lambda x: x.day)
test_df['hour'] = test_df.datetime.apply(lambda x: x.hour)

train_df["dayofweek"] = train_df["datetime"].dt.dayofweek
test_df["dayofweek"] = test_df["datetime"].dt.dayofweek
```
추가적으로 dayofweek을 이용해 요일을 숫자로 나타낸 것이다. 즉, datetime 열에 있는 날짜 정보를 사용하여 요일을 새로운 열 dayofweek에 추가하는 코드이다. 다음과 같이 추가된다.
  * 0: 월요일 (Monday)
  * 1: 화요일 (Tuesday)
  * 2: 수요일 (Wednesday)
  * 3: 목요일 (Thursday)
  * 4: 금요일 (Friday)
  * 5: 토요일 (Saturday)
  * 6: 일요일 (Sunday)
```
fig,(ax1)= plt.subplots(nrows=1)
fig.set_size_inches(12,5)

sns.pointplot(data=train_df, x="hour", y="count", hue="dayofweek", ax=ax1)
```
![image](https://github.com/user-attachments/assets/6662fa75-eee7-47e4-a93d-16bbaa261852)

시각화를 하면 다음과 같이 나온다. 위에서 확인했듯 주중에는 출퇴근 시간대에 대여량이 많은 것을 확인할 수 있고, 주말에는 점심 이후부터 저녁까지 대여량이 증가하는 것을 확인할 수 있다.
##### 2. 결측값 처리
다음으로 앞서 확인했던 windspeed에 대해서도 처리를 하려고 한다. 풍속이 0인 것과 아닌 것의 세트를 나누어 예측을 통해 풍속을 구하는 방법을 이용하려고 한다.
```
trainWind0 = train_df.loc[train_df['windspeed'] == 0]
trainWind1 = train_df.loc[train_df['windspeed'] != 0]
```
```
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostRegressor, CatBoostClassifier

def predict_windspeed(data):
    dataWind0 = data.loc[data['windspeed'] == 0]
    dataWind1 = data.loc[data['windspeed'] != 0]

    # 풍속을 예측할 feature
    wcol = ['season', 'weather', 'temp', 'atemp', 'humidity', 'year', 'month', 'hour', 'dayofweek']
    # wcol = ['season', 'weather', 'temp', 'atemp', 'humidity', 'year', 'month']

    # 풍속이 0이 아닌 데이터들의 타입을 스트링으로 바꿔준다.
    dataWind1['windspeed'] = dataWind1['windspeed'].astype('str')

    # rf = RandomForestClassifier()
    # rf = RandomForestRegressor()
    # rf = XGBRegressor()
    rf = CatBoostClassifier(verbose=False)
    # rf = CatBoostRegressor(verbose=False)
    
    rf.fit(dataWind1[wcol], dataWind1["windspeed"])
    windvalues = rf.predict(dataWind0[wcol])

    # windvalues의 차원을 1차원으로 변환
    windvalues = windvalues.ravel()

    predictWind0 = dataWind0
    predictWind1 = dataWind1

    # 값이 0으로 기록된 풍속에 대해 예측한 값을 넣어준다.
    predictWind0['windspeed'] = windvalues
    # 0이 아닌 풍속이 있는 데이터프레임에 예측한 값이 있는 데이터프레임을 합쳐준다.
    data = pd.concat([predictWind1, predictWind0])

    data['windspeed'] = data['windspeed'].astype('float')

    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)

    return data
```

RandomForestClassifier, RandomForestRegressor, XGBRegressor, CatBoostClassifier, CatBoostRegressor 모델을 사용했을 때 가장 점수가 좋게 나온 것은 이상치를 제거하지 않고 RandomForestClassifier와 CatBoostClassifier이다. 뿐만 아니라 각각 예측에 학습한 데이터는 ['season', 'weather', 'temp', 'atemp', 'humidity', 'year', 'month'], ['season', 'weather', 'temp', 'atemp', 'humidity', 'year', 'month', 'hour', 'dayofweek'] 일 때 가장 높게 나왔다. 그 중 가장 점수가 높게 나온 CatBoostClassifer를 이용하려고 한다.

##### 3. Feature Selection
  * 신호와 잡음을 구분해야 한다.
  * 피처가 많다고 무조건 좋은 성능을 내지 않는다.
  * 피처를 하나씩 추가하고 변경해 가면서 성능이 좋지 않은 피처는 제거하도록 한다.
```
train_df['year'] = train_df['year'].replace({2011: 0, 2012: 1})
test_df['year'] = test_df['year'].replace({2011: 0, 2012: 1})
```
위와 같이 year을 2011이면 0, 2012이면 1로 바꾸면, season, holiday, workingday, weather, year, month, dayofweek은 숫자의 크기가 의미가 있는 것이 아니기 때문에 categorical feature이다.  따라서 전부 categorical 데이터로 변환할 것이다.
```
categorical_feature = ['season', 'holiday', 'workingday', 'weather', 'dayofweek', 'month', 'year', 'hour']

for i in categorical_feature:
    train_df[i] = train_df[i].astype('category')
    test_df[i] = test_df[i].astype('category')
```
```
feature_names = ["season", "holiday", "workingday", "weather", "temp", "atemp", "humidity", "windspeed", #'month', # month 추가
                 "year", "hour", "dayofweek"]
```
다음과 같이 모델 훈련에 사용될 feature을 선택했다. month의 경우엔 사용했을 때 오히려 성능이 안 좋아져서 사용하지 않았다.

최종적으로 count까지 분리하면 학습 데이터는 11개의 feature만 남는다.
```
X = train_df[feature_names]
y = train_df["count"]
X.head()
```
![image](https://github.com/user-attachments/assets/733440b3-4211-4722-ad39-ec1fd8b83c97)

---

## 모델 학습

