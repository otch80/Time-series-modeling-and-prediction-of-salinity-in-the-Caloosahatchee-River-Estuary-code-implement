# Time-series-modeling-and-prediction-of-salinity-in-the-Caloosahatchee-River-Estuary-code-implement


> Chelsea Qiu,Yongshan Wan, 2013, Time series modeling and prediction of salinity in the Caloosahatchee River Estuary, WATER RESOURCES RESEARCH 에 publish 된 논문의 수리 모델을 학술적 연구를 위해 구현해보는 repository입니다.

- 원문 링크 :  https://doi.org/10.1002/wrcr.20415

<br><br>

## 사용 데이터

- 공공데이터API를 이용해 수집한 데이터를 조합해 사용했습니다

|day|ta|hm|td|dc10Tca|inflow|rainfall|관측소|dist|elevation|temp|salt|
|---:|:--------------------|-----:|-----:|-----:|----------:|---------:|-----------:|:---------|-------:|------------:|-------:|
|2018-02-28 20:00:00|11.2|96|10.5|10|136.6|18.6| 낙동대교|9.4296|3.04|5.54|0.31|1.51985e+09|-0.286259|
|2018-02-28 21:00:00|12.3|89|10.5|10|136.6|19.4|낙동대교|9.4296|3.03|5.56|0.31|1.51985e+09|-0.286259|
|2018-02-28 22:00:00|11.6|87|9.5|10|136.6|21.1|낙동대교|9.4296|3.02|5.59|0.31|1.51986e+09|-0.286259|
|2018-02-28 23:00:00|10.3|89|8.5|10|136.6|22.5|낙동대교|9.4296|3|5.6|0.31|1.51986e+09|-0.286259|
|2018-03-01 01:00:00|6.5|80|3.2|3.23529|235|0.1|낙동대교|9.4296|3.01|5.6|0.31|1.51987e+09|-0.164757|

<br><br>