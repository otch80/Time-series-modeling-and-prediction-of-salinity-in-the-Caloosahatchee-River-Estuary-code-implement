from dateutil.relativedelta import relativedelta 
from sklearn.metrics import mean_squared_error
from datetime import timedelta, datetime
from tqdm import tqdm
from tsms import TSMS
import matplotlib.pyplot as plt
import scipy.stats as ss
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')
plt.rc('font', family='Malgun Gothic')

# file lpad
df = pd.read_csv("./dataset/완성데이터.csv")

# standard scaling
df['zscore'] = ss.zscore(df['inflow'])


### find threshold ###
# inflow threshold를 알고 있으면 찾을 필요가 없다
# point = "을숙도대교P20"
# term = 0.4
# threshold_min = df.loc[df['관측소'] == point,'zscore'].mean() - term
# threshold_max = df.loc[df['관측소'] == point,'zscore'].mean() + term

# print(f"threshold_min : {threshold_min}, threshold_max : {threshold_max}")
# print(f"중간 : {round(df.loc[(df['zscore'] > threshold_min) & (df['zscore'] < threshold_max)].shape[0] / df.shape[0] * 100,2)}% \n작은거 : {round(df.loc[df['zscore'] <= threshold_min].shape[0] / df.shape[0] * 100,2)}% \n큰거 : {round(df.loc[df['zscore'] >= threshold_max].shape[0] / df.shape[0] * 100,2)}%")


## 전체 학습 데이터의 최적의 A, B 찾기
def find_tidal_sqt(): # scaling-factor
    A = 0
    B = 1
    A_grad = 0
    B_grad = 0
    
    for epoch in range(100):
        loss_log = []
        for i in range(1, df.shape[0]):
            pred = np.e ** (-B * df['inflow'][i] + np.log(A))
            loss = pred - df['salt'][i]

            A_grad += 2 * loss
            B_grad += 2 * np.log(df['inflow'][i]) * (loss)
            loss_log.append(loss)

        A -= lr * A_grad / df.shape[0]
        B -= lr * B_grad / df.shape[0]

        print(f"Epoch : {epoch} - A : {A} / B :{B} / loss : {sum(loss_log) / len(loss_log)}")
    return A, B

# 전체 학습 데이터의 최적의 K 찾기
def find_tidal_scf():
    value = 0
    for i in tqdm(range(1,df.shape[0])):
        if (df['elevation'][i] - df['elevation'][i-1] == 0):
            value += (df['salt'][i] - df['salt'][i-1]) * H
        else:
            value += ((df['salt'][i] - df['salt'][i-1]) * H)  / (df['elevation'][i] - df['elevation'][i-1])
    return value / (df['elevation'].shape[0]-1)


# A, B = find_tidal_sqt() # A, B를 모르는 경우에 실행
# 모든 관측지점에서 공통적으로 1:5 비율을 가지고 있음
A = 1
B = 5 

# maximum tidal range
H = df['inflow'].max() - df['inflow'].min()
K = find_tidal_scf()


point_threshold_dict = {'낙동대교' : 0.4,'갑문상류' : 0.4, '을숙도대교P20' : 0.4, '낙동강상류3km' : 0.3, '낙동강상류7.5km' : 0.3, '낙동강상류9km' : 0.3, '낙동강상류10km' : 0.35}
salt_scaling_factor = 0.3 # Greedy result
tidal_scaling_factor = 0.3 # Greedy result 

num_epoch = 5000
lr = 0.001


for point, g in df.groupby('관측소'):
    
    ############################
    # find elevation point - 측정깊이가 고려되지 않은 문제점
    most_freq_ele = g.elevation.value_counts().index[0]
    target_df = g.loc[g['elevation']==most_freq_ele].reset_index(drop=True)
    ############################
    
    
    ############################
    # train-test split
    size = target_df.shape[0]
    rate = 0.8
    limit = int(size * rate)
    
    train_x = target_df.loc[:limit,['unixtime','inflow','rainfall','elevation', 'zscore']].reset_index(drop=True)
    train_y = target_df.loc[:limit,'salt'].reset_index(drop=True)

    test_x = target_df.loc[limit:,['unixtime','inflow','rainfall','elevation', 'zscore']].reset_index(drop=True)
    test_y = target_df.loc[limit:,'salt'].reset_index(drop=True)
    ############################

    ############################
    # Train
    print(f"[Train] training start - {point}")
    
    tsms = TSMS() # datset, poi column, threshold
    
    tsms.A = 1
    tsms.B = 5

    # maximum tidal range
    tsms.H = H
    tsms.K = K
    
    tsms.train(train_x, train_y, lr, point_threshold_dict[point], num_epoch, verbose=False)
    tsms.predict(train_x)
    
    print(f"{point} Test MSE : {mean_squared_error(tsms.pred, train_y)}")
    
    plt.title(point)
    plt.plot(tsms.train_error_log,label="Train")
    plt.plot(tsms.valid_error_log,label="Valid")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()