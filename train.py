import scipy.stats as ss
import pandas as pd
import numpy as np
from model import TSMS
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


df = pd.read_csv("./dataset/train.csv")

A = 5 # const
B = 1 # const
maximum_tidal_range = 1 # const
salt_scaling_factor = 1 # const
tidal_scaling_factor = 1 # const

size = df.shape[0]
rate = 0.5 # train / test split rate
limit = int(size * rate)

train_x = df.loc[:limit, ["day","inflow_avg","elevation_avg","zscore"]]
train_y = df.loc[:limit, ['poi_list']]

test_x = df.loc[limit:, ["day","inflow_avg","elevation_avg","zscore"]].reset_index(drop=True)
test_y = df.loc[limit:, ['poi_list']].reset_index(drop=True)

alpha_beta = []

# Train
poi_cnt = 6
for i in range(poi_cnt):
    print(f"[Train] poi_{i+1}")
    tsms = TSMS()  # datset, poi column, threshold
    tsms.load_dataset(train_x, train_y[f'poi_{i+1}_avg_salt'], 0.7)
    tsms.set_para(A, B, maximum_tidal_range, salt_scaling_factor, tidal_scaling_factor)

    num_epoch = 10
    lr = 0.01
    tsms.find_model_para_SGD(num_epoch, lr)
    alpha_beta.append([tsms.alpha, tsms.beta])
    print("")

# Predict
pred = []
for i, (alpha, beta) in enumerate(alpha_beta):
    prev_salt = train_y.loc[limit - 1, [f'poi_{i + 1}_avg_salt']]
    prev_elev = train_x.loc[limit - 1, 'elevation_avg']
    pred = tsms.predict(alpha, beta, test_x, prev_salt, prev_elev)
    print(f"[Test] poi_{i + 1}")
    print(f">>> MAE : {mean_absolute_error(test_y.loc[:, [f'poi_{i + 1}_avg_salt']], pred)}")
    print(f">>> MSE : {mean_squared_error(test_y.loc[:, [f'poi_{i + 1}_avg_salt']], pred)}")
    print("")