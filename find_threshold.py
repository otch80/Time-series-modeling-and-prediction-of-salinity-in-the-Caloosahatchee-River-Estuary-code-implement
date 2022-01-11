import scipy.stats as ss
import pandas as pd
import numpy as np

df = pd.read_csv("./dataset/train.csv")
df['zscore'] = ss.zscore(df['inflow'])

for term in np.range(0.1,1,0.1):
    threshold_min = df['zscore'].mean() - term
    threshold_max = df['zscore'].mean() + term

    print(f"threshold_min : {round(threshold_min,5)} / threshold_max : {round(threshold_max,5)}")
    print(f"Up : {round(df.loc[df['zscore'] >= threshold_max].shape[0] / df.shape[0] * 100,2)}%")
    print(f"Mid : {round(df.loc[(df['zscore'] > threshold_min) & (df['zscore'] < threshold_max)].shape[0] / df.shape[0] * 100,2)}%")
    print(f"Lo : {round(df.loc[df['zscore'] <= threshold_min].shape[0] / df.shape[0] * 100,2)}%")

