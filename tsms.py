from tqdm import tqdm
import pandas as pd
import numpy as np
import math
import random
import scipy.stats as ss


# TIme Series Modeling and prediction of Salinity
class TSMS:
#     def __init__(self):

    def train(self, train_x, train_y, learning_rate, threshold, num_epoch, valid=True, verbose = True, window=7):
        self.valid = valid
        if (self.valid):
            split_rate = 0.9
            
            self.train_x = train_x.loc[:int(train_x.shape[0] * split_rate)].reset_index(drop=True)
            self.train_y = train_y.loc[:int(train_y.shape[0] * split_rate)].reset_index(drop=True)
            
            
            self.valid_x = train_x.loc[int(train_x.shape[0] * split_rate):].reset_index(drop=True)
            self.valid_y = train_y.loc[int(train_y.shape[0] * split_rate):].reset_index(drop=True)
            
            # valid inflow (MA 추가)
            self.valid_inflow = self.valid_x['inflow']
            self.valid_inflow_MA = self.valid_x['inflow'].rolling(window=window).mean()
            self.valid_inflow_threshold = self.valid_x['zscore']

            # valid rainfall (MA 추가)
            self.valid_rainfall = self.valid_x['rainfall']
            self.valid_rainfall_MA = self.valid_x['rainfall'].rolling(window=window).mean()

            # valid elevation
            self.valid_elevation = self.valid_x['elevation']
            
            # valid error log
            self.valid_error_log = [0 for _ in range(num_epoch)]
            
        else:
            ### data load ###
            self.train_x = train_x.reset_index(drop=True)
            self.train_y = train_y.reset_index(drop=True)

        # inflow (MA 추가)
        self.train_inflow = self.train_x['inflow']
        self.train_inflow_MA = self.train_x['inflow'].rolling(window=window).mean()
        self.train_inflow_threshold = self.train_x['zscore']

        # rainfall (MA 추가)
        self.train_rainfall = self.train_x['rainfall']
        self.train_rainfall_MA = self.train_x['rainfall'].rolling(window=window).mean()

        # elevation
        self.train_elevation = self.train_x['elevation']
        #################

        self.set_para(learning_rate, threshold)
        self.Gradient_Descent(num_epoch,verbose)

        

    def set_para(self, learning_rate, threshold, salt_scaling_factor = 1):
        self.learning_rate = learning_rate # lr
        self.salt_occean = 35  # 바다물 농도

        # threshold
        avg = self.train_x.zscore.mean()
        self.upper_thresholds = avg + threshold # overflow
        self.lower_thresholds = avg - threshold # underflow

        self.a = salt_scaling_factor # scaling factor

        

    # 학습 데이터가 시간에 따른 특징을 가지기 때문에 랜덤 추출하는 SGD 방식은 적합하지 않다
    # 상황에 따라 적용해아 하는 수식이 3개 있기 때문에 batch를 적용할 수 없다 (일관성 부족)
    # 따라서 전체 데이터에 대한 학습을 수행할 Gradient Descent 방식을 적용한다
    def Gradient_Descent(self, num_epoch, verbose=True):
        
        self.train_pred = [0] * self.train_x.shape[0] # 예측값
        self.error_log = []

        self.train_Sq = [0] * len(train_y)                          # 각 t시점마다의 inflow에 의한 염분 변화량을 미리 계산
        self.train_Sr = [0] * len(train_y)                          # 각 t시점마다의 rainfall에 의한 염분 변화량을 미리 계산
        self.train_Sh = self.calc_elevation(self.train_elevation)   # 각 t시점마다의 elevation에 의한 염분 변화량을 미리 계산

        train_inflow_MA_term = self.train_inflow_MA.max() - self.train_inflow_MA.min()

        state = 0
        square_error = 0
        absolute_error = 0

        self.alpha = [0 for _ in range(3)] # Model-parameter
        self.beta = [1 for _ in range(3)] # Model-parameter

        for epoch in range(num_epoch):
            loss = 0
            self.train_pred[0] = self.train_y[0]
            weight_gradient = [0] * 3
            bias_gradient = [0] * 3

            for time in range(1, self.train_x.shape[0]):
                now_inflow_MA = 0 if np.isnan(self.train_inflow_MA[time]) else self.train_inflow_MA[time]
                before_inflow_MA = 0 if np.isnan(self.train_inflow_MA[time-1]) else self.train_inflow_MA[time-1]
                self.train_Sq[time] = self.calc_inflow(self.train_inflow[time], now_inflow_MA, before_inflow_MA, self.train_y[time-1], train_inflow_MA_term)
                self.train_Sr[time] = self.calc_rainfall(self.train_rainfall_MA[time], self.train_rainfall_MA[time-1], self.train_y[time-1])

                # Model Formulation
                if (self.train_inflow_threshold[time] >= self.upper_thresholds): # overflow
                    state = 0
                    self.train_pred[time] = self.alpha[0] + self.beta[0] * self.train_y[time - 1] + self.train_Sh[time]
                elif (self.train_inflow_threshold[time] <= self.lower_thresholds): # underflow
                    state = 2
                    self.train_pred[time] = self.alpha[2] + self.beta[2] * self.train_y[time - 1] + (self.train_Sr[time] + self.train_Sh[time])
                else: 
                    state = 1
                    self.train_pred[time] = self.alpha[1] + self.beta[1] * self.train_y[time - 1] + (self.train_Sq[time] + self.train_Sr[time] + self.train_Sh[time])
                
                loss += (self.train_pred[time] - self.train_y[time]) ** 2 # 전체 오차 계산용 (MSE)

                weight_gradient[state] += 2 * self.train_y[time-1] * (self.train_pred[time] - self.train_y[time])
                
                # Bias는 alpha + (threshold에 따른 다른 변수)
                if (state == 1):
                    bias_gradient[state] += 2 * ((self.train_pred[time] - self.train_y[time]) - (self.train_Sr[time] + self.train_Sh[time]))
                elif (state == 2):
                    bias_gradient[state] += 2 * ((self.train_pred[time] - self.train_y[time]) - (self.train_Sq[time] + self.train_Sr[time] + self.train_Sh[time]))
                else:
                    bias_gradient[state] += 2 * (self.train_pred[time] - self.train_y[time]) - self.train_Sh[time]

            # Weight update
            self.beta[0] -= self.learning_rate * (weight_gradient[0] / len(self.train_y)) # 평균 기울기
            self.beta[1] -= self.learning_rate * (weight_gradient[1] / len(self.train_y))
            self.beta[2] -= self.learning_rate * (weight_gradient[2] / len(self.train_y))

            # Bias update
            self.alpha[0] -= self.learning_rate * (bias_gradient[0] / len(self.train_y)) # 평균 기울기
            self.alpha[1] -= self.learning_rate * (bias_gradient[1] / len(self.train_y))
            self.alpha[2] -= self.learning_rate * (bias_gradient[2] / len(self.train_y))            

            # MSE
            mse = loss / self.train_y.shape[0]
            self.error_log.append(mse)
            
            # valid
            if (self.valid):
                valid_pred = self.predict(self.valid_x)
                valid_squared_loss = 0
                for i in range(len(valid_pred)):
                    valid_squared_loss += (valid_pred[i] - self.valid_y[i]) ** 2
                self.valid_error_log[epoch] = (valid_squared_loss / len(valid_pred))

            if (verbose):
                print(f"Eppch - {epoch+1} \t: alpha - {self.alpha},\t beta - {self.beta},\t MSE - {mse}")

        print(f">>> learning finished : alpha - {self.alpha},\t beta - {self.beta}, \t MSE - {mse}")

    def predict(self, test_x, window=7):
        
        self.test_inflow = test_x['inflow']
        self.test_inflow_MA = test_x['inflow'].rolling(window=window).mean()

        self.test_rainfall = test_x['rainfall']
        self.test_rainfall_MA = test_x['rainfall'].rolling(window=window).mean()

        self.test_elevation = test_x['elevation']
        self.test_zscore = test_x['zscore']

        self.pred = [0] * len(test_x)
        self.test_Sq = [0] * len(test_x)
        self.test_Sr = [0] * len(test_x)
        self.test_Sh = self.calc_elevation(self.test_elevation)

        test_inflow_MA_term = self.test_inflow_MA.max() - self.test_inflow_MA.min()

        for time in range(1, len(test_x)):
            now_inflow_MA = 0 if np.isnan(self.test_inflow_MA[time]) else self.test_inflow_MA[time]
            before_inflow_MA = 0 if np.isnan(self.test_inflow_MA[time-1]) else self.test_inflow_MA[time-1]
            self.test_Sq[time] = self.calc_inflow(self.test_inflow[time], now_inflow_MA, before_inflow_MA, self.pred[time-1], test_inflow_MA_term)
            self.test_Sr[time] = self.calc_rainfall(self.test_rainfall_MA[time], self.test_rainfall_MA[time-1], self.pred[time-1])
            # Model Formulation (수정 후)
            if (self.test_zscore[time] >= self.upper_thresholds):
                self.pred[time] = (self.alpha[0] + self.beta[0] * self.pred[time - 1])
            elif (self.test_zscore[time] <= self.lower_thresholds):
                self.pred[time] = (self.alpha[2] + self.beta[2] * self.pred[time - 1] + (self.test_Sr[time] + self.test_Sh[time]))
            else:
                self.pred[time] = self.alpha[1] + self.beta[1] * self.pred[time - 1] + (self.test_Sq[time] + self.test_Sr[time] + self.test_Sh[time])
        
        return self.pred


    # 1. t 시점에서 강우량에 의한 염분 변화량 계산
    def calc_rainfall(self, now_rainfall_MA, before_rainfall_MA, before_pred):
        delta_R = self.delta_rainfall_MA(now_rainfall_MA, before_rainfall_MA)
        delta = delta_R * (1 - before_pred / (self.a * self.salt_occean))
        return delta

    def delta_rainfall_MA(self, noe_MA, before_MA):
        now = 0 if np.isnan(noe_MA) else noe_MA
        before = 0 if np.isnan(before_MA) else before_MA
        return np.tanh(now - before) # MA 차이가 1 이상인 경우 예측값이 발산하는 문제 발생, 음수값을 살리기 위한 tanh 사용


    # 2. inflow에 의한 염분 변화량 계산
    def calc_inflow(self, inflow, now_inflow_MA, before_inflow_MA, before_pred, term):
        salt_by_freshwater_at_T = self.A * (math.e ** (-(self.B) * inflow))
        delta_Q = (now_inflow_MA - before_inflow_MA) / term # Step-impluse
        # delta_Q = self.delta_inflow_MA(now_inflow_MA, before_inflow_MA) 
        delta = delta_Q * (salt_by_freshwater_at_T - before_pred)
        return delta

    # def delta_inflow_MA(self, now_MA, before_MA):
    #     now = 0 if np.isnan(now_MA) else now_MA
    #     before = 0 if np.isnan(before_MA) else before_MA

    #     return np.tanh(now - before) # MA 차이가 1 이상인 경우 예측값이 발산하는 문제 발생, 음수값을 살리기 위한 tanh 사용



    # 3. t 시점에서 water level에 의한 염분 변화량 계산
    def calc_elevation(self, elevation):
        delta_df = [0] * len(elevation)
        for time in range(1, len(elevation)):
            delta_df[time] = self.K * ((elevation[time] - elevation[time - 1]) / self.H)
        return delta_df

    def find_tidal_scf(self): # scaling-factor
        value = 0
        for i in range(1,self.train_x.shape[0]):
            if (self.train_elevation[i] - self.train_elevation[i-1] == 0):
                value += (self.train_y[i] - self.train_y[i-1]) * self.H
            else:
                value += ((self.train_y[i] - self.train_y[i-1]) * self.H)  / (self.train_elevation[i] - self.train_elevation[i-1])
        self.K = value / (self.train_elevation.shape[0]-1)