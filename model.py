from tqdm import tqdm
import pandas as pd
import numpy as np
import math
import random
import scipy.stats as ss


# TIme Series Modeling and prediction of Salinity
class TSMS:
    def load_dataset(self, train, label, threshold):
        self.data = train

        self.salt = label  # 염분
        self.salt_occean = 35  # psu

        self.inflow_col = ['inflow']
        avg = self.data.zscore.mean()
        self.upper_thresholds = avg + threshold
        self.lower_thresholds = avg - threshold

        self.Sq = [0] * train.shape[0]  # 각 시점에 해당하는 inflow에 의한 염분 변화량
        self.Sr = [0] * train.shape[0]  # 각 시점에 해당하는 rainfall에 의한 염분 변화량
        self.Sh = [0] * train.shape[0]  # 각 시점에 해당하는 elevation에 의한 염분 변화량

    def set_para(self, A, B, maximum_tidal_range, salt_scaling_factor, tidal_scaling_factor,alpha=[random.random() for _ in range(3)], beta=[random.random() for _ in range(3)]):
        self.alpha = alpha
        self.beta = beta
        self.a = salt_scaling_factor
        self.A = A
        self.B = B
        self.H = maximum_tidal_range
        self.k = tidal_scaling_factor

    def find_model_para_SGD(self, num_epoch, learning_rate, verbose=False):
        for epoch in range(num_epoch):
            gradient = square_error = absolute_error = x = state = 0

            for index in range(self.data.shape[0]):

                if (index == 0):
                    # Inflow
                    self.Sq[index] = 0
                    # Rainfall
                    self.Sr[index] = 0
                    # Elevation
                    self.Sh[index] = 0

                    # Model Formulation (수정 후 - x 값 수정)
                    if (self.data.loc[index, 'zscore'] >= self.upper_thresholds):
                        state = 0
                        y_pred = self.alpha[0] + self.beta[0] * 0

                    elif (self.data.loc[index, 'zscore'] <= self.lower_thresholds):  # x 값 설정이 잘못되었음
                        state = 2
                        y_pred = self.alpha[2] + self.beta[2] * 0 + (self.Sr[index] + self.Sh[index])

                    else:
                        state = 1
                        y_pred = self.alpha[1] + self.beta[1] * 0 + (self.Sq[index] + self.Sr[index] + self.Sh[index])

                    x = 1  # 시간의 변화에 따른 변화값을 인덱스로 계산
                    gradient = (y_pred - 0)
                else:
                    self.Sq[index] = self.fresh_water(index)
                    self.Sr[index] = self.rainfall(index)
                    self.Sh[index] = self.elevation(index)

                    # Model Formulation (수정 후 - x 값 수정)
                    if (self.data.loc[index, 'zscore'] >= self.upper_thresholds):
                        state = 0
                        y_pred = self.alpha[0] + self.beta[0] * self.salt[index - 1]

                    elif (self.data.loc[index, 'zscore'] <= self.lower_thresholds):  # x 값 설정이 잘못되었음
                        state = 2
                        y_pred = self.alpha[2] + self.beta[2] * self.salt[index - 1] + (self.Sr[index] + self.Sh[index])

                    else:
                        state = 1
                        y_pred = self.alpha[1] + self.beta[1] * self.salt[index - 1] + (
                                    self.Sq[index] + self.Sr[index] + self.Sh[index])

                    x = 1  # 시간의 변화에 따른 변화값은 무조건 1
                    gradient = (y_pred - self.salt[index - 1])

                # update
                self.alpha[state] = self.alpha[state] - learning_rate * gradient
                self.beta[state] = self.beta[state] - learning_rate * gradient * x

                # Squared Error
                square_error += (y_pred - self.salt[index]) ** 2
                # Absolute Error
                absolute_error += abs(y_pred - self.salt[index])
            # MSE
            mse = square_error / self.data.shape[0]
            # MAE
            mae = absolute_error / self.data.shape[0]

            if (verbose):
                print(f">>> epoch - {epoch + 1} \t alpha = {round(self.alpha[state], 5)} \t beta = {round(self.beta[state], 5)} \t MAE = {round(mae, 5)} \t MSE = {round(mse, 5)}")

        print(f">>> learning finished")
        print(f">>> alpha = {self.alpha}")
        print(f">>> beta = {self.beta}")
        print(f">>> [Train] MAE = {round(mae, 5)}")
        print(f">>> [Train] MSE = {round(mse, 5)}")

    def predict(self, alpha, beta, test, prev_salt, prev_elev):
        pred = [0] * test.shape[0]
        for index in range(test.shape[0]):
            if (index == 0):
                # Inflow
                delta_Q = 1 # const
                self.Sq[index] = delta_Q * (self.A * (math.e ** (-(self.B) * self.data.loc[index, self.inflow_col])) - prev_salt)
                # Rainfall
                delta_R = 1 # const
                self.Sr[index] = delta_R * (1 - prev_salt / (self.a * self.salt_occean))
                # Elevation
                self.Sh[index] = self.k * ((self.data.loc[index, 'elevation_avg'] - prev_elev) / self.H)

                # Model Formulation
                if (test.loc[index, 'zscore'] >= self.upper_thresholds):
                    pred[index] = (self.alpha[0] + self.beta[0] * prev_salt)
                elif (test.loc[index, 'zscore'] <= self.lower_thresholds):
                    pred[index] = (self.alpha[2] + self.beta[2] * prev_salt + (self.Sr[index] + self.Sh[index]))
                else:
                    pred[index] = (self.alpha[1] + self.beta[1] * prev_salt + (
                                self.Sq[index] + self.Sr[index] + self.Sh[index]))
            else:
                self.Sq[index] = self.fresh_water(index)
                self.Sr[index] = self.rainfall(index)
                self.Sh[index] = self.elevation(index)

                # Model Formulation
                if (test.loc[index, 'zscore'] >= self.upper_thresholds):
                    pred[index] = (self.alpha[0] + self.beta[0] * pred[index - 1])

                elif (test.loc[index, 'zscore'] <= self.lower_thresholds):
                    pred[index] = (self.alpha[2] + self.beta[2] * pred[index - 1] + (self.Sr[index] + self.Sh[index]))

                else:
                    pred[index] = (self.alpha[1] + self.beta[1] * pred[index - 1] + (
                                self.Sq[index] + self.Sr[index] + self.Sh[index]))

        return pred


    # 1. t 시점에서 강우량에 의한 염분 변화량 계산
    def salt_step_impulse_rainfall(self, rainfall):
        # 차후 공식 계산 되면 수정할 계획
        if (rainfall > 1):
            predict_salt = 1
        elif (rainfall < 0):
            predict_salt = 0
        else:
            predict_salt = rainfall

        return predict_salt

    def rainfall(self, time):
        delta_R = self.salt_step_impulse_rainfall(0)  # 현재 강우량 데이터 없음
        change_of_salt_by_rainfall = delta_R * (1 - self.salt[time - 1] / (self.a * self.salt_occean))
        return change_of_salt_by_rainfall

    # 2. t 시점에서 담수에 의한 염분 변화량 계산
    def salt_step_impulse_inflow(self):
        # 차후 공식 계산 되면 수정할 계획
        predict_salt = 1
        return predict_salt

    def salt_on_freshwater(self, time):
        salt_by_freshwater_at_T = self.A * (math.e ** (
                    -(self.B) * self.data.loc[time, self.inflow_col]))  # self.salt[time] 가 아니라  t시점의 inflow 값이 들어가야함
        return salt_by_freshwater_at_T

    def fresh_water(self, time):
        # salt_by_freshwater_at_T : T 시점의 담수량 Q에 해당하는 염분 S
        salt_by_freshwater_at_T = self.salt_on_freshwater(time)  # A,B : 상수, freshwater_at_T : t 시점에서의 담수 유입량
        delta_Q = self.salt_step_impulse_inflow()  # t 시점에서의 담수유입량 이동평균

        # [담수에 의한 염분 변화량] = [t일 이전의 이동평균 담수유입량의 크기와 염도의 상승 또는 하상의 함수에 의한 salinity step impulse] * ((t시점의 담수량 Q가 주어졌을 떄의 염분 S) - (t-1 시점의 염분))
        change_of_salt_by_freshwater = delta_Q * (salt_by_freshwater_at_T - self.salt[time - 1])

        return change_of_salt_by_freshwater

    # 3. t 시점에서 water level에 의한 염분 변화량 계산
    def elevation(self, time):
        change_of_salt_by_elevation = self.k * (
                    (self.data.loc[time, 'elevation_avg'] - self.data.loc[time - 1, 'elevation_avg']) / self.H)
        return change_of_salt_by_elevation
