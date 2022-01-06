import pandas as pd
import numpy as np

# TIme Series Modeling and prediction of Salinity
class TSMS:
    def __init__(self, df, window=7):
        self.data = df.values[:,
                    :3]  # data[:,0] : 담수 유입량(freshwater inflow), data[:,1] : 강수량(rainfall), data[:,2] : 조수 높이 (elevation)
        self.salt = df.values[:, -1]  # 염분
        self.salt_occean = 35  # psu
        self.MA = df.rolling(window=window).mean()

    def set_para(self, alpha, beta, A, B, upper_thresholds, lower_thresholds, maximum_tidal_range, salt_scaling_factor,
                 tidal_scaling_factor):
        # set-parameter (grid)
        self.a = salt_scaling_factor
        self.alpha = alpha
        self.beta = beta
        self.A = A
        self.B = B
        self.H = maximum_tidal_range
        self.k = tidal_scaling_factor

        self.upper_thresholds = upper_thresholds
        self.lower_thresholds = lower_thresholds

    def predict(self, time):
        self.time = time  # const
        self.Sq = self.fresh_water()
        self.Sr = self.rainfall()
        self.Sh = self.elevation()

        predict = self.salt_At_Time()

        return predict

    def salt_At_Time(self):
        # St = A + B * S(t-1) + S(t)(Q,R,H)
        # flow regime(flow pattern) 은 하구의 위치에 따라 달라지는 상한 및 하한 임계값인 QUT와 QLT에 따라 T일의 총 유입으로 정의된다.
        if (self.data[self.time, 0] >= self.upper_thresholds):
            pred = self.alpha[0] + self.beta[0] * self.salt[self.time - 1]  # 강이 범람한 경우 염분 계산은 큰 의미가 없다 / 입실론은 형식상 표기
        elif (self.data[self.time, 0] <= self.lower_thresholds):
            pred = self.alpha[2] + self.beta[2] * self.salt[self.time - 1] + (self.Sr + self.Sh)
        else:
            pred = self.alpha[1] + self.beta[1] * self.salt[self.time - 1] + (self.Sq + self.Sr + self.Sh)

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

    def rainfall(self):
        delta_R = self.salt_step_impulse_rainfall(0)  # 현재 강우량 데이터 없음
        change_of_salt_by_rainfall = delta_R * [1 - self.salt[self.time - 1] / (self.a * self.salt_occean)]
        return change_of_salt_by_rainfall

    # 2. t 시점에서 담수에 의한 염분 변화량 계산
    def salt_step_impulse_inflow(self):
        # 차후 공식 계산 되면 수정할 계획
        predict_salt = 1
        return predict_salt

    def salt_on_freshwater(self):
        salt_by_freshwater_at_T = self.A * (math.e ** (-(self.B) * self.salt[self.time]))
        return salt_by_freshwater_at_T

    def fresh_water(self):
        # salt_by_freshwater_at_T : T 시점의 담수량 Q에 해당하는 염분 S
        salt_by_freshwater_at_T = self.salt_on_freshwater()  # A,B : 상수, freshwater_at_T : t 시점에서의 담수 유입량
        delta_Q = self.salt_step_impulse_inflow()  # t 시점에서의 담수유입량 이동평균

        # [담수에 의한 염분 변화량] = [t일 이전의 이동평균 담수유입량의 크기와 염도의 상승 또는 하상의 함수에 의한 salinity step impulse] * ((t시점의 담수량 Q가 주어졌을 떄의 염분 S) - (t-1 시점의 염분))
        change_of_salt_by_freshwater = delta_Q * (salt_by_freshwater_at_T - self.salt[self.time - 1])

        return change_of_salt_by_freshwater

    # 3. t 시점에서 water level에 의한 염분 변화량 계산
    def elevation(self):
        change_of_salt_by_elevation = self.k * ((self.data[self.time, 2] - self.data[self.time - 1, 2]) / self.H)
        return change_of_salt_by_elevation