from abc import ABCMeta, abstractmethod
import sys
import numpy as np
import scipy.io
from scipy.special import erf, erfc
import os

from filters import load_hankel_filter, load_fft_filter
# from . import filters.load_fft_filter


class BaseEm(metaclass=ABCMeta):
    """
    人工信号源による周波数・時間領域電磁探査法の、一次元順解析をデジタルフィルタ法を用いて行う。
    ※ 詳細については、齋藤(2019)を参照。
    Implementation has been based on Ward and Hohmann (1987) and Zhanhui Li,et.al.(2018)
    Developed MATLAB and Python version by I.Saito since 2019
    """
    def __init__(self, x, y, z, tx, ty, tz, res, thickness, hankel_filter):
        """
        コンストラクタ。インスタンスが生成される時に実行されるメソッド。
        サブクラス Fdem or Tdem 内で用いられる変数を設定する。

        引数
        ----------
        x :  array-like
           受信点のx座標 [m] ; 例：[10]
        y :  array-like
           受信点のy座標 [m] ; 例：[20]
        z :  array-like
           受信点のz座標 [m] ; 例：[30]
        tx :  array-like
           送信点のx座標 [m] ; 例：[40]　
        ty :  array-like
           送信点のy座標 [m] ; 例：[50]
        tz :  array-like
           送信点のz座標 [m] ; 例：[60]
        　　res : array-like
           層の比抵抗 [ohm-m] ; 例：np.array([100, 100])
        　　thickness : array-like
           層厚 [m]  ; 例：np.array([60])
        hankel_filter ： str
           Hankel変換用のデジタルフィルターの名前
        "Werthmuller201" , "mizunaga90", "anderson801", "kong241", "key201"のいずれか。

        """

        # 必ず渡させる引数
        self.x = x[0]
        self.y = y[0]
        self.z = z[0]
        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.res = res  # resistivity
        self.thickness = thickness
        self.num_layer = len(res) + 1

        # デフォルト引数に設定しておく引数(変えたいユーザもいる)
        self.hankel_filter = hankel_filter
        self.moment = 1

        # 指定しておく引数（変えたいユーザはほぼいない）
        self.mu0 = 4 * np.pi * 1e-7
        self.mu = np.ones(self.num_layer) * self.mu0
        #self.mu[1:len(self.res)-1] = [,,,]　透磁率:muを層毎に設定したい場合はここに入力
        self.epsrn = 8.85418782 * 1e-12

        # 渡された引数を基に変数を設定
        self.sigma = 1 / res
        self.r = np.sqrt((self.x - self.tx[0]) ** 2 + (self.y - self.ty[0]) ** 2)
        self.h = np.zeros((1, self.num_layer - 1)) # h:層境界のz座標
        for ii in range(2, self.num_layer):  # (2) 2 <= self.rlayer <= self.num_layer-1
            self.h[0, ii - 1] = self.h[0, ii - 2] + self.thickness[ii - 2]


    def em1d_base(self):
        """

        (1) Hankel変換用フィルタ係数の読み込み
        (2) 計算できない条件が入力された場合の例外処理
        (3) 送受信点が含まれる層の特定
        　
        返り値　※ selfに格納される
        ----------
        wt0 :  array-like
            積分区間における第0次Bessel関数
        wt1 :  array-like
            積分区間における第1次Bessel関数
        y_base :  array-like
            Hankel変換における積分区間
        filter_length :  int
            フィルタ長
        ｒLayer :  int
            受信点の含まれる層
        ｔLayer :  int
            送信点の含まれる層
        """

        self.y_base, self.wt0, self.wt1 = load_hankel_filter(self.hankel_filter)
        self.filter_length = len(self.y_base)

        # 計算できない送受信点の位置関係が入力された場合の処理
        if self.r == 0:
            self.r = 1e-8
        self.cos_phai = (self.x - self.tx[0]) / self.r
        self.sin_phai = (self.y - self.ty[0]) / self.r
        if self.hankel_filter == "anderson801":
            self.delta_z = 1e-4
            if (self.tz[0] in self.h):
                    self.tz[0] = self.tz[0] - self.delta_z
            if (self.tz[0] == self.z):
                self.tz[0] = self.tz[0] - self.delta_z
        else:
            self.delta_z = 1e-8
            if (self.tz[0] in self.h):
                    self.tz[0] = self.tz[0] - self.delta_z
            if (self.tz[0] == self.z):
                self.tz[0] = self.tz[0] - self.delta_z

        # 送受信点が含まれる層の特定
        def calc_transceiver_existing_layer(self):
            #self.h = np.zeros((1, self.num_layer - 1))
            # identify the index of receiver existing layer
            if self.z <= 0:  # (1) self.rlayer = 1
                self.rlayer = 1
            else:
                self.rlayer = []
            if self.num_layer == 2 and self.z > 0:  # (1') self.rlayer = 2
                self.rlayer = 2
            elif (self.num_layer >= 3) and (self.z > 0):
                for ii in range(2, self.num_layer):  # (2') 2 <= self.rlayer <= self.num_layer-1
                    if self.z > self.h[0, ii - 2] and self.z <= self.h[0, ii - 1]:
                        self.rlayer = ii
            if self.rlayer == []:  # (3) self.rlayer = self.num_layer
                self.rlayer = ii + 1

            # identify the index of trasmitter existing layer
            if self.tz[0] <= 0:  # (1) self.tlayer = 1
                self.tlayer = 1
            else:
                self.tlayer = []
            if self.num_layer == 2 and self.tz[0] > 0:  # (2) self.tlayer = 2
                self.tlayer = 2
            elif (self.num_layer >= 3) and (self.tz[0] > 0):
                for ii in range(2, self.num_layer):  # (2') 2 <= self.tlayer <= self.num_layer-1
                    if self.tz[0] > self.h[0, ii - 2] and self.tz[0] <= self.h[0, ii - 1]:
                        self.tlayer = ii
            if self.tlayer == []:  # (3) self.tlayer = self.num_layer
                self.tlayer = ii + 1
        calc_transceiver_existing_layer(self)

    def compute_reflection_coefficients(self, Y, Z,tanhuh):
        """
        各層の,上側、下側における境界係数の計算
        ※ 齋藤(2019) 2.5.1 (式 2.36 ～ 式2.37 参照)

        引数
        ----------
        Y :  array-like
            アドミタンス
        Z :  array-like
            インピーダンス
        tanhuh :  array-like
            層情報を含む変数

        返り値　※ selfに格納される
        -------
        R_te_up : array-like
            下側層境界におけるteモードの境界係数
        R_tm_up : array-like
            下側層境界におけるtmモードの境界係数
        R_te_down : array-like
            上側層境界におけるteモードの境界係数
        R_tm_down :　array-like
            上側層境界におけるtmモードの境界係数
        """

        r_te = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        r_tm = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        R_te = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        R_tm = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)

        # te, tmモードおける、下側境界の境界係数の計算　
        Ytilda = np.zeros((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        Ztilda = np.zeros((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        Ytilda[self.num_layer - 1] = Y[self.num_layer - 1]  # (1) Ytilda{self.num_layer}
        Ztilda[self.num_layer - 1] = Z[self.num_layer - 1]

        r_te[self.num_layer - 1] = 0
        r_tm[self.num_layer - 1] = 0

        for ii in range(self.num_layer - 1, self.tlayer, -1):  # (2) Ytilda{self.num_layer-1,self.num_layer-2,\,self.tlayer}
            numerator_Y = Ytilda[ii] + Y[ii - 1] * tanhuh[ii - 1]
            denominator_Y = Y[ii - 1] + Ytilda[ii] * tanhuh[ii - 1]
            Ytilda[ii - 1] = Y[ii - 1] * numerator_Y / denominator_Y

            numerator_Z = Ztilda[ii] + Z[ii - 1] * tanhuh[ii - 1]
            denominator_Z = Z[ii - 1] + Ztilda[ii] * tanhuh[ii - 1]
            Ztilda[ii - 1] = Z[ii - 1] * numerator_Z / denominator_Z

            r_te[ii - 1] = (Y[ii - 1] - Ytilda[ii]) / (Y[ii - 1] + Ytilda[ii])
            r_tm[ii - 1] = (Z[ii - 1] - Ztilda[ii]) / (Z[ii - 1] + Ztilda[ii])
        if self.tlayer != self.num_layer:
            r_te[self.tlayer - 1] = (Y[self.tlayer - 1] - Ytilda[self.tlayer]) / (Y[self.tlayer - 1] + Ytilda[self.tlayer])
            r_tm[self.tlayer - 1] = (Z[self.tlayer - 1] - Ztilda[self.tlayer]) / (Z[self.tlayer - 1] + Ztilda[self.tlayer])

        # te,tmモードおける、上側境界の境界係数の計算
        Yhat = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        Zhat = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        Yhat[0] = Y[0]  # (1)Y{0}
        Zhat[0] = Z[0]

        R_te[0] = 0
        R_tm[0] = 0

        for ii in range(2, self.tlayer):
            numerator_Y = Yhat[ii - 2] + Y[ii - 1] * tanhuh[ii - 1]
            denominator_Y = Y[ii - 1] + Yhat[ii - 2] * tanhuh[ii - 1]
            Yhat[ii - 1] = Y[ii - 1] * numerator_Y / denominator_Y  # (2)Yhat{2,3,\,self.tlayer-2,self.tlayer-1}

            numerator_Z = Zhat[ii - 2] + Z[ii - 1] * tanhuh[ii - 1]
            denominator_Z = Z[ii - 1] + Zhat[ii - 2] * tanhuh[ii - 1]
            Zhat[ii - 1] = Z[ii - 1] * numerator_Z / denominator_Z

            R_te[ii - 1] = (Y[ii - 1] - Yhat[ii - 2]) / (Y[ii - 1] + Yhat[ii - 2])
            R_tm[ii - 1] = (Z[ii - 1] - Zhat[ii - 2]) / (Z[ii - 1] + Zhat[ii - 2])
        if self.tlayer != 1 :
            R_te[self.tlayer - 1] = (Y[self.tlayer - 1] - Yhat[self.tlayer - 2]) / (Y[self.tlayer - 1] + Yhat[self.tlayer - 2])
            R_tm[self.tlayer - 1] = (Z[self.tlayer - 1] - Zhat[self.tlayer - 2]) / (Z[self.tlayer - 1] + Zhat[self.tlayer - 2])

        return r_te, r_tm, R_te, R_tm

    def compute_damping_coefficients(self, r_te, r_tm, R_te, R_tm, u, Y, Z):
        """
        各層の、上側、下側における減衰係数の計算
        ※ 齋藤(2019) 2.5 参照

        引数
        ----------
        r_te : array-like
            下側層境界におけるteモードの境界係数
        r_tm : array-like
            下側層境界におけるtmモードの境界係数
        R_te : array-like
            上側層境界におけるteモードの境界係数
        R_tm :　array-like
            上側層境界におけるtmモードの境界係数
        u : array-like
            各層の情報を含む変数
        Y : array-like
            各層のアドミタンス
        Z : array-like
            各層のインピーダンス

        返り値　※ selfに格納される
        -------
        U_te : array-like
            上側層境界におけるteモードの境界係数
        U_tm : array-like
            上側層境界におけるtmモードの境界係数
        D_te : array-like
            下側層境界におけるteモードの境界係数
        D_tm :　array-like
            下側層境界におけるtmモードの境界係数
        """

        U_te = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        U_tm = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        D_te = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        D_tm = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)

        # In the layer containing the source (tlayer)
        if self.tlayer == 1:
            U_te[self.tlayer - 1] = 0
            U_tm[self.tlayer - 1] = 0
            D_te[self.tlayer - 1] = self.kernel_te_down_sign * r_te[self.tlayer - 1] * np.exp(
                -u[self.tlayer - 1] * (self.h[0, self.tlayer - 1] - self.tz[0]))
            D_tm[self.tlayer - 1] = self.kernel_tm_down_sign * r_tm[self.tlayer - 1] * np.exp(
                -u[self.tlayer - 1] * (self.h[0, self.tlayer - 1] - self.tz[0]))
        elif self.tlayer == self.num_layer:
            U_te[self.tlayer - 1] = self.kernel_te_up_sign * R_te[self.tlayer - 1] * np.exp(
                u[self.tlayer - 1] * (self.h[0, self.tlayer - 2] - self.tz[0]))
            U_tm[self.tlayer - 1] = self.kernel_tm_up_sign * R_tm[self.tlayer - 1] * np.exp(
                u[self.tlayer - 1] * (self.h[0, self.tlayer - 2] - self.tz[0]))
            D_te[self.tlayer - 1] = 0
            D_tm[self.tlayer - 1] = 0
        else:
            U_te[self.tlayer - 1] = 1 / (1 - R_te[self.tlayer - 1] * r_te[self.tlayer - 1] * np.exp(
                -2 * u[self.tlayer - 1] * (self.h[0, self.tlayer - 1] - self.h[0, self.tlayer - 2]))) * \
                                    R_te[self.tlayer - 1] * (
                                            self.kernel_te_down_sign * r_te[self.tlayer - 1] * np.exp(
                                        u[self.tlayer - 1] * (self.h[0, self.tlayer - 2] - 2 * self.h[
                                            0, self.tlayer - 1] + self.tz[
                                                                  0])) + self.kernel_te_up_sign * np.exp(
                                        u[self.tlayer - 1] * (self.h[0, self.tlayer - 2] - self.tz[0])))
            U_tm[self.tlayer - 1] = 1 / (1 - R_tm[self.tlayer - 1] * r_tm[self.tlayer - 1] * np.exp(
                -2 * u[self.tlayer - 1] * (self.h[0, self.tlayer - 1] - self.h[0, self.tlayer - 2]))) * \
                                    R_tm[self.tlayer - 1] * (
                                            self.kernel_tm_down_sign * r_tm[self.tlayer - 1] * np.exp(
                                        u[self.tlayer - 1] * (self.h[0, self.tlayer - 2] - 2 * self.h[
                                            0, self.tlayer - 1] + self.tz[
                                                                  0])) + self.kernel_tm_up_sign * np.exp(
                                        u[self.tlayer - 1] * (self.h[0, self.tlayer - 2] - self.tz[0])))
            D_te[self.tlayer - 1] = 1 / (1 - R_te[self.tlayer - 1] * r_te[self.tlayer - 1] * np.exp(
                -2 * u[self.tlayer - 1] * (self.h[0, self.tlayer - 1] - self.h[0, self.tlayer - 2]))) * \
                                    r_te[self.tlayer - 1] * (
                                            self.kernel_te_up_sign * R_te[self.tlayer - 1] * np.exp(
                                        -u[self.tlayer - 1] * (self.h[0, self.tlayer - 1] - 2 * self.h[
                                            0, self.tlayer - 2] + self.tz[
                                                                   0])) + self.kernel_te_down_sign * np.exp(
                                        -u[self.tlayer - 1] * (self.h[0, self.tlayer - 1] - self.tz[0])))
            D_tm[self.tlayer - 1] = 1 / (1 - R_tm[self.tlayer - 1] * r_tm[self.tlayer - 1] * np.exp(
                -2 * u[self.tlayer - 1] * (self.h[0, self.tlayer - 1] - self.h[0, self.tlayer - 2]))) * \
                                    r_tm[self.tlayer - 1] * (
                                            self.kernel_tm_up_sign * R_tm[self.tlayer - 1] * np.exp(
                                        -u[self.tlayer - 1] * (self.h[0, self.tlayer - 1] - 2 * self.h[
                                            0, self.tlayer - 2] + self.tz[0])) + self.kernel_tm_down_sign * np.exp(
                                        -u[self.tlayer - 1] * (self.h[0, self.tlayer - 1] - self.tz[0])))

        # for the layers above the tlayer
        if self.rlayer < self.tlayer:
            if self.tlayer == self.num_layer:
                D_te[self.tlayer - 2] = (Y[self.tlayer - 2] * (1 + R_te[self.tlayer - 1]) + Y[
                    self.tlayer - 1] * (1 - R_te[self.tlayer - 1])) / (2 * Y[self.tlayer - 2]) \
                                        * self.kernel_te_up_sign * (np.exp(
                    -u[self.tlayer - 1] * (self.tz[0] - self.h[0, self.tlayer - 2])))
                D_tm[self.tlayer - 2] = (Z[self.tlayer - 2] * (1 + R_tm[self.tlayer - 1]) + Z[
                    self.tlayer - 1] * (1 - R_tm[self.tlayer - 1])) / (2 * Z[self.tlayer - 2]) \
                                        * self.kernel_tm_up_sign * (np.exp(
                    -u[self.tlayer - 1] * (self.tz[0] - self.h[0, self.tlayer - 2])))
            elif self.tlayer != 1 and self.tlayer != self.num_layer:
                D_te[self.tlayer - 2] = (Y[self.tlayer - 2] * (1 + R_te[self.tlayer - 1]) + Y[
                    self.tlayer - 1] * (1 - R_te[self.tlayer - 1])) / (2 * Y[self.tlayer - 2]) \
                                        * (D_te[self.tlayer - 1] * np.exp(
                    -u[self.tlayer - 1] * (self.h[0, self.tlayer - 1] - self.h[0, self.tlayer - 2])) \
                                           + self.kernel_te_up_sign * np.exp(
                            -u[self.tlayer - 1] * (self.tz[0] - self.h[0, self.tlayer - 2])))
                D_tm[self.tlayer - 2] = (Z[self.tlayer - 2] * (1 + R_tm[self.tlayer - 1]) + Z[
                    self.tlayer - 1] * (1 - R_tm[self.tlayer - 1])) / (2 * Z[self.tlayer - 2]) \
                                        * (D_tm[self.tlayer - 1] * np.exp(
                    -u[self.tlayer - 1] * (self.h[0, self.tlayer - 1] - self.h[0, self.tlayer - 2])) \
                                           + self.kernel_tm_up_sign * np.exp(
                            -u[self.tlayer - 1] * (self.tz[0] - self.h[0, self.tlayer - 2])))

            for jj in range(self.tlayer - 2, 0, -1):
                D_te[jj - 1] = (Y[jj - 1] * (1 + R_te[jj]) + Y[jj] * (1 - R_te[jj])) / (2 * Y[jj - 1]) * D_te[
                    jj] * np.exp(-u[jj] * (self.h[0, jj] - self.h[0, jj - 1]))
                D_tm[jj - 1] = (Z[jj - 1] * (1 + R_tm[jj]) + Z[jj] * (1 - R_tm[jj])) / (2 * Z[jj - 1]) * D_tm[
                    jj] * np.exp(-u[jj] * (self.h[0, jj] - self.h[0, jj - 1]))
            for jj in range(self.tlayer - 1, 1, -1):
                U_te[jj - 1] = D_te[jj - 1] * np.exp(u[jj - 1] * (self.h[0, jj - 2] - self.h[0, jj - 1])) * \
                               R_te[jj - 1]
                U_tm[jj - 1] = D_tm[jj - 1] * np.exp(u[jj - 1] * (self.h[0, jj - 2] - self.h[0, jj - 1])) * \
                               R_tm[jj - 1]
            U_te[0] = 0
            U_tm[0] = 0

        # for the layers below the tlayer
        if self.rlayer > self.tlayer:
            if self.tlayer == 1:
                U_te[self.tlayer] = (Y[self.tlayer] * (1 + r_te[self.tlayer - 1]) + Y[self.tlayer - 1] * (
                        1 - r_te[self.tlayer - 1])) / (2 * Y[self.tlayer]) \
                                    * self.kernel_te_down_sign * (
                                        np.exp(-u[self.tlayer - 1] * (self.h[0, self.tlayer - 1] - self.tz[0])))
                U_tm[self.tlayer] = (Z[self.tlayer] * (1 + r_tm[self.tlayer - 1]) + Z[self.tlayer - 1] * (
                        1 - r_tm[self.tlayer - 1])) / (2 * Z[self.tlayer]) \
                                    * self.kernel_tm_down_sign * (
                                        np.exp(-u[self.tlayer - 1] * (self.h[0, self.tlayer - 1] - self.tz[0])))
            elif self.tlayer != 1 and self.tlayer != self.num_layer:
                U_te[self.tlayer] = (Y[self.tlayer] * (1 + r_te[self.tlayer - 1]) + Y[self.tlayer - 1] * (
                        1 - r_te[self.tlayer - 1])) / (2 * Y[self.tlayer]) \
                                    * (U_te[self.tlayer - 1] * np.exp(
                    -u[self.tlayer - 1] * (self.h[0, self.tlayer - 1] - self.h[0, self.tlayer - 2])) \
                                       + self.kernel_te_down_sign * np.exp(
                            -u[self.tlayer - 1] * (self.h[0, self.tlayer - 1] - self.tz[0])))
                U_tm[self.tlayer] = (Z[self.tlayer] * (1 + r_tm[self.tlayer - 1]) + Z[self.tlayer - 1] * (
                        1 - r_tm[self.tlayer - 1])) / (2 * Z[self.tlayer]) \
                                    * (U_tm[self.tlayer - 1] * np.exp(
                    -u[self.tlayer - 1] * (self.h[0, self.tlayer - 1] - self.h[0, self.tlayer - 2])) \
                                       + self.kernel_tm_down_sign * np.exp(
                            -u[self.tlayer - 1] * (self.h[0, self.tlayer - 1] - self.tz[0])))
            for jj in range(self.tlayer + 2, self.num_layer + 1):
                U_te[jj - 1] = (Y[jj - 1] * (1 + r_te[jj - 2]) + Y[jj - 2] * (1 - r_te[jj - 2])) / (
                        2 * Y[jj - 1]) * U_te[jj - 2] * np.exp(
                    -u[jj - 2] * (self.h[0, jj - 2] - self.h[0, jj - 3]))
                U_tm[jj - 1] = (Z[jj - 1] * (1 + r_tm[jj - 2]) + Z[jj - 2] * (1 - r_tm[jj - 2])) / (
                        2 * Z[jj - 1]) * U_tm[jj - 2] * np.exp(
                    -u[jj - 2] * (self.h[0, jj - 2] - self.h[0, jj - 3]))
            for jj in range(self.tlayer + 1, self.num_layer):
                D_te[jj - 1] = U_te[jj - 1] * np.exp(-u[jj - 1] * (self.h[0, jj - 1] - self.h[0, jj - 2])) * \
                               r_te[jj - 1]
                D_tm[jj - 1] = U_tm[jj - 1] * np.exp(-u[jj - 1] * (self.h[0, jj - 1] - self.h[0, jj - 2])) * \
                               r_tm[jj - 1]
            D_te[self.num_layer - 1] = 0
            D_tm[self.num_layer - 1] = 0
        return U_te, U_tm, D_te, D_tm

    def compute_kernel(self, transmitter, omega):
        """
        送信点の種類に応じ、受信点の位置での核関数を求める。
        ※ 齋藤(2019) 第三章 参照

        引数
        ----------
        transmitter : str
            送信器の名前
        omega : int or float
            角周波数

        返り値　※ selfに格納される
        -------
        kernel : dic
            電磁場の水平・z成分の核関数
        """

        ztilda = np.ones((1, self.num_layer, 1), dtype=np.complex)
        ytilda = np.ones((1, self.num_layer, 1), dtype=np.complex)
        ztilda[0, 0, 0] = 1j * omega * self.mu0
        ztilda[0, 1:self.num_layer, 0] = 1j * omega * self.mu[1:self.num_layer]

        if self.displacement_current:
            # ytilda[0, 0, 0] = 1e-13
            ytilda[0, 0, 0] = 1j * omega * self.epsrn
        else:
            # ytilda[0, 0, 0] = 1e-13
            ytilda[0, 0, 0] = 1e-13

        ytilda[0, 1:self.num_layer, 0] = self.sigma[0:self.num_layer - 1] #+ 1j * omega * self.epsrn
        self.ztilda = ztilda
        self.ytilda = ytilda

        k = np.zeros((1, self.num_layer), dtype=np.complex)
        #k[0, 0] = (omega ** 2.0 * self.mu[0] * self.epsrn) ** 0.5
        # k[0, 1:self.num_layer] = (omega ** 2.0 * self.mu[1:self.num_layer] * self.epsrn \
        #                             - 1j * omega * self.mu[1:self.num_layer] * self.sigma) ** 0.5
        k[0, 0] = 0
        k[0, 1:self.num_layer] = (- 1j * omega * self.mu[1:self.num_layer] * self.sigma) ** 0.5
        self.k = k

        u = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=complex)
        u[0] = self.lamda
        for ii in range(1, self.num_layer):
            u[ii] = (self.lamda ** 2 - k[0,ii] ** 2) ** 0.5

        tanhuh = np.zeros((self.num_layer - 1, self.filter_length, self.num_dipole), dtype=complex)
        for ii in range(1, self.num_layer - 1):
            tanhuh[ii] = (1-np.exp(-2*u[ii]* self.thickness[ii - 1]))/((1+np.exp(-2*u[ii]*self.thickness[ii - 1])))


        Y = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        Z = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)

        for ii in range(0, self.num_layer):
            Y[ii] = u[ii] / self.ztilda[0, ii, 0]
            Z[ii] = u[ii] / self.ytilda[0, ii, 0]

        r_te, r_tm, R_te, R_tm = self.compute_reflection_coefficients(Y, Z, tanhuh)
        U_te, U_tm, D_te, D_tm = self.compute_damping_coefficients(r_te, r_tm, R_te, R_tm, u, Y, Z)

        # compute Damping coefficient
        if self.rlayer == 1:
            e_up = np.zeros((self.filter_length, self.num_dipole), dtype=np.complex)
            e_down = np.exp(u[self.rlayer - 1] * (self.z - self.h[0, self.rlayer - 1]))
        elif self.rlayer == self.num_layer:
            e_up = np.exp(-u[self.rlayer - 1] * (self.z - self.h[0, self.rlayer - 2]))
            e_down = np.zeros((self.filter_length, self.num_dipole), dtype=np.complex)
        else:
            e_up = np.exp(-u[self.rlayer - 1] * (self.z - self.h[0, self.rlayer - 2]))
            e_down = np.exp(u[self.rlayer - 1] * (self.z - self.h[0, self.rlayer - 1]))

        def krondel(nn, mm):
            if nn == mm:
                return 1
            else:
                return 0

        # kenel function for 1D electromagnetic
        if transmitter == "vmd":
            kernel_te = U_te[self.rlayer - 1] * e_up + D_te[self.rlayer - 1] * e_down   \
                        + krondel(self.rlayer, self.tlayer) * np.exp(-u[self.tlayer - 1] * np.abs(self.z - self.tz[0]))
            kernel_te_hr = U_te[self.rlayer - 1] * e_up - D_te[self.rlayer - 1] * e_down \
                           +  krondel(self.rlayer, self.tlayer) \
                           * (self.z - self.tz[0]) / np.abs(self.z - self.tz[0]) \
                           * np.exp(-u[self.tlayer - 1] * np.abs(self.z - self.tz[0]))
            kernel = []
            kernel_e_phai = kernel_te * self.lamda ** 2 / u[self.tlayer - 1]
            kernel_h_r = kernel_te_hr * self.lamda ** 2 * u[self.rlayer - 1] / u[self.tlayer - 1]
            kernel_h_z = kernel_e_phai * self.lamda
            kernel.extend((kernel_e_phai,kernel_h_r,kernel_h_z))
        elif transmitter == "circular_loop":
            kernel_te = U_te[self.rlayer - 1] * e_up + D_te[self.rlayer - 1] * e_down \
                        + krondel(self.rlayer, self.tlayer) * np.exp(-u[self.tlayer - 1] * np.abs(self.z - self.tz[0]))
            kernel_te_hr = -U_te[self.rlayer - 1] * e_up + D_te[self.rlayer - 1] * e_down \
                           - krondel(self.rlayer, self.tlayer) \
                           * (self.z - self.tz[0]) / np.abs(self.z - self.tz[0]) \
                           * np.exp(-u[self.tlayer - 1] * np.abs(self.z - self.tz[0]))
            kernel = []
            besk1 = scipy.special.jn(1, self.lamda * self.r)
            besk0 = scipy.special.jn(0, self.lamda * self.r)

            kernel_e_phai = kernel_te * self.lamda * besk1 / u[self.tlayer - 1]
            kernel_h_r = kernel_te_hr * self.lamda * besk1 * u[self.rlayer - 1] / u[self.tlayer - 1]
            kernel_h_z = kernel_te * self.lamda ** 2 * besk0 / u[self.tlayer - 1]
            kernel.extend((kernel_e_phai, kernel_h_r, kernel_h_z))
        elif transmitter == "coincident_loop":
            kernel_te = U_te[self.rlayer - 1] * e_up + D_te[self.rlayer - 1] * e_down \
                        - krondel(self.rlayer, self.tlayer) * np.exp(-u[self.tlayer - 1] * np.abs(self.z - self.tz[0]))
            kernel = []
            besk1rad = scipy.special.jn(1, self.lamda * self.rad)
            kernel_h_z = kernel_te * self.lamda * besk1rad / u[self.tlayer - 1]
            kernel.append(kernel_h_z)
        elif transmitter == "hmdx" or transmitter == "hmdy":
            kernel = []
            kernel_tm_er = (-U_tm[self.rlayer - 1] * e_up + D_tm[self.rlayer - 1] * e_down \
                        - np.sign(self.z - self.tz[0]) * krondel(self.rlayer, self.tlayer) \
                            * np.exp(-u[self.tlayer - 1] * np.abs(self.z -self.tz[0]))) \
                            * u[self.rlayer - 1] /  u[self.tlayer - 1]
            kernel_te_er = U_te[self.rlayer - 1] * e_up + D_te[self.rlayer - 1] * e_down \
                        + np.sign(self.z - self.tz[0]) * krondel(self.rlayer, self.tlayer) \
                        * np.exp(-u[self.tlayer - 1] * np.abs(self.z - self.tz[0]))
            kernel_tm_ez = (U_tm[self.rlayer - 1] * e_up + D_tm[self.rlayer - 1] * e_down \
                         + krondel(self.rlayer, self.tlayer) * np.exp(-u[self.tlayer - 1] * np.abs(self.z -self.tz[0]))) \
                         / u[self.tlayer - 1]
            kernel_tm_hr = (U_tm[self.rlayer - 1] * e_up + D_tm[self.rlayer - 1] * e_down \
                        + krondel(self.rlayer, self.tlayer) * np.exp(-u[self.tlayer - 1] * np.abs(self.z -self.tz[0]))) \
                        / u[self.tlayer - 1]
            kernel_te_hr = (-U_te[self.rlayer - 1] * e_up + D_te[self.rlayer - 1] * e_down \
                            - krondel(self.rlayer, self.tlayer) \
                            * np.exp(-u[self.tlayer - 1] * np.abs(self.z - self.tz[0]))) * u[self.rlayer - 1]
            kernel_te_hz = U_te[self.rlayer - 1] * e_up + D_te[self.rlayer - 1] * e_down \
                           + np.sign(self.z - self.tz[0]) * krondel(self.rlayer, self.tlayer) \
                          * np.exp(-u[self.tlayer - 1] * np.abs(self.z - self.tz[0]))
            kernel.extend((kernel_tm_er , kernel_te_er, kernel_tm_ez, kernel_tm_hr, kernel_te_hr, kernel_te_hz))
        elif transmitter == "ved" or transmitter == "z_grounded_wire":
            kernel_tm = U_tm[self.rlayer - 1] * e_up + D_tm[self.rlayer - 1] * e_down \
                        + krondel(self.rlayer, self.tlayer) * np.exp(-u[self.tlayer - 1] * np.abs(self.z - self.tz[0]))
            kernel_tm_er = -U_tm[self.rlayer - 1] * e_up + D_tm[self.rlayer - 1] * e_down \
                           - (self.z - self.tz[0]) / np.abs(self.z - self.tz[0]) * krondel(self.rlayer, self.tlayer)  \
                           * np.exp(-u[self.tlayer - 1] * np.abs(self.z - self.tz[0]))
            kernel = []
            kernel_e_phai = kernel_tm_er * u[self.rlayer - 1] / u[self.tlayer - 1]
            kernel_e_z = kernel_tm / u[self.tlayer - 1]
            kernel_h_r = kernel_tm / u[self.tlayer - 1]
            kernel.extend((kernel_e_phai, kernel_e_z ,kernel_h_r))
        elif transmitter == "hedx" or transmitter == "hedy" or \
              transmitter == "grounded_wire" or transmitter == "loop_source" or \
              transmitter == "x_line_source" or transmitter == "y_line_source":
            kernel = []
            # p.231,232,233 eq.4.142, 4.144, 4.147, 4.150, 4.152
            kernel_tm_er = (-U_tm[self.rlayer - 1] * e_up + D_tm[self.rlayer - 1] * e_down \
                        - krondel(self.rlayer, self.tlayer) * np.exp(-u[self.tlayer - 1] * np.abs(self.z - self.tz[0]))) \
                        * u[self.rlayer - 1]
            kernel_te_er = (U_te[self.rlayer - 1] * e_up + D_te[self.rlayer - 1] * e_down \
                        + krondel(self.rlayer, self.tlayer) * np.exp(-u[self.tlayer - 1] * np.abs(self.z - self.tz[0]))) \
                        / u[self.tlayer - 1]
            kernel_tm_ez = U_tm[self.rlayer - 1] * e_up + D_tm[self.rlayer - 1] * e_down \
                        + (1-krondel(self.z - 1e-2, self.tz[0])) * np.sign(self.z - self.tz[0]) * krondel(self.rlayer, self.tlayer) * np.exp(-u[self.tlayer - 1] * np.abs(self.z - self.tz[0]))
            kernel_tm_hr = U_tm[self.rlayer - 1] * e_up + D_tm[self.rlayer - 1] * e_down \
                            + np.sign(self.z - self.tz[0]) * krondel(self.rlayer, self.tlayer)  \
                            * np.exp(-u[self.tlayer - 1] * np.abs(self.z - self.tz[0]))
            kernel_te_hr = (-U_te[self.rlayer - 1] * e_up + D_te[self.rlayer - 1] * e_down \
                    - np.sign(self.z - self.tz[0]) * krondel(self.rlayer, self.tlayer) \
                    * np.exp(-u[self.tlayer - 1] * np.abs(self.z - self.tz[0]))) * u[self.rlayer - 1] / u[self.tlayer - 1]
            kernel_te_hz = kernel_te_er
            kernel.extend((kernel_tm_er , kernel_te_er, kernel_tm_ez, kernel_tm_hr, kernel_te_hr, kernel_te_hz))
        return kernel

    def compute_hankel_transform(self, transmitter, omega):
        """
        送信点の種類毎に、デジタルフィルタ法を用いHankel変換を解く
        ※ 齋藤(2019) 第三章 参照
        Hankel変換の詳細については、第四章に記載

        引数
        ----------
        transmitter : str
            送信器の名前
        omega : int or float
            角周波数

        返り値　※ selfに格納される
        -------
        ans : array-like
            電磁場のx,y,z3成分
        """

        kernel = self.compute_kernel(transmitter, omega)
        ans = {}
        if transmitter == "vmd":
            e_phai = np.dot(self.wt1.T, kernel[0])/ self.r  # / self.r derive from digital filter convolution
            h_r = np.dot(self.wt1.T, kernel[1]) / self.r
            if self.fdtd == 4:
                h_z = np.dot(self.wt0.T, kernel[2] * (omega**0.5)) / self.r
            else:
                h_z = np.dot(self.wt0.T, kernel[2]) / self.r
            ans["e_x"] = - 1 / (4 * np.pi) * self.ztilda[0, self.tlayer - 1] * - self.sin_phai * e_phai
            ans["e_y"] = - 1 / (4 * np.pi) * self.ztilda[0, self.tlayer - 1] * self.cos_phai * e_phai
            ans["e_z"] = 0
            ans["h_x"] = 1 / (4 * np.pi) * self.cos_phai * h_r
            ans["h_y"] = 1 / (4 * np.pi) * self.sin_phai * h_r
            ans["h_z"] = self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1] / (4 * np.pi ) * h_z
        elif transmitter == "hmdx":
            tm_er_1 = np.dot(self.wt0.T, kernel[0] * self.lamda) / self.r
            tm_er_2 = np.dot(self.wt1.T, kernel[0]) / self.r
            te_er_1 = np.dot(self.wt0.T, kernel[1] * self.lamda) / self.r
            te_er_2 = np.dot(self.wt1.T, kernel[1]) / self.r
            tm_ez = np.dot(self.wt1.T, kernel[2] * self.lamda**2) / self.r
            tm_hr_1 = np.dot(self.wt0.T, kernel[3] * self.lamda) / self.r
            tm_hr_2 = np.dot(self.wt1.T, kernel[3]) / self.r
            te_hr_1 = np.dot(self.wt0.T, kernel[4] * self.lamda) / self.r
            te_hr_2 = np.dot(self.wt1.T, kernel[4]) / self.r
            te_hz = np.dot(self.wt1.T, kernel[5]* self.lamda**2) / self.r


            amp_tm_ex_1 = -(self.ztilda * self.ytilda)[0,self.tlayer - 1]  * (self.x - self.tx[0]) * (self.y - self.ty[0]) \
                      / (4 * np.pi * self.ytilda[0, self.rlayer - 1] * self.r ** 2)
            amp_tm_ex_2 = (self.ztilda * self.ytilda)[0,self.tlayer - 1]    * (self.x - self.tx[0]) * (self.y - self.ty[0]) \
                      / (2 * np.pi * self.ytilda[0, self.rlayer - 1] * self.r ** 3)
            amp_te_ex_1 = - self.ztilda[0, self.tlayer - 1] * (self.x - self.tx[0]) * (self.y - self.ty[0]) \
                      / (4 * np.pi * self.r ** 2)
            amp_te_ex_2 = self.ztilda[0, self.tlayer - 1] * (self.x - self.tx[0]) * (self.y - self.ty[0]) \
                      / (2 * np.pi * self.r ** 3)
            amp_tm_ey_1 = -(self.ztilda * self.ytilda)[0,self.tlayer - 1]  * (self.y - self.ty[0]) ** 2  \
                       / (4 * np.pi * self.ytilda[0, self.rlayer - 1] * self.r ** 2)
            amp_tm_ey_2 = --(self.ztilda * self.ytilda)[0,self.tlayer - 1]/ (4 * np.pi * self.ytilda[0, self.rlayer - 1]) \
                     * (2 * (self.y - self.ty[0]) ** 2 / self.r ** 3 - 1 / self.r)
            amp_te_ey_1 = self.ztilda[0, self.tlayer - 1] * (self.x - self.tx[0]) ** 2 / (4 * np.pi * self.r ** 2)
            amp_te_ey_2 = - self.ztilda[0, self.tlayer - 1]  / (4 * np.pi) \
                      * (2 * (self.x - self.tx[0]) ** 2 / self.r ** 3 - 1 / self.r)
            amp_tm_ez = -(self.ztilda )[0,self.tlayer - 1] * (self.y - self.ty[0]) / (4 * np.pi * self.r)

            amp_tm_hx_1 = self.k[0,self.tlayer - 1] ** 2  * (self.y - self.ty[0]) ** 2 / self.r ** 2 / (4 * np.pi)
            amp_tm_hx_2 =  - self.k[0,self.tlayer - 1] ** 2 * (2 * (self.y - self.ty[0]) ** 2 / self.r ** 3 - 1 / self.r) / (4 * np.pi)
            amp_te_hx_1 = (self.x - self.tx[0]) ** 2 / (4 * np.pi * self.r ** 2) * self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1]
            amp_te_hx_2 = - self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1]  * (2 * (self.x - self.tx[0]) ** 2 / self.r ** 3 - 1 / self.r) / (4 * np.pi)

            amp_tm_hy_1 = -self.k[0,self.tlayer - 1]** 2 / (4 * np.pi) * (self.x - self.tx[0]) * (self.y - self.ty[0]) / self.r ** 2
            amp_tm_hy_2 = - amp_tm_hy_1 / self.r * 2
            amp_te_hy_1 = self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1] \
                      / (4 * np.pi) * (self.x - self.tx[0]) * (self.y - self.ty[0]) / self.r ** 2
            amp_te_hy_2 = -self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1] \
                       / (2 * np.pi) * (self.x - self.tx[0]) * (self.y - self.ty[0]) / self.r ** 3
            amp_te_hz = self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1] * (self.x - self.tx[0]) / (4 * np.pi * self.r)

            ans["e_x"] = amp_tm_ex_1 * tm_er_1 + amp_tm_ex_2 * tm_er_2 + amp_te_ex_1 * te_er_1 + amp_te_ex_2 * te_er_2
            ans["e_y"] = amp_tm_ey_1 * tm_er_1 + amp_tm_ey_2 * tm_er_2 + amp_te_ey_1 * te_er_1 + amp_te_ey_2 * te_er_2
            ans["e_z"] = amp_tm_ez * tm_ez
            ans["h_x"] = amp_tm_hx_1 * tm_hr_1 + amp_tm_hx_2 * tm_hr_2 + amp_te_hx_1 * te_hr_1 + amp_te_hx_2 * te_hr_2
            ans["h_y"] = amp_tm_hy_1 * tm_hr_1 + amp_tm_hy_2 * tm_hr_2 + amp_te_hy_1 * te_hr_1 + amp_te_hy_2 * te_hr_2
            ans["h_z"] = amp_te_hz * te_hz
        elif transmitter == "hmdy":
            tm_er_1 = np.dot(self.wt0.T, kernel[0] * self.lamda) / self.r
            tm_er_2 = np.dot(self.wt1.T, kernel[0]) / self.r
            te_er_1 = np.dot(self.wt0.T, kernel[1] * self.lamda) / self.r
            te_er_2 = np.dot(self.wt1.T, kernel[1]) / self.r
            tm_ez = np.dot(self.wt1.T, kernel[2] * self.lamda**2) / self.r
            tm_hr_1 = np.dot(self.wt0.T, kernel[3] * self.lamda) / self.r
            tm_hr_2 = np.dot(self.wt1.T, kernel[3]) / self.r
            te_hr_1 = np.dot(self.wt0.T, kernel[4] * self.lamda) / self.r
            te_hr_2 = np.dot(self.wt1.T, kernel[4]) / self.r
            te_hz = np.dot(self.wt1.T, kernel[5]* self.lamda**2) / self.r

            amp_tm_ex_1 = (self.ztilda * self.ytilda)[0,self.tlayer - 1] * (self.x - self.tx[0]) ** 2 \
                      / (4 * np.pi * self.ytilda[0, self.rlayer - 1] * self.r ** 2)
            amp_tm_ex_2 = - (self.ztilda * self.ytilda)[0,self.tlayer - 1] / (4 * np.pi * self.ytilda[0, self.rlayer - 1]) \
                     * (2 * (self.x - self.tx[0]) ** 2 / self.r ** 3 - 1 / self.r)
            amp_te_ex_1 = -self.ztilda[0, self.tlayer - 1] * (self.y - self.ty[0]) ** 2 \
                      / (4 * np.pi * self.r ** 2)
            amp_te_ex_2 =  self.ztilda[0, self.tlayer - 1] / (4 * np.pi) \
                       * (2 * (self.y - self.ty[0]) ** 2 / self.r ** 3 - 1 / self.r)

            amp_tm_ey_1 = (self.ztilda * self.ytilda)[0,self.tlayer - 1]    * (self.x - self.tx[0]) * (self.y - self.ty[0])  \
                       / (4 * np.pi * self.ytilda[0, self.rlayer - 1] * self.r ** 2)
            amp_tm_ey_2 = -(self.ztilda * self.ytilda)[0,self.tlayer - 1]* (self.x - self.tx[0]) * (self.y - self.ty[0]) \
                       / (2 * np.pi * self.ytilda[0, self.rlayer - 1]* self.r ** 3)
            amp_te_ey_1 = self.ztilda[0, self.tlayer - 1] * (self.x - self.tx[0]) * (self.y - self.ty[0]) / (4 * np.pi * self.r ** 2)
            amp_te_ey_2 = - self.ztilda[0, self.tlayer - 1]  * (self.x - self.tx[0]) * (self.y - self.ty[0])  / (2 * np.pi * self.r ** 3)

            amp_tm_hx_1 = (self.ztilda * self.ytilda)[0,self.tlayer - 1] * (self.x - self.tx[0]) * (self.y - self.ty[0]) / (4 * np.pi * self.r ** 2)
            amp_tm_hx_2 = - amp_tm_hx_1 * 2 / self.r
            amp_te_hx_1 = self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1] \
                        * (self.x - self.tx[0]) * (self.y - self.ty[0]) / (4 * np.pi * self.r ** 2)
            amp_te_hx_2 = - amp_te_hx_1* 2 / self.r
            amp_tm_hy_1 = -(self.ztilda * self.ytilda)[0,self.tlayer - 1]    * (self.x - self.tx[0]) ** 2 / (4 * np.pi * self.r ** 2)
            amp_tm_hy_2 = (self.ztilda * self.ytilda)[0,self.tlayer - 1]    / (4 * np.pi) * (2 * (self.x - self.tx[0]) ** 2 / self.r ** 3 - 1 / self.r)
            amp_te_hy_1 = self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1] \
                        * (self.y - self.ty[0]) ** 2 / (4 * np.pi * self.r ** 2)
            amp_te_hy_2 = - self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1] \
                      * (2 * (self.y - self.ty[0]) ** 2 / self.r ** 3 - 1 / self.r) / (4 * np.pi)
            amp_te_hz =  self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1] * (self.y - self.ty[0]) / (4 * np.pi * self.r)

            ans["e_x"] = amp_tm_ex_1 * tm_er_1 + amp_tm_ex_2 * tm_er_2 + amp_te_ex_1 * te_er_1 + amp_te_ex_2 * te_er_2
            ans["e_y"] = amp_tm_ey_1 * tm_er_1 + amp_tm_ey_2 * tm_er_2 + amp_te_ey_1 * te_er_1 + amp_te_ey_2 * te_er_2
            ans["e_z"] = -(self.ztilda * self.ytilda)[0,self.tlayer - 1]    * (self.x - self.tx[0]) / (4 * np.pi * self.ytilda[0, self.rlayer - 1] * self.r) * tm_ez
            ans["h_x"] = amp_tm_hx_1 * tm_hr_1 + amp_tm_hx_2 * tm_hr_2 + amp_te_hx_1 * te_hr_1 + amp_te_hx_2 * te_hr_2
            ans["h_y"] = amp_tm_hy_1 * tm_hr_1 + amp_tm_hy_2 * tm_hr_2 + amp_te_hy_1 * te_hr_1 + amp_te_hy_2 * te_hr_2
            ans["h_z"] = amp_te_hz * te_hz
        elif transmitter == "ved":
            e_phai = np.dot(self.wt1.T, kernel[0] * self.lamda ** 2) / self.r
            e_z = np.dot(self.wt0.T, kernel[1] * self.lamda ** 3) / self.r
            h_r = np.dot(self.wt1.T, kernel[2] * self.lamda ** 2) / self.r

            ans["e_x"] = - 1 / (4 * np.pi * self.ytilda[0, self.rlayer - 1]) * self.cos_phai * e_phai
            ans["e_y"] = - 1 / (4 * np.pi * self.ytilda[0, self.rlayer - 1]) * self.sin_phai * e_phai
            ans["e_z"] = 1 / (4 * np.pi * self.ytilda[0, self.rlayer - 1]) * e_z
            ans["h_x"] = - 1 / (4 * np.pi) * self.sin_phai * h_r
            ans["h_y"] = -1 / (4 * np.pi) * self.cos_phai * h_r
            ans["h_z"] = 0
        elif transmitter == "hedx":
            tm_er_1 = np.dot(self.wt0.T, kernel[0] * self.lamda) / self.r
            tm_er_2 = np.dot(self.wt1.T, kernel[0]) / self.r
            te_er_1 = np.dot(self.wt0.T, kernel[1] * self.lamda) / self.r
            te_er_2 = np.dot(self.wt1.T, kernel[1]) / self.r
            tm_ez = np.dot(self.wt1.T, kernel[2] * self.lamda ** 2) / self.r
            tm_hr_1 = np.dot(self.wt0.T, kernel[3] * self.lamda) / self.r
            tm_hr_2 = np.dot(self.wt1.T, kernel[3]) / self.r
            te_hr_1 = np.dot(self.wt0.T, kernel[4] * self.lamda) / self.r
            te_hr_2 = np.dot(self.wt1.T, kernel[4]) / self.r
            te_hz = np.dot(self.wt1.T, kernel[5] * self.lamda**2) / self.r

            amp_tm_ex_g_1 = (self.x - self.tx[0]) ** 2 / (4 * np.pi * self.ytilda[0, self.rlayer - 1] * self.r ** 2)
            amp_tm_ex_g_2 =  - (2 * (self.x - self.tx[0]) ** 2 / self.r ** 3 - 1 / self.r) / (4 * np.pi * self.ytilda[0, self.rlayer - 1])
            amp_te_ex_g_1 = + self.ztilda[0, self.tlayer - 1] * (self.x - self.tx[0]) ** 2 / (4 * np.pi * self.r ** 2)
            amp_te_ex_g_2 = - self.ztilda[0, self.tlayer - 1] * (2 * (self.x - self.tx[0]) ** 2 / self.r ** 3 - 1 / self.r) / (4 * np.pi)
            amp_te_ex_line = - self.ztilda[0, self.tlayer - 1] / (4 * np.pi)
            amp_tm_ey_g_1 = (self.x - self.tx[0]) * (self.y - self.ty[0]) / (4 * np.pi * self.ytilda[0, self.rlayer - 1] * self.r ** 2)
            amp_tm_ey_g_2 = - (self.x - self.tx[0]) * (self.y - self.ty[0]) / (2 * np.pi * self.ytilda[0, self.rlayer - 1]* self.r**3 )
            amp_te_ey_g_1 = + self.ztilda[0, self.tlayer - 1] * (self.x - self.tx[0]) * (self.y - self.ty[0]) / (4 * np.pi * self.r ** 2)
            amp_te_ey_g_2 = - self.ztilda[0, self.tlayer - 1] * (self.x - self.tx[0]) * (self.y - self.ty[0]) / (2 * np.pi * self.r ** 3)
            amp_tm_ez = (self.x - self.tx[0]) / (4 * np.pi * self.ytilda[0, self.rlayer - 1] * self.r)
            amp_tm_hx_g_1 = (self.x - self.tx[0]) * (self.y - self.ty[0]) / (4 * np.pi * self.r ** 2)
            amp_tm_hx_g_2 = - (self.x - self.tx[0]) * (self.y - self.ty[0]) / (2 * np.pi * self.r ** 3)
            amp_te_hx_g_1 = + (self.x - self.tx[0]) * (self.y - self.ty[0]) / (4 * np.pi * self.r ** 2) * self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1]
            amp_te_hx_g_2 = - (self.x - self.tx[0]) * (self.y - self.ty[0]) / (2 * np.pi * self.r ** 3) * self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1]
            amp_tm_hy_g_1 = -(self.x - self.tx[0]) ** 2 / (4 * np.pi * self.r ** 2)
            amp_tm_hy_g_2 = (2 * (self.x - self.tx[0]) ** 2 / self.r ** 3 - 1 / self.r) / (4 * np.pi)
            amp_te_hy_g_1 = - (self.x - self.tx[0]) ** 2 / (4 * np.pi * self.r ** 2) * self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1]
            amp_te_hy_g_2 = +(2 * (self.x - self.tx[0]) ** 2 / self.r ** 3 - 1 / self.r) / (4 * np.pi) * self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1]
            amp_te_hy_line = self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1] / (4 * np.pi)
            amp_te_hz_line = self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1] * (self.y - self.ty[0]) / (4 * np.pi * self.r)

            ans["e_x"] = amp_tm_ex_g_1 * tm_er_1 + amp_tm_ex_g_2 * tm_er_2 + amp_te_ex_g_1 * te_er_1 + amp_te_ex_g_2 * te_er_2 + amp_te_ex_line * te_er_1
            ans["e_y"] = amp_tm_ey_g_1 * tm_er_1 + amp_tm_ey_g_2 * tm_er_2 + amp_te_ey_g_1 * te_er_1 + amp_te_ey_g_2 * te_er_2
            ans["e_z"] = amp_tm_ez * tm_ez
            ans["h_x"] = amp_tm_hx_g_1 * tm_hr_1 + amp_tm_hx_g_2 * tm_hr_2 + amp_te_hx_g_1 * te_hr_1 + amp_te_hx_g_2 * te_hr_2
            ans["h_y"] = amp_tm_hy_g_1 * tm_hr_1 + amp_tm_hy_g_2 * tm_hr_2 + amp_te_hy_g_1 * te_hr_1 + amp_te_hy_g_2 * te_hr_2 + amp_te_hy_line * te_hr_1
            ans["h_z"] = amp_te_hz_line * te_hz
        elif transmitter == "hedy":
            tm_er_1 = np.dot(self.wt0.T, kernel[0] * self.lamda) / self.r
            tm_er_2 = np.dot(self.wt1.T, kernel[0]) / self.r
            te_er_1 = np.dot(self.wt0.T, kernel[1] * self.lamda) / self.r
            te_er_2 = np.dot(self.wt1.T, kernel[1]) / self.r
            tm_ez = np.dot(self.wt1.T, kernel[2] * self.lamda**2) / self.r
            tm_hr_1 = np.dot(self.wt0.T, kernel[3] * self.lamda) / self.r
            tm_hr_2 = np.dot(self.wt1.T, kernel[3]) / self.r
            te_hr_1 = np.dot(self.wt0.T, kernel[4] * self.lamda) / self.r
            te_hr_2 = np.dot(self.wt1.T, kernel[4]) / self.r
            te_hz = np.dot(self.wt1.T, kernel[5] * self.lamda**2) / self.r

            amp_tm_ex_g_1 = (self.x - self.tx[0]) * (self.y - self.ty[0]) / (4 * np.pi * self.ytilda[0, self.rlayer - 1] * self.r ** 2)
            amp_tm_ex_g_2 = - (self.x - self.tx[0]) * (self.y - self.ty[0])  / (2 * np.pi * self.ytilda[0, self.rlayer - 1] * self.r ** 3)
            amp_te_ex_g_1 = self.ztilda[0, self.tlayer - 1] * (self.x - self.tx[0]) * (self.y - self.ty[0]) / (4 * np.pi * self.r ** 2)
            amp_te_ex_g_2 = - self.ztilda[0, self.tlayer - 1] * (self.x - self.tx[0]) * (self.y - self.ty[0]) / (2 * np.pi * self.r ** 3)
            amp_tm_ey_g_1 = (self.y - self.ty[0]) ** 2 / (4 * np.pi * self.ytilda[0, self.rlayer - 1] * self.r ** 2)
            amp_tm_ey_g_2 = - (2 * (self.y - self.ty[0]) ** 2 / self.r ** 3 - 1 / self.r) / (4 * np.pi* self.ytilda[0, self.rlayer - 1])
            amp_te_ey_g_1 =  self.ztilda[0, self.tlayer - 1] * (self.y - self.ty[0]) ** 2 / (4 * np.pi * self.r ** 2)
            amp_te_ey_g_2 = -self.ztilda[0, self.tlayer - 1] * (2 * (self.y - self.ty[0]) ** 2 / self.r ** 3 - 1 / self.r) / (4 * np.pi)
            amp_te_ey_line = - self.ztilda[0, self.tlayer - 1] / (4 * np.pi)
            amp_tm_ez = (self.y - self.ty[0]) / (4 * np.pi * self.ytilda[0, self.rlayer - 1] * self.r)
            amp_tm_hx_g_1 = (self.y - self.ty[0]) ** 2 / (4 * np.pi * self.r ** 2)
            amp_tm_hx_g_2 = - (2 * (self.y - self.ty[0]) ** 2 / self.r ** 3 - 1 / self.r) / (4 * np.pi)
            amp_te_hx_g_1 = + (self.y - self.ty[0]) ** 2 / (4 * np.pi * self.r ** 2) * self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1]
            amp_te_hx_g_2 = - (2 * (self.y - self.ty[0]) ** 2 / self.r ** 3 - 1 / self.r) / (4 * np.pi) * self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1]
            amp_te_hx_line = -self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1] / (4 * np.pi)
            amp_tm_hy_g_1 = - (self.x - self.tx[0]) * (self.y - self.ty[0]) / (4 * np.pi * self.r ** 2)
            amp_tm_hy_g_2 = (self.x - self.tx[0]) * (self.y - self.ty[0]) / (2 * np.pi * self.r ** 3)
            amp_te_hy_g_1 = - (self.x - self.tx[0]) * (self.y - self.ty[0]) / (4 * np.pi * self.r ** 2) * self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1]
            amp_te_hy_g_2 = (self.x - self.tx[0]) * (self.y - self.ty[0]) / (2 * np.pi * self.r ** 3) * self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1]
            amp_te_hz_line = - self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1] * (self.x - self.tx[0]) / (4 * np.pi * self.r)

            ans["e_x"] = amp_tm_ex_g_1 * tm_er_1 + amp_tm_ex_g_2 * tm_er_2 + amp_te_ex_g_1 * te_er_1 + amp_te_ex_g_2 * te_er_2
            ans["e_y"] = amp_tm_ey_g_1 * tm_er_1 + amp_tm_ey_g_2 * tm_er_2 + amp_te_ey_g_1 * te_er_1 + amp_te_ey_g_2 * te_er_2 + amp_te_ey_line * te_er_1
            ans["e_z"] = amp_tm_ez * tm_ez
            ans["h_x"] = amp_tm_hx_g_1 * tm_hr_1 + amp_tm_hx_g_2 * tm_hr_2 + amp_te_hx_g_1 * te_hr_1 + amp_te_hx_g_2 * te_hr_2 + amp_te_hx_line * te_hr_1
            ans["h_y"] = amp_tm_hy_g_1 * tm_hr_1 + amp_tm_hy_g_2 * tm_hr_2 + amp_te_hy_g_1 * te_hr_1 + amp_te_hy_g_2 * te_hr_2
            ans["h_z"] = amp_te_hz_line * te_hz
        elif transmitter == "circular_loop":
            e_phai = np.dot(self.wt1.T, kernel[0]) / self.rad
            h_r = np.dot(self.wt1.T, kernel[1]) / self.rad
            h_z = np.dot(self.wt1.T, kernel[2]) / self.rad
            ans["e_x"] = self.ztilda[0, self.tlayer - 1] * self.rad * self.sin_phai / 2 * e_phai
            ans["e_y"] = -self.ztilda[0, self.tlayer - 1] * self.rad * self.cos_phai / 2 * e_phai
            ans["e_z"] = 0
            ans["h_x"] = -self.rad * self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1] * self.cos_phai / 2 * h_r
            ans["h_y"] = -self.rad * self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1] * self.sin_phai / 2 * h_r
            ans["h_z"] = self.rad * self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1] / 2 * h_z
        elif transmitter == "coincident_loop":
            h_z_co = np.dot(self.wt1.T, kernel[0]) / self.rad
            ans["e_x"] = 0
            ans["e_y"] = 0
            ans["e_z"] = 0
            ans["h_x"] = 0
            ans["h_y"] = 0
            ans["h_z"] = (1 * np.pi * self.rad ** 2 * h_z_co)
        elif transmitter == "grounded_wire":
            tm_er_g_first = np.dot(self.wt1.T, kernel[0][:, 0]) / self.rn[0, 0]
            tm_er_g_end = np.dot(self.wt1.T, kernel[0][:, self.num_dipole - 1]) / self.rn[0, self.num_dipole - 1]
            te_er_g_first = np.dot(self.wt1.T, kernel[1][:, 0]) / self.rn[0, 0]
            te_er_g_end = np.dot(self.wt1.T, kernel[1][:, self.num_dipole - 1]) / self.rn[0, self.num_dipole - 1]
            tm_ez_1 = np.dot(self.wt0.T, kernel[2][:, 0] * self.lamda[:, 0]) / self.rn[0, 0]
            tm_ez_2 = np.dot(self.wt0.T, kernel[2][:, self.num_dipole - 1] * self.lamda[:, self.num_dipole - 1]) / self.rn[0, self.num_dipole - 1]
            tm_hr_g_first = np.dot(self.wt1.T, kernel[3][:, 0]) / self.rn[0, 0]
            tm_hr_g_end = np.dot(self.wt1.T, kernel[3][:, self.num_dipole - 1]) / self.rn[0, self.num_dipole - 1]
            te_hr_g_first = np.dot(self.wt1.T, kernel[4][:, 0]) / self.rn[0, 0]
            te_hr_g_end = np.dot(self.wt1.T, kernel[4][:, self.num_dipole - 1]) / self.rn[0, self.num_dipole - 1]
            te_hz_l = np.dot(self.wt1.T, kernel[5] * self.lamda ** 2) / self.rn
            te_ex_l = np.dot(self.wt0.T, kernel[1] * self.lamda) / self.rn
            te_hy_l = np.dot(self.wt0.T, kernel[4] * self.lamda) / self.rn

            amp_tm_ex_1 = (self.xx[0] / self.rn[0,0]) / (4 * np.pi * self.ytilda[0, self.rlayer - 1])
            amp_tm_ex_2 = (-self.xx[self.num_dipole-1] / self.rn[0, self.num_dipole-1]) / (4 * np.pi * self.ytilda[0, self.rlayer - 1])
            amp_te_ex_1 = (self.xx[0] / self.rn[0, 0]) * self.ztilda[0, self.tlayer - 1] / (4 * np.pi)
            amp_te_ex_2 = (-self.xx[self.num_dipole-1] / self.rn[0, self.num_dipole-1]) * self.ztilda[0, self.tlayer - 1] / (4 * np.pi)
            te_ex_line = -self.ztilda[0, self.tlayer - 1] / (4 * np.pi)
            amp_tm_ey_1 = (self.yy[0] / self.rn[0,0]) / (4 * np.pi * self.ytilda[0, self.rlayer - 1])
            amp_tm_ey_2 = (-self.yy[self.num_dipole-1] / self.rn[0, self.num_dipole-1]) / (4 * np.pi * self.ytilda[0, self.rlayer - 1])
            amp_te_ey_1 = (self.yy[0] / self.rn[0, 0]) * self.ztilda[0, self.tlayer - 1] / (4 * np.pi)
            amp_te_ey_2 =  (-self.yy[self.num_dipole-1] / self.rn[0, self.num_dipole-1]) * self.ztilda[0, self.tlayer - 1] / (4 * np.pi)
            amp_tm_ez_1 = 1 / (4 * np.pi * self.ytilda[0, self.rlayer - 1])
            amp_tm_ez_2 = -1 / (4 * np.pi * self.ytilda[0, self.rlayer - 1])
            amp_tm_hx_1 = 1 / (4 * np.pi) * self.yy[0] / self.rn[0,0]
            amp_tm_hx_2 = - 1 / (4 *np.pi) * self.yy[self.num_dipole-1] / self.rn[0,self.num_dipole-1]
            amp_te_hx_1 = self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1] / (4 * np.pi) * self.yy[0] / self.rn[0,0]
            amp_te_hx_2 = - self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1] / (4 *np.pi) * self.yy[self.num_dipole-1] / self.rn[0,self.num_dipole-1]
            amp_tm_hy_1 = -1 / (4 * np.pi) * self.xx[0] / self.rn[0,0]
            amp_tm_hy_2 = 1 / ( 4 *np.pi) * self.xx[self.num_dipole-1] / self.rn[0,self.num_dipole-1]
            amp_te_hy_1 = -self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1] / (4 * np.pi) * self.xx[0] / self.rn[0,0]
            amp_te_hy_2 = self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1] / (4 *np.pi) * self.xx[self.num_dipole-1] / self.rn[0,self.num_dipole-1]
            te_hy_line = self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1] / (4 * np.pi)

            ans["e_x"] = (amp_tm_ex_1 * tm_er_g_first + amp_tm_ex_2 * tm_er_g_end + amp_te_ex_1 * te_er_g_first + amp_te_ex_2 * te_er_g_end) \
                       + te_ex_line * self.ds * np.dot(te_ex_l, np.ones((self.num_dipole,1)))
            ans["e_y"] = amp_tm_ey_1 * tm_er_g_first + amp_tm_ey_2 * tm_er_g_end + amp_te_ey_1 * te_er_g_first + amp_te_ey_2 * te_er_g_end
            ans["e_z"] = amp_tm_ez_1 * tm_ez_1 + amp_tm_ez_2 * tm_ez_2
            ans["h_x"] = (amp_tm_hx_1 * tm_hr_g_first + amp_tm_hx_2 * tm_hr_g_end + amp_te_hx_1 * te_hr_g_first + amp_te_hx_2 * te_hr_g_end)
            ans["h_y"] = amp_tm_hy_1 * tm_hr_g_first + amp_tm_hy_2 * tm_hr_g_end + amp_te_hy_1 * te_hr_g_first + amp_te_hy_2 * te_hr_g_end \
                         + te_hy_line * self.ds * np.dot(te_hy_l, np.ones((self.num_dipole,1)))
            ans["h_z"] = np.dot(self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1] * self.yy / self.rn * self.ds / (4*np.pi) , te_hz_l.T)
        elif transmitter == "loop_source":
            te_ex_l = np.dot(self.wt0.T, kernel[1] * self.lamda) / self.rn
            te_hy_l = np.dot(self.wt0.T, kernel[4] * self.lamda) / self.rn
            te_hz_l = np.dot(self.wt1.T, kernel[5] * self.lamda ** 2) / self.rn
            te_ex_line = -self.ztilda[0, self.tlayer - 1] / (4 * np.pi)
            te_hy_line = self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1] / (4 * np.pi)

            ans["e_x"] =  te_ex_line * self.ds * np.dot(te_ex_l, np.ones((self.num_dipole,1)))
            ans["e_y"] = 0
            ans["e_z"] = 0
            ans["h_x"] = 0
            ans["h_y"] = te_hy_line * self.ds * np.dot(te_hy_l, np.ones((self.num_dipole,1)))
            ans["h_z"] = np.dot(self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1] * self.yy / self.rn * self.ds / (4*np.pi) , te_hz_l.T)
        elif transmitter == "x_line_source":
            te_er_1 = np.dot(self.wt0.T, kernel[1] * self.lamda) / self.r
            te_hr_1 = np.dot(self.wt0.T, kernel[4] * self.lamda) / self.r
            te_hz = np.dot(self.wt1.T, kernel[5] * self.lamda**2) / self.r

            amp_te_ex_line = - self.ztilda[0, self.tlayer - 1] / (4 * np.pi)
            amp_te_hy_line = self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1] / (4 * np.pi)
            amp_te_hz_line = self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1] * (self.y - self.ty[0]) / (4 * np.pi * self.r)

            ans["e_x"] = self.ds * amp_te_ex_line * te_er_1
            ans["e_y"] = 0
            ans["e_z"] = 0
            ans["h_x"] = 0
            ans["h_y"] = self.ds * amp_te_hy_line * te_hr_1
            ans["h_z"] = self.ds * amp_te_hz_line * te_hz
        elif transmitter == "y_line_source":
            te_er_1 = np.dot(self.wt0.T, kernel[1] * self.lamda) / self.r
            te_hr_1 = np.dot(self.wt0.T, kernel[4] * self.lamda) / self.r
            te_hz = np.dot(self.wt1.T, kernel[5] * self.lamda ** 2) / self.r

            amp_te_ey_line = - self.ztilda[0, self.tlayer - 1] / (4 * np.pi)
            amp_te_hx_line = -self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1] / (4 * np.pi)
            amp_te_hz_line = - self.ztilda[0, self.tlayer - 1] / self.ztilda[0, self.rlayer - 1] * (self.x - self.tx[0]) / (4 * np.pi * self.r)
            ans["e_x"] = 0
            ans["e_y"] = self.ds * amp_te_ey_line * te_er_1
            ans["e_z"] = 0
            ans["h_x"] = self.ds * amp_te_hx_line * te_hr_1
            ans["h_y"] = 0
            ans["h_z"] = self.ds * amp_te_hz_line * te_hz
        return ans

    @abstractmethod
    def repeat_computation(self):
        """

        設定した周波数・時間に応じた電磁応答の計算

        返り値　
        -------
        ans : dic
            電磁場のx,y,z3成分
        """

        pass

    def vmd(self, dipole_mom):
        """
        (1)vmdに特有の変数の設定
        (2)測定周波数 or 時間に応じた繰り返し計算の実行
        　 (repeat_computation)のwrapper

        返り値　
        -------
        ans : dic
            電磁場のx,y,z3成分
        """
        transmitter = sys._getframe().f_code.co_name
        self.em1d_base()
        self.lamda = self.y_base / self.r
        self.moment = dipole_mom
        self.num_dipole = 1
        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = 0
        self.kernel_tm_down_sign = 0

        ans = self.repeat_computation(transmitter)
        return ans

    def hmdx(self, dipole_mom):
        """
        (1)hmdxに特有の変数の設定
        (2)測定周波数 or 時間に応じた繰り返し計算の実行
        　 (repeat_computation)のwrapper

        返り値　
        -------
        ans : dic
            電磁場のx,y,z3成分
        """

        transmitter = sys._getframe().f_code.co_name
        self.em1d_base()

        self.lamda = self.y_base / self.r
        self.moment = dipole_mom
        self.num_dipole = 1
        self.kernel_te_up_sign = -1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = 1
        self.kernel_tm_down_sign = 1

        ans = self.repeat_computation(transmitter)
        return ans

    def hmdy(self, dipole_mom):
        """
        (1)hmdyに特有の変数の設定
        (2)測定周波数 or 時間に応じた繰り返し計算の実行
        　 (repeat_computation)のwrapper

        返り値　
        -------
        ans : dic
            電磁場のx,y,z3成分
        """

        transmitter = sys._getframe().f_code.co_name
        self.em1d_base()

        self.lamda = self.y_base / self.r
        self.moment = dipole_mom
        self.num_dipole = 1
        self.kernel_te_up_sign = -1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = 1
        self.kernel_tm_down_sign = 1

        ans = self.repeat_computation(transmitter)
        return ans

    def ved(self, ds, current):
        """
        (1)vedに特有の変数の設定
        (2)測定周波数 or 時間に応じた繰り返し計算の実行
        　 (repeat_computation)のwrapper

        返り値　
        -------
        ans : dic
            電磁場のx,y,z3成分
        """

        transmitter = sys._getframe().f_code.co_name
        self.em1d_base()

        self.lamda = self.y_base / self.r
        self.moment = ds * current
        self.num_dipole = 1
        self.kernel_te_up_sign = 0
        self.kernel_te_down_sign = 0
        self.kernel_tm_up_sign = 1
        self.kernel_tm_down_sign = 1

        ans = self.repeat_computation(transmitter)
        return ans

    def hedx(self, ds, current):
        """
        (1)hedxに特有の変数の設定
        (2)測定周波数 or 時間に応じた繰り返し計算の実行
        　 (repeat_computation)のwrapper

        返り値　
        -------
        ans : dic
            電磁場のx,y,z3成分
        """

        transmitter = sys._getframe().f_code.co_name
        self.em1d_base()

        # 送信源固有のパラメータ設定
        self.lamda = self.y_base / self.r
        self.moment = ds * current
        self.num_dipole = 1
        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = -1
        self.kernel_tm_down_sign = 1
        ans = self.repeat_computation(transmitter)
        return ans

    def hedy(self, ds, current):
        """
        (1)hedyに特有の変数の設定
        (2)測定周波数 or 時間に応じた繰り返し計算の実行
        　 (repeat_computation)のwrapper

        返り値　
        -------
        ans : dic
            電磁場のx,y,z3成分
        """

        transmitter = sys._getframe().f_code.co_name
        self.em1d_base()

        # 送信源固有のパラメータ設定
        self.lamda = self.y_base / self.r
        self.moment = ds * current
        self.num_dipole = 1
        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = -1
        self.kernel_tm_down_sign = 1
        ans = self.repeat_computation(transmitter)
        return ans

    def arbitrary_magnetic_dipole(self, dipole_mom, phai, theta):
        """
        (1)arbitrary_magnetic_dipoleに特有の変数の設定
        (2)測定周波数 or 時間に応じた繰り返し計算の実行
        　 (repeat_computation)のwrapper

        返り値　
        -------
        ans : dic
            電磁場のx,y,z3成分
        """

        phai = phai * np.pi / 180
        theta = theta * np.pi / 180
        ans = np.zeros((self.num_plot, 6), dtype=complex)
        em_field ,self.freqs = self.hmdx(dipole_mom)
        ans[:, 0] = em_field["e_x"]
        ans[:, 1] = em_field["e_y"]
        ans[:, 2] = em_field["e_z"]
        ans[:, 3] = em_field["h_x"]
        ans[:, 4] = em_field["h_y"]
        ans[:, 5] = em_field["h_z"]
        ans_hmdx = ans

        ans = np.zeros((self.num_plot, 6), dtype=complex)
        em_field, self.freqs = self.hmdy(dipole_mom)
        ans[:, 0] = em_field["e_x"]
        ans[:, 1] = em_field["e_y"]
        ans[:, 2] = em_field["e_z"]
        ans[:, 3] = em_field["h_x"]
        ans[:, 4] = em_field["h_y"]
        ans[:, 5] = em_field["h_z"]
        ans_hmdy = ans

        ans = np.zeros((self.num_plot, 6), dtype=complex)
        em_field, self.freqs = self.vmd(dipole_mom)
        ans[:, 0] = em_field["e_x"]
        ans[:, 1] = em_field["e_y"]
        ans[:, 2] = em_field["e_z"]
        ans[:, 3] = em_field["h_x"]
        ans[:, 4] = em_field["h_y"]
        ans[:, 5] = em_field["h_z"]
        ans_vmd = ans

        ans = np.sin(phai) * (np.cos(theta) * ans_hmdx + np.sin(theta) * ans_hmdy) + np.cos(phai) * ans_vmd
        ans = {"e_x": ans[:, 0], "e_y": ans[:, 1], "e_z": ans[:, 2] , "h_x": ans[:, 3], "h_y": ans[:, 4], "h_z": ans[:, 5]}
        return ans, self.freqs

    def arbitrary_electric_dipole(self, ds, current, phai,theta):
        """
        (1)arbitrary_electric_dipoleに特有の変数の設定
        (2)測定周波数 or 時間に応じた繰り返し計算の実行
        　 (repeat_computation)のwrapper

        返り値　
        -------
        ans : dic
            電磁場のx,y,z3成分
        """

        phai = phai * np.pi / 180
        theta = theta * np.pi / 180
        ans = np.zeros((self.num_plot, 6), dtype=complex)
        em_field, self.freqs = self.hedx(ds, current)
        ans[:, 0] = em_field["e_x"]
        ans[:, 1] = em_field["e_y"]
        ans[:, 2] = em_field["e_z"]
        ans[:, 3] = em_field["h_x"]
        ans[:, 4] = em_field["h_y"]
        ans[:, 5] = em_field["h_z"]
        ans_hedx = ans

        ans = np.zeros((self.num_plot, 6), dtype=complex)
        em_field, self.freqs = self.hedy(ds, current)
        ans[:, 0] = em_field["e_x"]
        ans[:, 1] = em_field["e_y"]
        ans[:, 2] = em_field["e_z"]
        ans[:, 3] = em_field["h_x"]
        ans[:, 4] = em_field["h_y"]
        ans[:, 5] = em_field["h_z"]
        ans_hedy = ans

        ans = np.zeros((self.num_plot, 6), dtype=complex)
        em_field, self.freqs = self.ved(ds, current)
        ans[:, 0] = em_field["e_x"]
        ans[:, 1] = em_field["e_y"]
        ans[:, 2] = em_field["e_z"]
        ans[:, 3] = em_field["h_x"]
        ans[:, 4] = em_field["h_y"]
        ans[:, 5] = em_field["h_z"]
        ans_ved = ans

        ans = np.sin(phai) * (np.cos(theta) * ans_hedx + np.sin(theta) * ans_hedy) + np.cos(phai) * ans_ved
        ans = {"e_x": ans[:, 0], "e_y": ans[:, 1], "e_z": ans[:, 2], "h_x": ans[:, 3], "h_y": ans[:, 4], "h_z": ans[:, 5]}
        return ans, self.freqs

    def circular_loop(self, current,  rad, turns):
        """
        (1)circular_loopに特有の変数の設定
        (2)測定周波数 or 時間に応じた繰り返し計算の実行
        　 (repeat_computationのwrapper)

        引数　
        -------
        rad : int or float
            ループ半径 [m]

        返り値　
        -------
        ans : dic
            電磁場のx,y,z3成分
        """
        transmitter = sys._getframe().f_code.co_name
        self.em1d_base()
        if rad == 0:
            raise Exception("ループ半径を設定してください")

        self.rad = rad
        self.lamda = self.y_base / self.rad
        self.moment = current * turns
        self.num_dipole = 1
        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = 0
        self.kernel_tm_down_sign = 0


        ans = self.repeat_computation(transmitter)
        return ans

    def coincident_loop(self, current,rad, turns):
        """
        (1)coincident_loopに特有の変数の設定
        (2)測定周波数 or 時間に応じた繰り返し計算の実行
        　 (repeat_computation)のwrapper

        引数　
        -------
        rad : int or float
            ループ半径 [m]

        返り値　
        -------
        ans : dic
            誘導起電力
        """
        transmitter = sys._getframe().f_code.co_name
        self.em1d_base()
        if rad == 0:
            raise Exception("ループ半径を設定してください")
        self.rad = rad
        self.lamda = self.y_base / self.rad
        self.moment = current * turns ** 2
        self.num_dipole = 1
        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = 0
        self.kernel_tm_down_sign = 0


        ans = self.repeat_computation(transmitter)
        return ans

    def grounded_wire(self, current):
        """
        (1)grounded_wireに特有の変数の設定
        (2)測定周波数 or 時間に応じた繰り返し計算の実行
        　 (repeat_computation)のwrapper

        返り値　
        -------
        ans : dic
            電磁場のx,y,z3成分
        """
        if len(np.unique(self.tz)) == 1:
            transmitter = sys._getframe().f_code.co_name
            self.em1d_base()

            # 送信源固有のパラメータ設定
            self.tz.append(self.tz[0])
            self.d = np.sqrt((self.tx[1] - self.tx[0]) ** 2 + (self.ty[1] - self.ty[0]) ** 2)
            cos_theta = (self.tx[1] - self.tx[0]) / self.d
            sin_theta = (self.ty[1] - self.ty[0]) / self.d

            self.num_v  = len(self.tx) # v means vertex
            #num_v_end = self.num_v
            num_d = 2 * self.d

            tx_dipole = np.array([])
            ty_dipole = np.array([])
            tz_dipole = np.array([])

            tx_dipole = np.append(tx_dipole, np.linspace(self.tx[0], self.tx[1], num_d))
            ty_dipole = np.append(ty_dipole, np.linspace(self.ty[0], self.ty[1], num_d))
            tz_dipole = np.append(tz_dipole, np.linspace(self.tz[0], self.tz[1], num_d))

            diff_x = np.abs(self.tx[0] - self.tx[1]) / (np.trunc(num_d) - 1)
            diff_y = np.abs(self.ty[0] - self.ty[1]) / (np.trunc(num_d) - 1)
            diff_z = np.abs(self.tz[0] - self.tz[1]) / (np.trunc(num_d) - 1)
            tx_dipole = tx_dipole + diff_x / 2
            ty_dipole = ty_dipole + diff_y / 2
            tz_dipole = tz_dipole + diff_z / 2

            if self.tx[0] < self.tx[1]:
                self.tx_dipole = np.insert(tx_dipole, 0, tx_dipole[0] - diff_x)
            else:
                self.tx_dipole = np.append(tx_dipole, tx_dipole[-1] - diff_x)

            if self.ty[0] < self.ty[1]:
                self.ty_dipole = np.insert(ty_dipole, 0, ty_dipole[0] - diff_y)
            else:
                self.ty_dipole = np.append(ty_dipole, ty_dipole[-1] - diff_y)

            if self.tz[0] < self.tz[1]:
                self.tz_dipole = np.insert(tz_dipole, 0, tz_dipole[0] - diff_z)
            else:
                self.tz_dipole = np.append(tz_dipole, tz_dipole[-1] - diff_z)

            num_dipole = np.trunc(num_d) + 1 - 1
            self.num_dipole = int(num_dipole)

            tx = np.ones((1, self.num_dipole))
            ty = np.ones((1, self.num_dipole))
            tz = np.ones((1, self.num_dipole))
            for ii in range(0, self.num_dipole):
                tx[0, ii] = 0.5 * (self.tx_dipole[ii + 1] + self.tx_dipole[ii])
                ty[0, ii] = 0.5 * (self.ty_dipole[ii + 1] + self.ty_dipole[ii])
                tz[0, ii] = 0.5 * (self.tz_dipole[ii + 1] + self.tz_dipole[ii])

            self.ds = self.d / self.num_dipole

            def change_coordinate(x, y, z, cos_theta, sin_theta):
                rot_theta = np.array([[cos_theta, sin_theta, 0], [-sin_theta, cos_theta, 0], [0, 0, 1]])
                new_coordinate = np.dot(rot_theta, np.array([x, y, z]))
                return new_coordinate

            new_transmitter = np.ones((self.num_dipole, 3))
            for ii in range(0, self.num_dipole):
                new_transmitter[ii, :] = change_coordinate(tx[0, ii], ty[0, ii], tz[0, ii], cos_theta, sin_theta)

            new_receiver = change_coordinate(self.x, self.y, self.z, cos_theta, sin_theta)
            self.xx = new_receiver[0] - new_transmitter[:,0]
            self.yy = new_receiver[1] - new_transmitter[:,1]
            self.z =  new_receiver[2]

            self.rn = np.ones((1, self.num_dipole))
            for ii in range(0, self.num_dipole):
                self.rn[0, ii] = np.sqrt(self.xx[ii] ** 2 + self.yy[ii] ** 2)

            y_base_wire = np.ones((self.filter_length, self.num_dipole)) * self.y_base
            self.lamda = y_base_wire / self.rn  # get all rn's lamda

            self.moment = current
            self.kernel_te_up_sign = 1
            self.kernel_te_down_sign = 1
            self.kernel_tm_up_sign = -1
            self.kernel_tm_down_sign = 1

            em_field, self.freqs = self.repeat_computation(transmitter)
            ans = np.zeros((self.num_plot, 6), dtype=complex)
            ans[:, 0] = cos_theta * em_field["e_x"] - sin_theta * em_field["e_y"]
            ans[:, 1] = cos_theta * em_field["e_y"] + sin_theta * em_field["e_x"]
            ans[:, 2] = em_field["e_z"]
            ans[:, 3] = cos_theta * em_field["h_x"] - sin_theta * em_field["h_y"]
            ans[:, 4] = cos_theta * em_field["h_y"] + sin_theta * em_field["h_x"]
            ans[:, 5] = em_field["h_z"]

            ans = {"e_x": ans[:, 0], "e_y": ans[:, 1], "e_z": ans[:, 2] , "h_x": ans[:, 3], "h_y": ans[:, 4], "h_z": ans[:, 5]}
            return ans, self.freqs
        else:
            num_v = 1
            for ii in range(num_v):
                self.d = np.sqrt((self.tx[ii + 1] - self.tx[ii]) ** 2 + (self.ty[ii + 1] - self.ty[ii]) ** 2 + (
                            self.tz[ii + 1] - self.tz[ii]) ** 2)
                cos_theta = (self.tx[ii + 1] - self.tx[ii]) / np.sqrt(
                    (self.tx[ii + 1] - self.tx[ii]) ** 2 + (self.ty[ii + 1] - self.ty[ii]) ** 2)
                sin_theta = (self.ty[ii + 1] - self.ty[ii]) / np.sqrt(
                    (self.tx[ii + 1] - self.tx[ii]) ** 2 + (self.ty[ii + 1] - self.ty[ii]) ** 2)
                cos_phai = (self.tz[ii + 1] - self.tz[ii]) / self.d
                sin_phai = np.sqrt((self.tx[ii + 1] - self.tx[ii]) ** 2 + (self.ty[ii + 1] - self.ty[ii]) ** 2) / self.d
                theta = np.arccos(cos_theta)
                phai = np.arccos(cos_phai)

                print("theta = " + str(np.rad2deg(theta)))
                print("phai = " + str(np.rad2deg(phai)))

            num_d = 2 * self.d
            tx_dipole = np.array([])
            ty_dipole = np.array([])
            tz_dipole = np.array([])

            tx_dipole = np.append(tx_dipole, np.linspace(self.tx[0], self.tx[1], num_d))
            ty_dipole = np.append(ty_dipole, np.linspace(self.ty[0], self.ty[1], num_d))
            tz_dipole = np.append(tz_dipole, np.linspace(self.tz[0], self.tz[1], num_d))

            diff_x = np.abs(self.tx[0] - self.tx[1]) / (np.trunc(num_d) - 1)
            diff_y = np.abs(self.ty[0] - self.ty[1]) / (np.trunc(num_d) - 1)
            diff_z = np.abs(self.tz[0] - self.tz[1]) / (np.trunc(num_d) - 1)
            tx_dipole = tx_dipole + diff_x / 2
            ty_dipole = ty_dipole + diff_y / 2
            tz_dipole = tz_dipole + diff_z / 2

            if self.tx[0] < self.tx[1]:
                self.tx_dipole = np.insert(tx_dipole, 0, tx_dipole[0] - diff_x)
            else:
                self.tx_dipole = np.append(tx_dipole, tx_dipole[-1] - diff_x)

            if self.ty[0] < self.ty[1]:
                self.ty_dipole = np.insert(ty_dipole, 0, ty_dipole[0] - diff_y)
            else:
                self.ty_dipole = np.append(ty_dipole, ty_dipole[-1] - diff_y)

            if self.tz[0] < self.tz[1]:
                self.tz_dipole = np.insert(tz_dipole, 0, tz_dipole[0] - diff_z)
            else:
                self.tz_dipole = np.append(tz_dipole, tz_dipole[-1] - diff_z)

            num_dipole = np.trunc(num_d) + 1 - 1
            num_dipole = int(num_dipole)
            self.num = num_dipole

            self.ds = self.d / self.num

            ans = np.zeros((self.num_plot, 6), dtype=complex)
            ans_hedx = np.zeros((self.num_plot, 6), dtype=complex)
            ans_hedy = np.zeros((self.num_plot, 6), dtype=complex)
            ans_ved = np.zeros((self.num_plot, 6), dtype=complex)

            for jj in range(num_dipole):
                self.tx[0] = 0.5 * (self.tx_dipole[jj + 1] + self.tx_dipole[jj])
                self.ty[0] = 0.5 * (self.ty_dipole[jj + 1] + self.ty_dipole[jj])
                self.tz[0] = 0.5 * (self.tz_dipole[jj + 1] + self.tz_dipole[jj])
                self.r = np.sqrt((self.x - self.tx[0]) ** 2 + (self.y - self.ty[0]) ** 2)

                em_field_hedx, self.freqs = self.hedx(1,current)
                ans_hedx[:, 0] = em_field_hedx["e_x"] * self.ds
                ans_hedx[:, 1] = em_field_hedx["e_y"] * self.ds
                ans_hedx[:, 2] = em_field_hedx["e_z"] * self.ds
                ans_hedx[:, 3] = em_field_hedx["h_x"] * self.ds
                ans_hedx[:, 4] = em_field_hedx["h_y"] * self.ds
                ans_hedx[:, 5] = em_field_hedx["h_z"] * self.ds

                em_field_hedy, self.freqs = self.hedy(1,current)
                ans_hedy[:, 0] = em_field_hedy["e_x"] * self.ds
                ans_hedy[:, 1] = em_field_hedy["e_y"] * self.ds
                ans_hedy[:, 2] = em_field_hedy["e_z"] * self.ds
                ans_hedy[:, 3] = em_field_hedy["h_x"] * self.ds
                ans_hedy[:, 4] = em_field_hedy["h_y"] * self.ds
                ans_hedy[:, 5] = em_field_hedy["h_z"] * self.ds

                em_field_ved, self.freqs = self.ved(1,current)
                ans_ved[:, 0] = em_field_ved["e_x"] * self.ds
                ans_ved[:, 1] = em_field_ved["e_y"] * self.ds
                ans_ved[:, 2] = em_field_ved["e_z"] * self.ds
                ans_ved[:, 3] = em_field_ved["h_x"] * self.ds
                ans_ved[:, 4] = em_field_ved["h_y"] * self.ds
                ans_ved[:, 5] = em_field_ved["h_z"] * self.ds

                ans = (sin_phai * (cos_theta * ans_hedx + sin_theta * ans_hedy) + cos_phai * ans_ved) + ans
            ans = {"e_x": ans[:, 0], "e_y": ans[:, 1], "e_z": ans[:, 2], "h_x": ans[:, 3], "h_y": ans[:, 4], "h_z": ans[:, 5]}
            return ans , self.freqs

    def loop_source(self, current, turns):
        """
        (1)loop_sourceに特有の変数の設定
        (2)測定周波数 or 時間に応じた繰り返し計算の実行
        　 (repeat_computation)のwrapper

        返り値　
        -------
        ans : dic
            電磁場のx,y,z3成分
        """

        if len(np.unique(self.tz)) == 1:
            transmitter = sys._getframe().f_code.co_name
            self.em1d_base()
            num_v = len(self.tx)

            # 送信源固有のパラメータ設定
            self.tx.append(self.tx[0])
            self.ty.append(self.ty[0])
            self.tz.append(self.tz[0])


            ans = np.zeros((self.num_plot, 6), dtype=complex)
            for ii in range (num_v):
                self.d = np.sqrt((self.tx[ii+1] - self.tx[ii]) ** 2 + (self.ty[ii+1] - self.ty[ii]) ** 2)
                cos_theta = (self.tx[ii+1] - self.tx[ii]) / self.d
                sin_theta = (self.ty[ii+1] - self.ty[ii]) / self.d

                num_d = 2 * self.d

                tx_dipole = np.array([])
                ty_dipole = np.array([])
                tz_dipole = np.array([])
                tx_dipole = np.append(tx_dipole, np.linspace(self.tx[ii], self.tx[ii+1], num_d))
                ty_dipole = np.append(ty_dipole, np.linspace(self.ty[ii], self.ty[ii+1], num_d))
                tz_dipole = np.append(tz_dipole, np.linspace(self.tz[ii], self.tz[ii+1], num_d))

                diff_x = np.abs(self.tx[ii] - self.tx[ii+1]) / (np.trunc(num_d) - 1)
                diff_y = np.abs(self.ty[ii] - self.ty[ii+1]) / (np.trunc(num_d) - 1)
                diff_z = np.abs(self.tz[ii] - self.tz[ii+1]) / (np.trunc(num_d) - 1)
                tx_dipole = tx_dipole + diff_x / 2
                ty_dipole = ty_dipole + diff_y / 2
                tz_dipole = tz_dipole + diff_z / 2

                if self.tx[ii] < self.tx[ii+1]:
                    self.tx_dipole = np.insert(tx_dipole, 0, tx_dipole[0] - diff_x)
                else:
                    self.tx_dipole = np.append(tx_dipole, tx_dipole[-1] - diff_x)

                if self.ty[ii] < self.ty[ii+1]:
                    self.ty_dipole = np.insert(ty_dipole, 0, ty_dipole[0] - diff_y)
                else:
                    self.ty_dipole = np.append(ty_dipole, ty_dipole[-1] - diff_y)

                if self.tz[ii] < self.tz[ii+1]:
                    self.tz_dipole = np.insert(tz_dipole, 0, tz_dipole[0] - diff_z)
                else:
                    self.tz_dipole = np.append(tz_dipole, tz_dipole[-1] - diff_z)

                num_dipole = np.trunc(num_d) + 1 - 1
                self.num_dipole = int(num_dipole)

                tx = np.ones((1, self.num_dipole))
                ty = np.ones((1, self.num_dipole))
                tz = np.ones((1, self.num_dipole))
                for ii in range(0, self.num_dipole):
                    tx[0, ii] = 0.5 * (self.tx_dipole[ii + 1] + self.tx_dipole[ii])
                    ty[0, ii] = 0.5 * (self.ty_dipole[ii + 1] + self.ty_dipole[ii])
                    tz[0, ii] = 0.5 * (self.tz_dipole[ii + 1] + self.tz_dipole[ii])

                self.ds = self.d / self.num_dipole


                def change_coordinate(x,y,z,cos_theta,sin_theta):
                    rot_theta = np.array([[cos_theta,sin_theta,0], [-sin_theta, cos_theta,0],[0,0,1]])
                    new_coordinate = np.dot(rot_theta, np.array([x, y, z]))
                    return new_coordinate

                new_transmitter = np.ones((self.num_dipole,3))
                for ii in range(0, self.num_dipole):
                    new_transmitter[ii,:] = change_coordinate(tx[0, ii], ty[0, ii], tz[0, ii],cos_theta,sin_theta)

                new_receiver = change_coordinate(self.x, self.y, self.z,cos_theta,sin_theta)

                self.xx = new_receiver[0] - new_transmitter[:,0]
                self.yy = new_receiver[1] - new_transmitter[:,1]
                self.z =  new_receiver[2]

                self.rn = np.ones((1, self.num_dipole))
                for ii in range(0, self.num_dipole):
                    self.rn[0, ii] = np.sqrt(self.xx[ii] ** 2 + self.yy[ii] ** 2)

                y_base_wire = np.ones((self.filter_length, self.num_dipole)) * self.y_base
                self.lamda = y_base_wire / self.rn  # get all rn's lamda
                self.moment = current * turns
                self.kernel_te_up_sign = 1
                self.kernel_te_down_sign = 1
                self.kernel_tm_up_sign = -1
                self.kernel_tm_down_sign = 1

                em_field, self.freqs = self.repeat_computation(transmitter)
                ans[:, 0] = cos_theta * em_field["e_x"] - sin_theta * em_field["e_y"] + ans[:, 0]
                ans[:, 1] = cos_theta * em_field["e_y"] + sin_theta * em_field["e_x"] + ans[:, 1]
                ans[:, 2] = em_field["e_z"] + ans[:, 2]
                ans[:, 3] = cos_theta * em_field["h_x"] - sin_theta * em_field["h_y"] + ans[:, 3]
                ans[:, 4] = cos_theta * em_field["h_y"] + sin_theta * em_field["h_x"] + ans[:, 4]
                ans[:, 5] = em_field["h_z"] + ans[:, 5]
            ans ={"e_x": ans[:, 0], "e_y": ans[:, 1], "e_z": ans[:, 2] , "h_x": ans[:, 3], "h_y": ans[:, 4], "h_z": ans[:, 5]}
            return ans, self.freqs
        else:
            num_v = len(self.tx)
            self.tx.append(self.tx[0])
            self.ty.append(self.ty[0])
            self.tz.append(self.tz[0])

            ans = 0
            for ii in range(num_v):
                self.d = np.sqrt((self.tx[ii + 1] - self.tx[ii]) ** 2 + (self.ty[ii + 1] - self.ty[ii]) ** 2 + (
                            self.tz[ii + 1] - self.tz[ii]) ** 2)
                cos_theta = (self.tx[ii + 1] - self.tx[ii]) / np.sqrt(
                    (self.tx[ii + 1] - self.tx[ii]) ** 2 + (self.ty[ii + 1] - self.ty[ii]) ** 2)
                sin_theta = (self.ty[ii + 1] - self.ty[ii]) / np.sqrt(
                    (self.tx[ii + 1] - self.tx[ii]) ** 2 + (self.ty[ii + 1] - self.ty[ii]) ** 2)
                # cos_phai = (self.tz[ii+1]-self.tz[ii]) / self.d
                sin_phai = np.sqrt((self.tx[ii + 1] - self.tx[ii]) ** 2 + (self.ty[ii + 1] - self.ty[ii]) ** 2) / self.d

                num_d = 2 * self.d
                tx_dipole = np.array([])
                ty_dipole = np.array([])
                tz_dipole = np.array([])

                tx_dipole = np.append(tx_dipole, np.linspace(self.tx[ii], self.tx[ii + 1], num_d))
                ty_dipole = np.append(ty_dipole, np.linspace(self.ty[ii], self.ty[ii + 1], num_d))
                tz_dipole = np.append(tz_dipole, np.linspace(self.tz[ii], self.tz[ii + 1], num_d))

                diff_x = np.abs(self.tx[ii] - self.tx[ii + 1]) / (np.trunc(num_d) - 1)
                diff_y = np.abs(self.ty[ii] - self.ty[ii + 1]) / (np.trunc(num_d) - 1)
                diff_z = np.abs(self.tz[ii] - self.tz[ii + 1]) / (np.trunc(num_d) - 1)
                tx_dipole = tx_dipole + diff_x / 2
                ty_dipole = ty_dipole + diff_y / 2
                tz_dipole = tz_dipole + diff_z / 2

                if self.tx[ii] < self.tx[ii + 1]:
                    self.tx_dipole = np.insert(tx_dipole, 0, tx_dipole[0] - diff_x)
                else:
                    self.tx_dipole = np.append(tx_dipole, tx_dipole[-1] - diff_x)

                if self.ty[ii] < self.ty[ii + 1]:
                    self.ty_dipole = np.insert(ty_dipole, 0, ty_dipole[0] - diff_y)
                else:
                    self.ty_dipole = np.append(ty_dipole, ty_dipole[-1] - diff_y)

                if self.tz[ii] < self.tz[ii + 1]:
                    self.tz_dipole = np.insert(tz_dipole, 0, tz_dipole[0] - diff_z)
                else:
                    self.tz_dipole = np.append(tz_dipole, tz_dipole[-1] - diff_z)

                num_dipole = np.trunc(num_d) + 1 - 1
                num_dipole = int(num_dipole)
                self.num = num_dipole

                self.ds = self.d / self.num

                ans_hedx = np.zeros((self.num_plot, 6), dtype=complex)
                ans_hedy = np.zeros((self.num_plot, 6), dtype=complex)
                ans_line = 0

                def x_line_source(self,current):
                    transmitter = sys._getframe().f_code.co_name
                    self.r = np.sqrt((self.x - self.tx[0]) ** 2 + (self.y - self.ty[0]) ** 2)
                    self.em1d_base()

                    self.lamda = self.y_base / self.r
                    self.moment = current
                    self.num_dipole = 1
                    self.kernel_te_up_sign = 1
                    self.kernel_te_down_sign = 1
                    self.kernel_tm_up_sign = -1
                    self.kernel_tm_down_sign = 1
                    ans, self.freqs = self.repeat_computation(transmitter)
                    return ans

                def y_line_source(self,current):
                    transmitter = sys._getframe().f_code.co_name
                    ans, self.freqs = self.repeat_computation(transmitter)
                    return ans

                for jj in range(num_dipole):
                    self.tx[0] = 0.5 * (self.tx_dipole[jj + 1] + self.tx_dipole[jj])
                    self.ty[0] = 0.5 * (self.ty_dipole[jj + 1] + self.ty_dipole[jj])
                    self.tz[0] = 0.5 * (self.tz_dipole[jj + 1] + self.tz_dipole[jj])

                    em_field_hedx = x_line_source(self,current)
                    ans_hedx[:, 0] = em_field_hedx["e_x"]
                    ans_hedx[:, 1] = em_field_hedx["e_y"]
                    ans_hedx[:, 2] = em_field_hedx["e_z"]
                    ans_hedx[:, 3] = em_field_hedx["h_x"]
                    ans_hedx[:, 4] = em_field_hedx["h_y"]
                    ans_hedx[:, 5] = em_field_hedx["h_z"]

                    em_field_hedy = y_line_source(self,current)
                    ans_hedy[:, 0] = em_field_hedy["e_x"]
                    ans_hedy[:, 1] = em_field_hedy["e_y"]
                    ans_hedy[:, 2] = em_field_hedy["e_z"]
                    ans_hedy[:, 3] = em_field_hedy["h_x"]
                    ans_hedy[:, 4] = em_field_hedy["h_y"]
                    ans_hedy[:, 5] = em_field_hedy["h_z"]

                    ans_line = sin_phai * (cos_theta * ans_hedx + sin_theta * ans_hedy) + ans_line
                ans = ans_line + ans
            ans = {"e_x": ans[:, 0] * turns, "e_y": ans[:, 1] * turns, "e_z": ans[:, 2]* turns, "h_x": ans[:, 3]* turns, "h_y": ans[:, 4]* turns,"h_z": ans[:, 5]* turns}
            return ans, self.freqs

class Fdem(BaseEm):
    """
    人工信号源による周波数領域電磁探査法の、一次元順解析をデジタルフィルタ法を用いて行う。
    ※ 詳細については、齋藤(2019)を参照。
    """
    def __init__(self, x, y, z, tx, ty, tz, res, thickness, hankel_filter, fdtd, dbdt, plot_number, freq, displacement_current=False):
        """

        コンストラクタ。インスタンスが生成される時に実行されるメソッド。
        サブクラス Fdem内で用いられる変数を設定する。

        引数
        ----------
        x :  array-like
           受信点のx座標 [m] ; 例：[10]
        y :  array-like
           受信点のy座標 [m] ; 例：[20]
        z :  array-like
           受信点のz座標 [m] ; 例：[30]
        tx :  array-like
           送信点のx座標 [m] ; 例：[40]　
        ty :  array-like
           送信点のy座標 [m] ; 例：[50]
        tz :  array-like
           送信点のz座標 [m] ; 例：[60]
        　　res : array-like
           層の比抵抗 [ohm-m] ; 例：np.array([100, 100])
        　　thickness : array-like
           層厚 [m]  ; 例：np.array([60])
        hankel_filter ： str
           Hankel変換用のデジタルフィルターの名前
        "Werthmuller201" , "mizunaga90", "anderson801", "kong241", "key201"のいずれか。
        fdtd ： int
           計算が、周波数領域か時間領域かを表す記号。時間領域の場合の、周波数⇒時間への変換方法の指定も兼ねる。
           1 : 周波数領域
        dbdt ： int
           受信応答が電磁場かその時間微分かを指定する記号。
           1: 電磁場 2 : 電磁場の時間微分
　       plot_number：int
           プロット数(測定点の数)
        freq ： array-like
            測定周波数 ; 例 np.logspace(-2, 5, self.plot_number)
        """

        # コンストラクタの継承
        super().__init__(x, y, z, tx, ty, tz, res, thickness, hankel_filter)
        # 必ず渡される引数
        self.fdtd = fdtd
        self.dbdt = dbdt
        self.num_plot = plot_number
        self.freqs = freq
        self.displacement_current = displacement_current

    def repeat_computation(self, transmitter):
        ans = np.zeros((self.num_plot, 6), dtype=complex)
        omega = 2 * np.pi * self.freqs
        for index, omega in enumerate(omega):
            em_field = self.compute_hankel_transform(transmitter, omega)
            # 電場の計算
            ans[index, 0] = em_field["e_x"]
            ans[index, 1] = em_field["e_y"]
            ans[index, 2] = em_field["e_z"]
            # 磁場の計算
            ans[index, 3] = em_field["h_x"]
            ans[index, 4] = em_field["h_y"]
            ans[index, 5] = em_field["h_z"]

            if self.dbdt == 0:
                ans[index, :] = ans[index, :] * 1j * omega

        ans = self.moment * ans
        ans = {"e_x": ans[:, 0], "e_y": ans[:, 1], "e_z": ans[:, 2], "h_x": ans[:, 3], "h_y": ans[:, 4], "h_z": ans[:, 5]}
        return ans, self.freqs

class Tdem(BaseEm):
    """
    人工信号源による時間領域電磁探査法の、一次元順解析をデジタルフィルタ法を用いて行う。
    ※ 詳細については、齋藤(2019)を参照。
    """
    def __init__(self, x, y, z, tx, ty, tz, res, thickness, hankel_filter, fdtd, dbdt, plot_number, dlag, time, displacement_current=False):

        """

        コンストラクタ。インスタンスが生成される時に実行されるメソッド。
        サブクラス Fdem内で用いられる変数を設定する。

        引数
        ----------
        x :  array-like
           受信点のx座標 [m] ; 例：[10]
        y :  array-like
           受信点のy座標 [m] ; 例：[20]
        z :  array-like
           受信点のz座標 [m] ; 例：[30]
        tx :  array-like
           送信点のx座標 [m] ; 例：[40]　
        ty :  array-like
           送信点のy座標 [m] ; 例：[50]
        tz :  array-like
           送信点のz座標 [m] ; 例：[60]
    　　res : array-like
           層の比抵抗 [ohm-m] ; 例：np.array([100, 100])
    　　thickness : array-like
           層厚 [m]  ; 例：np.array([60])
        hankel_filter ： str
           Hankel変換用のデジタルフィルターの名前
        "Werthmuller201" , "mizunaga90", "anderson801", "kong241", "key201"のいずれか。
        fdtd ： int
           計算が、周波数領域か時間領域かを表す記号。時間領域の場合の、周波数⇒時間への変換方法の指定も兼ねる。
           2:時間領域 (ffth・spline) 3 : 時間領域 (dlag(ffth・lagged convolution))
           4:時間領域(euler 2019 Serizawa(waseda)を参照)
        dbdt ： int
           受信応答が電磁場かその時間微分かを指定する記号。
           1: 電磁場 2 : 電磁場の時間微分
　       plot_number：int
           プロット数(測定点の数)
　       dlag：str
           dlagを用いる場合(fdtd=3の場合)の受信応答成分を入力。"e_x" or "e_y" or "e_z" or "h_x" or "h_y" or "h_z"
           fdtd=2の場合も、結果に影響しないが入力しておく。
        time ： array-like
            測定時間 ; 例 np.logspace(-6, 1, self.plot_number)
        """

        # コンストラクタの継承
        super().__init__(x, y, z, tx, ty, tz, res, thickness, hankel_filter)
        # 必ず渡される引数
        self.fdtd = fdtd
        self.dbdt = dbdt
        self.num_plot = plot_number
        self.dlag = dlag
        self.times = time
        self.displacement_current = displacement_current

    def repeat_computation(self, transmitter):
        if self.fdtd == 2 :
            ans = np.zeros((self.num_plot, 6),dtype=complex)
            from scipy import interpolate
            nFreqsPerDecade = 1000
            #freq = np.logspace(-6, 8, nFreqsPerDecade)　#werth
            #freq = np.logspace(-8, 12, nFreqsPerDecade) #key
            freq = np.logspace(-21, 21, nFreqsPerDecade) #ander
            freq_ans = np.zeros((len(freq),6), dtype=np.complex)
            omega = 2 * np.pi * freq
            for index, omega in enumerate(omega):
                hankel_result = self.compute_hankel_transform(transmitter, omega)
                freq_ans[index,0] = hankel_result["e_x"]
                freq_ans[index,1] = hankel_result["e_y"]
                freq_ans[index,2] = hankel_result["e_z"]
                freq_ans[index,3] = hankel_result["h_x"]
                freq_ans[index,4] = hankel_result["h_y"]
                freq_ans[index,5] = hankel_result["h_z"]

            f = interpolate.interp1d(2*np.pi*freq, freq_ans.T,kind='cubic')

            for time_index, time in enumerate(self.times):
                time_ans = self.ffht(time, f)
                ans[time_index, 0] = time_ans[0]
                ans[time_index, 1] = time_ans[1]
                ans[time_index, 2] = time_ans[2]
                ans[time_index, 3] = time_ans[3]
                ans[time_index, 4] = time_ans[4]
                ans[time_index, 5] = time_ans[5]
            ans = {"e_x": ans[:, 0], "e_y": ans[:, 1], "e_z": ans[:, 2], "h_x": ans[:, 3], "h_y": ans[:, 4], "h_z": ans[:, 5]}
            return ans , self.times
        elif self.fdtd == 3:
            nb = np.int(np.floor(10 * np.log(self.times[-1] / self.times[0])) + 1)
            ans = np.zeros((nb, 6), dtype=complex)
            emfield = {self.dlag:1}
            for ii, emfield in enumerate(emfield):
                if self.dbdt == 1:
                    time_ans, arg = self.dlagf0em(transmitter, nb, emfield)
                else:
                    time_ans, arg = self.dlagf1em(transmitter, nb, emfield)
                ans[:, ii] = time_ans
            ans = - 2 / np.pi * self.moment * ans

            # f = interpolate.interp1d(arg, ans.T, kind='cubic')
            # arg = self.times
            # ans = f(self.times).T
            return {self.dlag : ans[:, 0] / arg}, arg
        elif self.fdtd == 4:
            ans = np.zeros((self.num_plot - 1, 6), dtype=complex)
            self.y_base_time, self.wt0_time, self.wt1_time = load_fft_filter('raito_time_250')
            # matdata = scipy.io.loadmat('.' + self.filter_dir + "raito250_time.mat")
            # self.y_base_time = matdata["lamda"][0]
            # self.wt0_time = matdata["j0"][0]
            # self.wt1_time = matdata["j1"][0]
            self.filter_length_time = len(self.y_base_time)
            self.times_time = self.times[1:]
            for time_index, time in enumerate(self.times_time):
                hankel_result = self.euler_transform(transmitter, time)
                ans[time_index, 0] = hankel_result["e_x"]
                ans[time_index, 1] = hankel_result["e_y"]
                ans[time_index, 2] = hankel_result["e_z"]
                ans[time_index, 3] = hankel_result["h_x"]
                ans[time_index, 4] = hankel_result["h_y"]
                ans[time_index, 5] = hankel_result["h_z"]
            ans = {"e_x": ans[:, 0], "e_y": ans[:, 1], "e_z": ans[:, 2], "h_x": ans[:, 3], "h_y": ans[:, 4],
                   "h_z": ans[:, 5]}
            return ans, self.times_time


    def euler_transform(self, transmitter, time):
        """
        フーリエ変換のデジタルフィルタでオイラーのフィルタを用いた変換。
        時間微分でないものは後々実装予定。
        """
        ans = {}
        if self.dbdt == 0:
            e_x_set = np.zeros((self.filter_length_time, 1), dtype=complex)
            e_y_set = np.zeros((self.filter_length_time, 1), dtype=complex)
            e_z_set = np.zeros((self.filter_length_time, 1), dtype=complex)
            h_x_set = np.zeros((self.filter_length_time, 1), dtype=complex)
            h_y_set = np.zeros((self.filter_length_time, 1), dtype=complex)
            h_z_set = np.zeros((self.filter_length_time, 1), dtype=complex)
            time_range = time - self.times[0]
            for ii in range(self.filter_length_time):
                omega_set = self.y_base_time / time_range
                omega = omega_set[ii]
                hankel_result = self.compute_hankel_transform(transmitter, omega)
                e_x_set[ii] = hankel_result["e_x"]
                e_y_set[ii] = hankel_result["e_y"]
                e_z_set[ii] = hankel_result["e_z"]
                h_x_set[ii] = hankel_result["h_x"]
                h_y_set[ii] = hankel_result["h_y"]
                h_z_set[ii] = hankel_result["h_z"]

            ans["e_x"] = -np.dot(self.wt1_time.T, np.imag(e_x_set)) * (2.0 * time_range / np.pi) ** 0.5 / time_range
            ans["e_y"] = -np.dot(self.wt1_time.T, np.imag(e_y_set)) * (2.0 * time_range / np.pi) ** 0.5 / time_range
            ans["e_z"] = -np.dot(self.wt1_time.T, np.imag(e_z_set)) * (2.0 * time_range / np.pi) ** 0.5 / time_range
            ans["h_x"] = -np.dot(self.wt1_time.T, np.imag(h_x_set)) * (2.0 * time_range / np.pi) ** 0.5 / time_range
            ans["h_y"] = -np.dot(self.wt1_time.T, np.imag(h_y_set)) * (2.0 * time_range / np.pi) ** 0.5 / time_range
            ans["h_z"] = -np.dot(self.wt1_time.T, np.imag(h_z_set)) * (2.0 * time_range / np.pi) ** 0.5 / time_range

        return ans


    def ffht(self, time, f):
        """

        フーリエ正弦・余弦変換による周波数→時間領域への変換。
        (ただし、三次spline補間により計算時間を高速化)
        　
        返り値　※ selfに格納される
        ----------

        time :  int or float
            測定時刻
        f :  -
            spline補間により得られた周波数領域における電磁応答の多項式近似
        """

        base, cos, sin = load_fft_filter('anderson_sin_cos_filter_787')

        if self.dbdt == 1:
            #import key_filter_201 as kf
            #import Werthmuller_filter_201 as kf
            # import filter.anderson_sin_cos_filter_787 as kf
            omega_base = base / time
            f = f(omega_base)
            f_imag =  -2 / np.pi * np.imag(f) / omega_base
            ans = np.dot(f_imag, cos.T) / time
        else:
            #import Werthmuller_filter_201 as kf
            # import filter.anderson_sin_cos_filter_787 as kf
            #import key_filter_201 as kf
            omega_base = base / time
            f = f(omega_base)
            f_imag = (2 / np.pi) * np.imag(f)
            ans = np.dot(f_imag, sin.T) / time
        return ans

    def dlagf0em(self,transmitter,nb,emfield):
        abscis = 0.7866057737580476e0
        e = 1.10517091807564762e0
        er = .904837418035959573e0
        nofun = 0
        base, cos, sin = load_fft_filter('anderson_sin_cos_filter_787')
        # import filter.anderson_sin_cos_filter_787 as wt
        bmax = self.times[-1] #stet(2);
        tol = 1e-12
        ntol = 1
        key = np.zeros((1, 787))
        dwork = np.zeros((1, 787))
        dans = np.zeros((1, nb))[0]
        arg = np.zeros((1, nb))[0]

        # %C-----ERROR CHECKS
        if (nb < 1 or bmax <= 0.0e0):
            ierr = 1
            return
        y = bmax * er ** (np.fix(nb) - 1)
        if (y <= 0.0e0):
            ierr = 1
            return
        ierr = 0
        # %C-----INITIALIZE LAGGED CONVOLUTION LOOP
        for i in range (1, 788):
            key[0,i-1] = 0
        i = 787 + 1
        nb1 = np.fix(nb) + 1
        lag = -1

        # %C-----PRESET INITIAL FILTER ABSCISSA FOR STARTING BMAX, THE ARGUMENT
        # %C     USED IN THE EXTERNAL FUNCTION FUN(G).  NOTE THE ABSCISSAS
        # %C     ARE EQUALLY SPACED(E=EXP(.1e0), ER=1.0e0/E) IN LOG-SPACE.
        y1 = abscis / bmax
        # %C-----LAGGED CONVOLUTION LOOP 1010
        for ilag in range (1, nb + 1):
            lag = lag + 1
            istore = np.int(nb1 - ilag)
            if (lag > 0):
                y1 = y1 * e
            arg[istore-1] = abscis / y1
            # %C---------SPECIAL CASE FLAG NONE=1 IS SET IF FUN(G)=0 FOR ALL G IN
            # %C         FILTER FIXED RANGE(USING WEIGHTS 426-463).
            none = 0
            itol = np.fix(ntol)
            dsum = 0.0e0
            cmax = 0.0e0
            y = y1
            # %C---------BEGIN RIGHT SIDE CONVOLUTION AT WEIGHT 426(M=RETURN LABEL)
            m = 20
            i = 426
            y = y * e
            # %C---------CALL PSEUDO SUBROUTINE AT 100(RETURN TO 20 VIA M ASSIGNED)
            look = i + lag
            iq = look / 788
            ir = look % 788
            if (ir == 0):
                ir = 1
            iroll = iq * 787
            if (key[0,ir-1] <= iroll):
                # %C=========COMPUTE EXTERNAL FUN DIRECTLY ONLY WHEN NECESSARY
                key[0,ir-1] = iroll + ir
                g = y

                # % dwork(ir) = hankel(g,EmB,yBase) % g is nothing else but Omega
                hankel_result = self.compute_hankel_transform(transmitter, g)  # % g is nothing else but Omega
                dwork[:,ir-1] = np.imag(hankel_result[emfield]) / g
                nofun = np.fix(np.fix(nofun) + 1)

            c = dwork[0,ir-1] * cos[i-1]
            dsum = dsum + c
            goon = 1

            while (m != 0):
                while (goon == 1):
                    if (m == 20):
                        cmax = np.max([abs(c), cmax])
                        i = i + 1
                        y = y * e
                        if (i <= 461):
                            break
                        if (cmax == 0.0e0):
                            none = 1
                        cmax = tol * cmax
                        m = 30
                        break
                    if (m == 30):
                        if (~(abs(c) <= cmax)):
                            itol = np.fix(ntol)
                            i = i + 1
                            y = y * e
                            if (i <= 787):
                                break
                        itol = itol - 1
                        goon1 = 1
                        while (itol > 0 and i < 787):
                            i = i + 1
                            y = y * e
                            if (i <= 787):
                                goon1 = 0
                                break
                            itol = itol - 1
                        if (goon1 == 0):
                            break
                        itol = np.fix(ntol)
                        y = y1
                        m = 60
                        i = 425
                        break
                    if (m == 60):
                        if (~(abs(c) <= cmax and none == 0)):
                            itol = np.fix(ntol)
                            i = i - 1
                            y = y * er
                            if (i > 0):
                                break
                        itol = itol - 1
                        goon1 = 1
                        while (itol > 0 and i > 1):
                            i = i - 1
                            y = y * er
                            if (i > 0):
                                goon1 = 0
                                break
                            itol = itol - 1
                        if (goon1 == 0):
                            break
                        goon = 0
                        m = 0
                if (goon!=0):
                    look = i + lag
                    iq = look / 788
                    ir = look % 788
                    if (ir == 0):
                        ir = 1
                    iroll = iq * 787
                    if (key[0,ir-1] <= iroll):
                        key[0,ir-1] = iroll + ir
                        g = y

                        hankel_result = self.compute_hankel_transform(transmitter, g)  # g is nothing else but Omega
                        dwork[0,ir-1] = np.imag(hankel_result[emfield]) / g
                        nofun = np.fix(np.fix(nofun) + 1)
                    # %C=========USE EXISTING SAVED FUNCTIONAL VALUES IN DWORK(IR)
                    c = dwork[0,ir-1] * cos[i-1]
                    dsum = dsum + c
            dans[istore-1] = dsum #/ arg[istore-1]
            continue
        return dans,arg

    def dlagf1em(self, transmitter, nb, emfield):
        abscis = 0.7745022656977834e0
        e = 1.10517091807564762e0
        er = .904837418035959573e0
        nofun = 0
        base, cos, sin = load_fft_filter('anderson_sin_cos_filter_787')
        # import .filter_files.sin_cos_filter_787 as wt
        bmax = self.times[-1]  # stet(2);
        tol = 1e-12
        ntol = 1
        key = np.zeros((1, 787))
        dwork = np.zeros((1, 787))
        dans = np.zeros((1, nb))[0]
        arg = np.zeros((1, nb))[0]

        # %C-----ERROR CHECKS
        if (nb < 1 or bmax <= 0.0e0):
            ierr = 1
            return
        y = bmax * er ** (np.fix(nb) - 1)
        if (y <= 0.0e0):
            ierr = 1
            return
        ierr = 0
        # %C-----INITIALIZE LAGGEED CONVOLUTION LOOP
        for i in range (1, 788):
            key[0,i-1] = 0
        i = 787 + 1
        nb1 = np.fix(nb) + 1
        lag = -1

        # %C-----PRESET INITIAL FILTER ABSCISSA FOR STARTING BMAX, THE ARGUMENT
        # %C     USEe IN THE EXTERNAL FUNCTION FUN(G).  NOTE THE ABSCISSAS
        # %C     ARE EQUALLY SPACEe(E=eEXP(.1e0), ER=1.0e0/E) IN LOG-SPACE.
        y1 = abscis / bmax
        # %C-----LAGGED CONVOLUTION LOOP 1010
        for ilag in range (1, nb + 1):
            lag = lag + 1
            istore = np.int(nb1 - ilag)
            if (lag > 0):
                y1 = y1 * e
            arg[istore-1] = abscis / y1
            # %C---------SPECIAL CASE FLAG NONE=1 IS SET IF FUN(G)=0 FOR ALL G IN
            # %C         FILTER FIXED RANGE(USING WEIGHTS 426-463).
            none = 0
            itol = np.fix(ntol)
            dsum = 0.0e0
            cmax = 0.0e0
            y = y1
            # %C---------BEGIN RIGHT SIDE CONVOLUTION AT WEIGHT 426(M=RETURN LABEL)
            m = 20
            i = 426
            y = y * e
            # %C---------CALL PSEUeO SUBROUTINE AT 100(RETURN TO 20 VIA M ASSIGNEe)
            look = i + lag
            iq = look / 788
            ir = look % 788
            if (ir == 0):
                ir = 1
            iroll = iq * 787
            if (key[0,ir-1] <= iroll):
                # %C=========COMPUTE EXTERNAL FUN DIRECTLY ONLY WHEN NECESSARY
                key[0,ir-1] = iroll + ir
                g = y

                # % dwork(ir) = hankel(g,EmB,yBase) % g is nothing else but Omega
                hankel_result = self.compute_hankel_transform(transmitter, g)  # % g is nothing else but Omega
                dwork[0,ir-1] = np.imag(hankel_result[emfield])
                nofun = np.fix(np.fix(nofun) + 1)

            c = dwork[0,ir-1] * sin[i-1]
            dsum = dsum + c
            goon = 1


            while (m != 0):
                while (goon == 1):
                    if (m == 20):
                        cmax = np.max([abs(c), cmax])
                        i = i + 1
                        y = y * e
                        if (i <= 463):
                            break
                        if (cmax == 0.0e0):
                            none = 1
                        cmax = tol * cmax
                        m = 30
                        break
                    if (m == 30):
                        if (~(abs(c) <= cmax)):
                            itol = np.fix(ntol)
                            i = i + 1
                            y = y * e
                            if (i <= 787):
                                break
                        itol = itol - 1
                        goon1 = 1
                        while (itol > 0 and i < 787):
                            i = i + 1
                            y = y * e
                            if (i <= 787):
                                goon1 = 0
                                break
                            itol = itol - 1
                        if (goon1 == 0):
                            break
                        itol = np.fix(ntol)
                        y = y1
                        m = 60
                        i = 425
                        break
                    if (m == 60):
                        if (~(abs(c) <= cmax and none == 0)):
                            itol = np.fix(ntol)
                            i = i - 1
                            y = y * er
                            if (i > 0):
                                break
                        itol = itol - 1
                        goon1 = 1
                        while itol > 0 and i > 1:
                            i = i - 1
                            y = y * er
                            if i > 0:
                                goon1 = 0
                                break
                            itol = itol - 1
                        if goon1 == 0:
                            break
                        goon = 0
                        m = 0
                if goon != 0:
                    look = i + lag
                    iq = look / 788
                    ir = look % 788
                    if ir == 0:
                        ir = 1
                    iroll = iq * 787
                    if key[0, ir-1] <= iroll:
                        key[0, ir-1] = iroll + ir
                        g = y
                        hankel_result = self.compute_hankel_transform(transmitter, g)  # g is nothing else but Omega
                        dwork[0, ir-1] = np.imag(hankel_result[emfield])
                        nofun = np.fix(np.fix(nofun) + 1)
                    # %C=========USE EXISTING SAVED FUNCTIONAL VALUES IN DWORK(IR)
                    c = dwork[0, ir-1] * sin[i-1]
                    dsum = dsum + c
            dans[istore-1] = dsum  # / arg[istore-1]
            continue
        return dans, arg



