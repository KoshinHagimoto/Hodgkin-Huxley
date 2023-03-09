import numpy as np
import matplotlib.pyplot as plt


class HodgkinHuxley:

    def __init__(self, eta=0.01, Cm=1.0, gNa=120.0, gK=36.0, gL=0.3, ENa=55.0, EK=-72.0, EL=-49.387):
        self.eta = eta  # デルタ
        self.Cm = Cm  # 膜容量(uF/cm^2)
        self.gNa = gNa  # Na+ の最大コンダクタンス(mS/cm^2)
        self.gK = gK  # K+ の最大コンダクタンス(mS/cm^2)
        self.gL = gL  # 漏れイオンの最大コンダクタンス(mS/cm^2)
        self.ENa = ENa  # Na+ の平衡電位(mV)
        self.EK = EK  # K+ の平衡電位(mV)
        self.EL = EL  # 漏れイオンの平衡電位(mV)

        self.V = -65.0  # 静止膜電位は任意の値をとれる. -Vを-(V+65)に変更
        self.m = 0.05
        self.h = 0.6
        self.n = 0.32

    def alpha_m(self):
        return 0.1 * (self.V + 40.0) / (1.0 - np.exp(-(self.V + 40.0) / 10.0))

    def beta_m(self):
        return 4.0 * np.exp(-(self.V + 65.0) / 18.0)

    def alpha_h(self):
        return 0.07 * np.exp(-(self.V + 65.0) / 20.0)

    def beta_h(self):
        return 1.0 / (1.0 + np.exp(-(self.V + 35.0) / 10.0))

    def alpha_n(self):
        return 0.01 * (self.V + 55.0) / (1.0 - np.exp(-(self.V + 55.0) / 10.0))

    def beta_n(self):
        return 0.125 * np.exp(-(self.V + 65.0) / 80.0)

    def INa(self):
        return self.gNa * self.m ** 3 * self.h * (self.V - self.ENa)

    def IK(self):
        return self.gK * self.n ** 4 * (self.V - self.EK)

    def IL(self):
        return self.gL + (self.V - self.EL)

    def step(self, I_inj=0):
        self.m += (self.alpha_m() * (1.0 - self.m) - self.beta_m() * self.m) * self.eta
        self.h += (self.alpha_h() * (1.0 - self.h) - self.beta_h() * self.h) * self.eta
        self.n += (self.alpha_n() * (1.0 - self.n) - self.beta_n() * self.n) * self.eta
        self.V += ((I_inj - self.INa() - self.IK() - self.IL()) / self.Cm) * self.eta
        return self.m, self.h, self.n, self.V


def generate_I_inj(t):
    t = t / 10
    return -10 * (t > 1000) + 10 * (t > 2000) + 10 * (t > 3000) - 10 * (t > 4000) \
           + 20 * (t > 5000) - 20 * (t > 6000) + 30 * (t > 7000) - 30 * (t > 8000)

def fi_curve():
    HH_fi = HodgkinHuxley(eta=0.05)

    T = 1000  # run length (ms)
    steps = int(T / 0.05)
    n = 10

    I_inj_fi = np.linspace(0, 20, n)  # range of current (mA)
    rate = np.zeros(n)  # to store spike

    for i, I in enumerate(I_inj_fi):
        t = np.arange(0, steps)
        I_ext = np.zeros(steps)
        I_ext[:] = I

        V_list = np.empty(0)
        for j in I_ext:
            result = HH_fi.step(j)
            V_list = np.append(V_list, result[3])
        for j in range(200):
            V_list[j] = -65
        spike = (V_list[1:] < 0) & (V_list[:-1] >= 0)  # zero crossing
        rate[i] = sum(spike) * 1000 / T

    plt.plot(I_inj_fi, rate)  # F-U curve
    plt.ylabel("f (Hz)");
    plt.xlabel("I (mA)");
    plt.grid()
    plt.show()


def main():
    HH = HodgkinHuxley()

    steps = 90000
    t = np.arange(0, steps)
    I_inj = generate_I_inj(t)
    m_list = []
    h_list = []
    n_list = []
    V_list = []
    for i in I_inj:
        result = HH.step(i)
        m_list.append(result[0])
        h_list.append(result[1])
        n_list.append(result[2])
        V_list.append(result[3])
    plt.plot(t, V_list, label='H-H simulation')
    plt.xlabel('t[ms]')
    plt.ylabel('V[mV]')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(t, I_inj, label='I_inj', c='orange')
    plt.ylim(-10, 30)
    plt.xlabel('t[ms]')
    plt.ylabel('I[mA]')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(t, m_list, label='m(t)')
    plt.plot(t, h_list, label='h(t)')
    plt.plot(t, n_list, label='n(t)')
    plt.xlabel('t[ms]')
    plt.ylabel('V[mV]')
    plt.legend()
    plt.grid()
    plt.show()

    plt.xlabel("t (ms)", fontsize=20)
    plt.ylabel("V, I", fontsize=20)
    plt.plot(t, V_list, '-', label="voltage V")
    plt.plot(t, I_inj, '-', label="current I_inj")
    plt.legend(fontsize=9, loc="upper right")
    plt.grid()
    plt.show()

    fi_curve()

if __name__ == '__main__':
    main()