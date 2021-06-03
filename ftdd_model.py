import numpy as np
import matplotlib.pyplot as plt


class FTDD:
    def __init__(self, sd):
        self.sd = sd

        self.FreqReUse = 9  # 1 tier
        self.NoUpLink = 12
        self.NtwSizeA = -2000
        self.NtwSizeB = 2000
        self.PlusShift = 12000
        self.MinusShift = -12000
        self.LogNormalP = ''
        self.LogNormal = ''
        self.No_Iterations = 10000  # 10000으로 설정
        self.SIR = self.modeling()
        self.pdf = ''
        self.cdf = ''
        self.hist = ''
        self.bin_left = ''

    def modeling(self):
        SIR = np.zeros((1, self.NoUpLink))
        for Loop in range(0, self.No_Iterations):
            # user 데이터 12 명
            SubX = np.random.uniform(self.NtwSizeA, self.NtwSizeB, size=[self.FreqReUse, self.NoUpLink])
            SubY = np.random.uniform(self.NtwSizeA, self.NtwSizeB, size=[self.FreqReUse, self.NoUpLink])
            # wanted signal 발생하는 공간
            Cell_x0 = SubX[0, :]
            Cell_y0 = SubY[0, :]
            # IC1
            Cell_x1 = SubX[1, :]
            Cell_y1 = SubY[1, :] + self.PlusShift
            # IC2
            Cell_x2 = SubX[2, :] + self.PlusShift
            Cell_y2 = SubY[2, :] + self.PlusShift
            # IC3
            Cell_x3 = SubX[3, :] + self.PlusShift
            Cell_y3 = SubY[3, :]
            # IC4
            Cell_x4 = SubX[4, :] + self.PlusShift
            Cell_y4 = SubY[4, :] + self.MinusShift
            # IC5
            Cell_x5 = SubX[5, :]
            Cell_y5 = SubY[5, :] + self.MinusShift
            # IC6
            Cell_x6 = SubX[6, :] + self.MinusShift
            Cell_y6 = SubY[6, :] + self.MinusShift
            # IC7
            Cell_x7 = SubX[7, :] + self.MinusShift
            Cell_y7 = SubY[7, :]
            # IC8
            Cell_x8 = SubX[8, :] + self.MinusShift
            Cell_y8 = SubY[8, :] + self.PlusShift

            ShiftX = np.array([Cell_x0, Cell_x1, Cell_x2, Cell_x3, Cell_x4, Cell_x5,
                                Cell_x6, Cell_x7, Cell_x8])
            ShiftY = np.array([Cell_y0, Cell_y1, Cell_y2, Cell_y3, Cell_y4, Cell_y5,
                                Cell_y6, Cell_y7, Cell_y8])

            Dist = np.sqrt(ShiftX**2+ ShiftY**2)
        # bs에서 전파가 정규분포같이, power modeling
        # 68%는 잘 도달하는데 16%, 16%는 잘 도달하지

        # pt(sub) = 1, pr(bs) = pt/ dist**4 but 건물과 같은 장애물을 고려해야함
        # pr는 일정한 값이 아니라 정규분포를 띔
        # 채널을 모델링할 때 공간의 특징을 고려해야 함 -> 장애물 -> multipath
        # 모델링

            NormalDistribution = np.random.randn(self.FreqReUse, self.NoUpLink)
            mu = 0
            SD = self.sd
            LogNormal=mu+SD*NormalDistribution

            LogNormalP=10**(LogNormal/10)/(Dist**4)
            self.LogNormal = LogNormal
            self.LogNormalP = LogNormalP
            PS = LogNormalP[0, :]
            PI1 = LogNormalP[1, :]
            PI2 = LogNormalP[2, :]
            PI3 = LogNormalP[3, :]
            PI4 = LogNormalP[4, :]
            PI5 = LogNormalP[5, :]
            PI6 = LogNormalP[6, :]
            PI7 = LogNormalP[7, :]
            PI8 = LogNormalP[8, :]

            PI = PI1+PI2+PI3+PI4+PI5+PI6+PI7+PI8
            SIRn = PS/PI
            SIRdB = 10*np.log10(SIRn)

            SIR = np.vstack((SIR, SIRdB))

        SIR = np.delete(SIR, 0, 0)
        SIR = SIR.flatten()
        return SIR

    def graph(self):
        plt.figure(2)
        hist, bin_left, patch=plt.hist(self.SIR, bins=100, label='sd%s'%self.sd)
        # plt.text(0, 0, 'sd=%s'%self.sd)
        plt.legend()
        plt.grid()
        self.hist, self.bin_left = hist, bin_left
        pdf = hist/np.size(self.SIR)
        self.pdf =pdf
        plt.figure(3)
        plt.plot(bin_left[:-1], pdf,'o-', lw=2, label='sd%s'%self.sd)
        # plt.text('sd=%s'%self.sd)
        plt.legend()

        cdf = np.cumsum(pdf)
        self.cdf = cdf
        plt.figure(4)
        plt.semilogy(bin_left[:-1], cdf, 'o-',lw=2, label='sd%s'%self.sd)
        # plt.axis([0, 100, 10**-5, 10**0])
        plt.xlabel('Signal to Interference Ratio (SIR) [dB]')
        plt.ylabel('Probability of SIR (SIR < x)')
        plt.title('Cumulative Density Function of SIR')
        plt.text(20, 1e-3, r'Frequency Reuse Factor=9')
        plt.axis([-5, 50, 1e-4, 1])
        # plt.text(0, 0, 'sd=%s'%self.sd)
        plt.grid(True)
        plt.legend()
        # plt.show()

    # pdf
    def get_max_prob(self):
        max_y = max(self.pdf)
        x_arg = self.pdf.argmax()
        x_db = self.bin_left[x_arg]

        return {'max_y': max_y, 'x_arg':x_arg, 'x_db':x_db}

# 리스트 내에서 가장 비슷한 값을 찾음
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
# pdf, cdf 상에서 둘 다 찾을 수 있음
def find_deci(array_pdf, array_bin_left, value):
    x_arg = np.where(array_bin_left == value)
    prob = array_pdf[x_arg]
    return prob


sd4 = FTDD(4)
sd6 = FTDD(6)
sd10 = FTDD(10)

sd4.graph()
sd6.graph()
sd10.graph()
plt.show()