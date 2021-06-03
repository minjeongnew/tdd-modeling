
import numpy as np
import matplotlib.pyplot as plt
# 1 tier
FreqReUse=9
NoUpLink=12
NtwSizeA=-2000
NtwSizeB=2000
PlusShift=12000
MinusShift=-12000
No_Iterations=10000


SIR=np.zeros((1,NoUpLink))

for Loop in range(0,No_Iterations):
    # user 데이터 12 명
    SubX = np.random.uniform(NtwSizeA, NtwSizeB, size=[FreqReUse, NoUpLink])
    SubY = np.random.uniform(NtwSizeA, NtwSizeB, size=[FreqReUse, NoUpLink])
    # wanted signal 발생하는 공간
    Cell_x0 = SubX[0, :]
    Cell_y0 = SubY[0, :]
    # IC1
    Cell_x1 = SubX[1, :]
    Cell_y1 = SubY[1, :] + PlusShift
    # IC2
    Cell_x2 = SubX[2, :] + PlusShift
    Cell_y2 = SubY[2, :] + PlusShift
    # IC3
    Cell_x3 = SubX[3, :] + PlusShift
    Cell_y3 = SubY[3, :]
    # IC4
    Cell_x4 = SubX[4, :] + PlusShift
    Cell_y4 = SubY[4, :] + MinusShift
    # IC5
    Cell_x5 = SubX[5, :]
    Cell_y5 = SubY[5, :] + MinusShift
    # IC6
    Cell_x6 = SubX[6, :] + MinusShift
    Cell_y6 = SubY[6, :] + MinusShift
    # IC7
    Cell_x7 = SubX[7, :] + MinusShift
    Cell_y7 = SubY[7, :]
    # IC8
    Cell_x8 = SubX[8, :] + MinusShift
    Cell_y8 = SubY[8, :] + PlusShift

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
    NormalDistribution = np.random.randn(FreqReUse, NoUpLink)
    mu = 0
    SD = 6
    LogNormal=mu+SD*NormalDistribution

    LogNormalP=10**(LogNormal/10)/(Dist**4)

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

SIR=np.delete(SIR, 0,0)
SIR=SIR.flatten()


plt.figure(2)
hist, bin_left, patch=plt.hist(SIR, bins=100)
plt.grid()


pdf=hist/np.size(SIR)
plt.figure(3)
plt.plot(bin_left[:-1], pdf, 'ro-', lw=2)


cdf=np.cumsum(pdf)
plt.figure(4)
plt.semilogy(bin_left[:-1], cdf, color='c', lw=2)
#plt.axis([0, 100, 10**-5, 10**0])
plt.xlabel('Signal to Interference Ratio (SIR) [dB]')
plt.ylabel('Probability of SIR (SIR < x)')
plt.title('Cumulative Density Function of SIR')
plt.text(20, 1e-3, r'Frequency Reuse Factor=9')
plt.axis([-5, 50, 1e-4, 1])
plt.grid(True)
plt.show()