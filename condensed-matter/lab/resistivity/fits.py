import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from resistivity import *

# ==========   High Temperatures [100-290 k]

## Select the High Temperature and corresponding Resistances
Tlim = 230
k1 = 0
# Search the first index where T > Tlim

while T[k1] < Tlim :
    k1 += 1
#Select the part of the list corresponding to T > Tlim
T1, S1 = np.array(T[k1:]), np.array(conductivity[k1:])
# Invert T
T1 = 1/T1
# Log(sigma)
dS1 = np.array(Dconductivity[k1:])/ S1 # Uncertainty on log(Sigma)
S1 = np.log(S1)


# Fit the linear plot
def func(x, a, b) :
    return a*x + b

popt1, pcov1 = curve_fit(func, T1, S1, sigma = dS1) # Fit with errors
a, b, Da, Db = popt1[0], popt1[1], np.sqrt(abs(pcov1[0][0])), np.sqrt(abs(pcov1[1][1]))

print 'High Temperature Fitting Coefficients'
print 'Tlim = ' + str(Tlim) + ' K'
print 'a = ' + str(a)
print 'Da = ' + str(Da)
print 'b = ' + str(b)
print 'Db = ' + str(Db)
print '\n'

plt.plot(T1, S1, color = 'blue', label = 'Data')
plt.plot(T1, func(T1, a, b), color = 'red', label = 'Fit')
plt.fill_between(T1, S1 - dS1, S1 + dS1, color = 'blue', label = 'Uncertainty', alpha = 0.25)
plt.xlabel(r'1/T $[K^{-1}]$')
plt.ylabel(r'$\log(\sigma) = \log(\sigma_{0}) - \frac{\Delta \varepsilon}{k_{B}T}$')
plt.title(r'Fit of the logarithm conductivity at High Temperatures')
plt.legend()
plt.grid()
plt.show()

fich1 = open('Fit_High_T_values.dat', 'w')
fich1.write('invT' + '\t' + 'logS' + '\t' + 'Fit' + '\n')
for i in range(len(T1)) :
    fich1.write( str(T1[i]) + '\t' + str(S1[i]) + '\t' + str(func(T1[i], a, b)) + '\n')
fich1.close()

fich = open('Coefficients.dat', 'w')
fich.write('Coefficients for the different fits \n' + 
            'High Temperature coefficients :\n' + 
            'a = ' + str(a) + '\n' +
            'Da = ' + str(Da) + 'n' +
            'b = ' + str(b) + '\n' +
            'Db = ' + str(Db) + '\n')

# ============ Low Temperatures [< 100 K] - Mott formula
## Select the Low Temperatures and corresponding Resistances
Tlim = 10
k2 = 0
# Search the first index where T > Tlim
while T[k2] < Tlim :
    k2 += 1

#Select the part of the lists corresponding to T < Tlim
T2, S2 = np.array(T[:k2]), np.array(conductivity[:k2])
dS2 = np.array(Dconductivity[:k2])/ S2

#fich3 = open('Fit_Low_Matlab.txt', 'w')
#for i in range(len(T2)) :
#    fich3.write(str(T2[i]) + '\t' + str(S2[i]) + '\t' + str(dS2[i]) + '\n')
#fich3.close()

S2 = np.log(S2)
X = np.array([1/(T2[i])**(0.25) for i in range(len(T2))])

#Fit the function
def func2(x, c, d) :
    return c*x + d

popt2, pcov2 = curve_fit(func2, X, S2, sigma = dS2)
c, d, Dc, Dd = popt2[0], popt2[1], np.sqrt(abs(pcov2[0][0])), np.sqrt(abs(pcov2[1][1]))

print 'Low Temperature Fitting Coefficients - Mott'
print 'Tlim = ' + str(Tlim) + ' K'
print 'c = ' + str(c)
print 'Dc = ' + str(Dc)
print 'd = ' + str(d)
print 'Dd = ' + str(Dd)
print '\n'

plt.plot(X, S2, label = 'Data')
plt.plot(X, func2(X, c, d), label = 'Fit')
plt.fill_between(X, S2 - dS2, S2 + dS2, color = 'blue', alpha = '0.25')
plt.xlabel(r'$1/T^{0.25}$  $[K^{-0.25}]$')
plt.ylabel(r'$\log(\sigma) = \log(\sigma_{0}) - (T_{0})^{0.25} * (1/T)^{0.25}$')
plt.legend()
plt.grid()
plt.title('Fit of the conductivity at Low Temperature (Mott Formula)')
plt.show()

fich2 = open('Fit_Low_T_values_Mott.dat', 'w')
fich2.write('invT_0.25' + '\t' + 'logS' + '\t' + 'Fit' + '\n')
for i in range(len(T2)) :
    fich2.write(str(T2[i]) + '\t' + str(S2[i]) + '\t' + str(func2(X[i], c, d)) + '\n')
fich2.close()

fich.write('\nLow Temperature coefficients Mott :\n' + 
            'c = ' + str(c) + '\n' +
            'Dc = ' + str(Dc) + 'n' +
            'd = ' + str(d) + '\n' +
            'Dd = ' + str(Dd) + '\n')


# ============ Low Temperatures [< 100 K] - Klein Formula
## Select the Low Temperatures and corresponding Resistances
Tlim = 30
k3 = 0
# Search the first index where T > Tlim
while T[k3] < Tlim :
    k3 += 1

#Select the part of the lists corresponding to T < Tlim
T3, S3 = np.array(T[:k3]), np.array(conductivity[:k3])
dS3 = np.array(Dconductivity[:k3])/ S3

def func3(x, e, f, g) :
    return e + g*np.sqrt(x) + f*x

popt3, pcov3 = curve_fit(func3, T3, S3)
e, f, g, De, Df, Dg = popt3[0], popt3[1], popt3[2], np.sqrt(abs(pcov3[0][0])), np.sqrt(abs(pcov3[1][1])), np.sqrt(abs(pcov3[2][2]))

print 'Low Temperature Fitting Coefficients - Klein'
print 'Tlim = ' + str(Tlim) + ' K'
print 'e = ' + str(e)
print 'De = ' + str(De)
print 'f = ' + str(f)
print 'Df = ' + str(Df)
print 'g = ' + str(g)
print 'Dg = ' + str(Dg)
print '\n'

fich.write('\nLow Temperature coefficients Klein :\n' + 
            'e = ' + str(e) + '\n' +
            'De = ' + str(De) + 'n' +
            'f = ' + str(f) + '\n' +
            'Df = ' + str(Df) + '\n' +
            'g = ' + str(g) + '\n' +
            'Dg = ' + str(Dg) + '\n')
fich.close()

plt.plot(T3, S3, label = 'Data')
plt.plot(T3, func3(T3, e, f, g), label = 'Fit')
plt.fill_between(T3, S3 - dS3, S3 + dS3, color = 'blue', alpha = '0.25')
plt.xlabel(r'$T$  $[K]$')
plt.ylabel(r'$\sigma = \sigma_{0} + A.T^{0.5} + B.T$')
plt.legend()
plt.grid()
plt.title('Fit of the conductivity at Low Temperature (Klein Formula)')
plt.show()

fich3 = open('Fit_Low_T_values_Klein.dat', 'w')
fich3.write('T' + '\t' + 'S' + '\t' + 'Fit' + '\n')
for i in range(len(T3)) :
    fich3.write(str(T3[i]) + '\t' + str(S3[i]) + '\t' + str(func3(T3[i], e, f, g)) + '\n')
fich3.close()
