import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

#===================== IMPORT

# Import data from files
h = np.loadtxt('6102017_heating_to_290K_Ida_Vcb_250uA.dat', delimiter = '\t')
c = np.loadtxt('692017_cooling_to_5K_Idc_Vab_250uA.dat', delimiter = '\t')

# Create lists
TB_H, R_H, VHm, TB_C, R_C, VCm = [], [], [], [], [], []

for i in range(len(h)) :
    TB_H += [h[i][4]]
    R_H += [h[i][7]]
    VHm += [h[i][8]]
    
for k in range(len(c)) :
    TB_C += [c[k][4]]
    R_C += [c[k][7]]
    VCm += [c[k][8]]

# ============ AVERAGE SECTION ==============

# Function that average the Resistance over a window dT at temperature T0
def avg(T, R, T0, dT):
    L = []
    for i in range(len(T)):
        if abs(T[i] - T0) <= dT :
            L += [R[i]]
    return sum(L)/len(L), min(L), max(L)

# Settings
DeltaT = [0.2, 0.3] # Temperature step (in K)
#dT = DeltaT*0.5
T_min, T_max = 4.7, 290.
T_start = T_min
#n =  #int((T_max - T_min)/DeltaT) # Number of steps, depending on the range and the step width

# First step
T, RH, RHmin, RHmax, DRH, RC, RCmin, RCmax, DRC = [], [], [], [], [], [], [], [], []
#s1, s2, s3 = avg(TB_H, R_H, T_start, dT)
#RH, RHmin, RHmax, DRH = [s1], [s2], [s3], [s3 - s2]
#s4, s5, s6 = avg(TB_C, R_C, T_start, dT)
#RC, RCmin, RCmax, DRC = [s4], [s5], [s6], [s6 - s5]

# Iterating
t0=T_start
while t0 < T_max :
    if t0 < 30 :
        Tstep = DeltaT[0]
    else :
        Tstep = DeltaT[1]
    dT = 0.5*Tstep
    t0 += Tstep                     # next temperature
    T += [t0]                   # Add t to the temperature list
    s1, s2, s3 = avg(TB_H, R_H, t0, dT)
    RH.append(s1) ; RHmin.append(s2) ; RHmax.append(s3) ; DRH.append(s3-s2)
    s4, s5, s6 = avg(TB_C, R_C, t0, dT)
    RC.append(s4) ; RCmin.append(s5) ; RCmax.append(s6) ; DRC.append(s6-s5)

# Save in files
fich1 = open('Resistances.dat', 'w')
fich1.write('TH \t RH \t DRH \t RC \t DRC \n')
for i in range(len(T)) :
    fich1.write(str(T[i]) + '\t' +
                str(RH[i]) + '\t' + str(RHmax[i] - RHmin[i]) + '\t' +
                str(RC[i]) + '\t' + str(RCmax[i] - RCmin[i]) +' \n')
fich1.close()

# ============= VAN DER PAUW ALGORITHM
delta = 0.0005 # Precision

def algorithm(delta, Ra, Rb) :
    z0 = 2*np.log(2) / (np.pi * (Ra + Rb))
    a = 1.
    while a > delta :
        y = 1/np.exp(np.pi * z0 * Ra) + 1/np.exp(np.pi * z0 * Rb)
        z = z0 - ( (1-y)/np.pi ) / (Ra / np.exp(np.pi * z0 * Ra)
                                    + Rb / np.exp(np.pi * z0 * Rb) )
        a = (z - z0)/z
        z0 = z
    return z

resistivity, conductivity = [], [] #Resistivity and conductivity lists
d = 442.e-9 #Thickness of the sample in METERS !!

# Apply the algorithm for all considered temperatures
for k in range(len(T)) :
    resistivity += [d/algorithm(delta, RC[k], RH[k])]


resistivity = 100*np.array(resistivity) # The Factor 100 is to convert from Ohms.m to Ohms.cm
conductivity = 1./resistivity

# ================= Computation of the uncertainties

### Thickness
Dd = 44.e-9
DL_L = Dd / d # This is the relative error on the thickness

### Contacts
DC_C = 2*(1.446e-3) + 2.040e-3 + 1.411e-3 # Total relative error on the contacts

### Resistances
def Ki(Ri, d, res):
    return np.pi*Ri*d*np.exp(-np.pi*d*Ri/res)/res

# This function computes the relative error on resistivity at a fixed T
# Inputs :  r(T) [resistivity at T], R1(T), R2(T) and the errors on resistances DR1(T), DR2(T)
def Drho_rho(r, R1, DR1, R2, DR2):
    global DL_L, DC_C, d
    K1, K2 = Ki(R1, d, r), Ki(R2, d, r)
    return (K1/(K1+K2) * DR1/R1) + (K2/(K1 + K2) * DR2/R2) + DC_C + DL_L

DRHO_RHO = [Drho_rho(resistivity[i], RH[i], DRH[i], RC[i], DRC[i]) for i in range(len(T))]
Dresistivity = [resistivity[i]*DRHO_RHO[i] for i in range(len(T))]
Dconductivity = [conductivity[i]*DRHO_RHO[i] for i in range(len(T))]

def Drho_rho2(r, R1, DR1, R2, DR2):
    global DL_L, DC_C, d
    K1, K2 = Ki(R1, d, r), Ki(R2, d, r)
    return (K1/(K1+K2) * DR1/R1) + (K2/(K1 + K2) * DR2/R2) + DC_C# + DL_L

DRHO_RHO2 = [Drho_rho2(resistivity[i], RH[i], DRH[i], RC[i], DRC[i]) for i in range(len(T))]
Dresistivity2 = [resistivity[i]*DRHO_RHO2[i] for i in range(len(T))]

# Compute res_min and res_max.
res_min, res_max = [], []
for k in range(len(T)) :
    res_min += [d/algorithm(delta, RCmin[k], RHmin[k])]
    res_max += [d/algorithm(delta, RCmax[k], RHmax[k])]
res_min, res_max = 100*np.array(res_min), 100*np.array(res_max)

# Plot the resistivity (with errors)
plt.plot(T, resistivity)
plt.fill_between(T, resistivity - Dresistivity, resistivity + Dresistivity, alpha = 0.25)
plt.xlabel('Temperature [K]')
plt.ylabel('Resistivity'+ r'$[\Omega .cm]$')
plt.title('Resistivity vs Temperature in a Diamond sample')
plt.grid()
plt.show()

# Plot the conductivity (with errors)
plt.plot(T, conductivity)
plt.fill_between(T, conductivity - Dconductivity, conductivity + Dconductivity, alpha = 0.25)
plt.xlabel('Temperature [K]')
plt.ylabel('Conductivity'+ r'$[(\Omega .cm)^{-1}]$')
plt.title('Conductivity vs Temperature in a Diamond sample')
plt.grid()
plt.show()


plt.plot(T, resistivity)
plt.fill_between(T, resistivity - Dresistivity, resistivity + Dresistivity, alpha = 0.25, color = 'blue', label = 'Real')
plt.fill_between(T, resistivity - Dresistivity2, resistivity + Dresistivity2, alpha = 0.25, color = 'green', label = 'Without Thickness')
plt.fill_between(T, res_min, res_max, alpha = 0.25, color = 'red', label = 'Enveloppe')
plt.xlabel('Temperature [K]')
plt.ylabel('Resistivity'+ r'$[\Omega .cm]$')
plt.title('Resistivity vs Temperature in a Diamond sample')
plt.legend()
plt.grid()
plt.show()

fich2 = open('Resistivity.dat', 'w')
fich2.write('T \t r \t dr \n')
for i in range(len(T)) :
    fich2.write(str(T[i]) + '\t' + str(resistivity[i]) + '\t' + str(Dresistivity[i]) + '\n' )
fich2.close()

fich3 = open('Conductivity.dat', 'w')
fich3.write('T \t c \t dc \n')
for i in range(len(T)) :
    fich3.write(str(T[i]) + '\t' + str(conductivity[i]) + '\t' + str(Dconductivity[i]) + '\n' )
fich3.close()

plt.plot(T, resistivity)
#plt.fill_between(T, resistivity - Dresistivity, resistivity + Dresistivity, alpha = 0.25, color = 'blue', label = 'Real')
plt.fill_between(T, resistivity - Dresistivity2, resistivity + Dresistivity2, alpha = 0.25, color = 'green', label = 'Without Thickness')
plt.fill_between(T, res_min, res_max, alpha = 0.25, color = 'red', label = 'Enveloppe')
plt.xlabel('Temperature [K]')
plt.ylabel('Resistivity'+ r'$[\Omega .cm]$')
plt.title('Resistivity vs Temperature in a Diamond sample')
plt.legend()
plt.grid()
plt.show()
