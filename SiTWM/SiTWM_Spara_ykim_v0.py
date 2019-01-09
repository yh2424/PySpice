#r# This example shows a low-pass RC Filter.

####################################################################################################

import math
import numpy as np
import matplotlib.pyplot as plt

####################################################################################################

import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()

####################################################################################################

from PySpice.Plot.BodeDiagram import bode_diagram
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

####################################################################################################

#f# circuit_macros('low-pass-rc-filter.m4')

circuit = Circuit('Low-Pass RC Filter')

circuit.SinusoidalVoltageSource('input', 'in', circuit.gnd, amplitude=1@u_V)

Rs0 = 50
Rt0 = 25

circuit.R('s', 'in', 1, Rs0@u_Ω)

n = 6
for i in range(1, n+1):
    # Transmission line (Passive segment)
    circuit.R(i, (4 * i - 3), (4 * i), 1.65@u_Ω)
    circuit.L(i, (4 * i), (4 * i + 1), 1.27e-10@u_H)
    circuit.C(i, (4 * i + 1), circuit.gnd, 2.24e-14@u_F)
    circuit.R(i*10, (4 * i + 1), circuit.gnd, 8e3@u_Ω)

    # Phase shifter (Active segment)
    Rs = 60  # 60
    Cj = 82e-15  # 82e-15

    circuit.R(i*100, (4 * i + 1), (4 * i + 3), Rs@u_Ω)
    circuit.C(i*10, (4 * i + 3), circuit.gnd, Cj@u_F)
    # mycircuit.add_resistor("Rsub%s" % (i), n1="n%s" %(4*i+1), n2="n%s" %(4*i+2), value=20.0)
    # mycircuit.add_capacitor("Cox%s" % (i), n1="n%s" %(4*i+2), n2=gnd, value=4.09e-14)


circuit.R('t', (4 * i + 1), circuit.gnd, Rt0@u_Ω )


simulator = circuit.simulator(temperature=25, nominal_temperature=25)
analysis = simulator.ac(start_frequency=1e8@u_Hz, stop_frequency=5e10@u_Hz, number_of_points=100,  variation='dec')


#f# save_figure('figure', 'low-pass-rc-filter-bode-diagram.png')



V1 = np.array(analysis['1'])
I1 = np.array((analysis['in'] - analysis['1'])/Rs0)
V2 = np.array(analysis[str(4 * i + 1)])
I2 = np.array((analysis[str(4 * i + 1)])/Rt0)

#Z0 = np.array((V1-V2)/I1)
# Z0 = np.array((V1)/I1)
# Z0 = 50
Z01 = Rs0
Z02 = Rt0

#Ref: Dean A. Frickey, "Conversions between S, Z, Y, h, ABCD and T parameters which are valid for complex source and load impedances" ,IEEE Trans. Microwave thieory and techniques, vol42, No2, 1994
a1 = (V1 + I1*Z01)/(2*np.sqrt(Z01))
b1 = (V1 - I1*Z01)/(2*np.sqrt(Z01))
# a2 = (V2 - I2*Z02)/(2*np.sqrt(Z02))
b2 = (V2 + I2*Z02)/(2*np.sqrt(Z02))

S11 = b1/a1
S21 = b2/a1

fig = plt.figure(figsize=(6,12))
plt.subplot(211)
plt.plot(analysis.frequency, 20*np.log10(np.absolute(S21)), 'o-')
plt.grid(True)
plt.ylabel('S21 [dB]')
# plt.title(mycircuit.title + " - AC Simulation")

plt.plot(analysis.frequency, 20*np.log10(np.absolute(S11)), 'o-')
plt.grid(True)
plt.ylabel('S parameters [dB]')

plt.subplot(212)
plt.plot(analysis.frequency, np.absolute((V1-V2)/I1), 'o-')
plt.grid(True)
plt.ylabel('Z0 [Ohm]')
plt.ylim(0,100)


plt.xlabel('Frequency [Hz]')
fig.savefig('ac_plot.png')
plt.show()