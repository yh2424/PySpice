import numpy as np
from StringIO import StringIO
import time
import os
from util.restrictedclass import FrozenClass, Property
from util.mathtools import BandWidth
import matplotlib.pyplot as plt
import scipy.optimize
from lxml import objectify as xmlobj

'''This module contains a number generic methods that can be used in
each of the analysis scripts. This is copy version of generic.py? or MachZehnder.py?
'''

class AnalysisResult(FrozenClass):
    '''Base class for basic analysis results.
    Attributes:
    -LaserPower (float): power (dBm) of the laser source during the 
        light current measurement
    -FiberOutput (float): optical power at the tip of the input fiber (dBm)
        during light current measurement
    -Date (str): string representation of the date when this measurement
        was performed.
    -Time (str): string representation of the time when this measurement
        was performed.
    '''
    def __init__(self, **kwargs):
        '''Initialise some attributes of this object.'''
        self._laserpower = 0.0
        self._fiberoutput = 0.0
        self._date = ''
        self._time = ''
        super(AnalysisResult, self).__init__(**kwargs)
    
    def __repr__(self):
        ''' Function for debugging help: prints out an overview of the object's 
        property values. '''
        return "%s(%s, %s)" % (self.__class__, 
                                       self.Date, self.Time)

    def __str__(self):
        ''' Function for debugging help: prints out an overview of the object's 
        property values. '''
        return self.__repr__()
    
    @Property
    def LaserPower(self):
        doc = ("LaserPower (float): power (dBm) of the laser source during ",
               "the light current measurement.")
        def fget(self):
            return FrozenClass.fget_template(self, 'LaserPower')
        def fset(self, dbms):
            FrozenClass.fset_template(self, dbms, 'LaserPower', 
                                      [float, int, np.float64])
        return locals()
    
    @Property
    def FiberOutput(self):
        doc = ("FiberOutput (float): optical power at the tip of the input ",
               "fiber (dBm) during light current measurement.")
        def fget(self):
            return FrozenClass.fget_template(self, 'FiberOutput')
        def fset(self, dbms):
            FrozenClass.fset_template(self, dbms, 'FiberOutput', 
                                      [float, int, np.float64])
        return locals()
    
    @Property
    def Date(self):
        doc = ("Date (str): string representation of the date when this ",
               "measurement was performed.")
        def fget(self):
            return FrozenClass.fget_template(self, 'Date')
        def fset(self, date_str):
            FrozenClass.fset_template(self, date_str, 'Date', str)
        return locals()
    
    @Property
    def Time(self):
        doc = ("Time (str): string representation of the time when this ",
               "measurement was performed.")
        def fget(self):
            return FrozenClass.fget_template(self, 'Time')
        def fset(self, time_str):
            FrozenClass.fset_template(self, time_str, 'Time', str)
        return locals()

def AnalyseWavelengthSweep(data, inputpower=6, outputloss=4, 
                           voltage_sweep_idx=None):
    '''Analyse the wavelength sweep measurement on a single waveguide 
    and return:
        -peak wavelength
        -2dB bandwidth
        -6dB bandwidth
        -insertion loss at peak wavelength
        -temperature at which measurement was performed
    Arguments:
    -data: the PortLossMeasurement node, read from XML using lxml.objectify
    -inputpower (float): the optical power at the tip of the input fiber (dBm)
    -outputloss (float): the optical loss between the tip of the output fiber
        and the power meter (dB)
    -voltage_sweep_idx (None, int): if 'data' contains wavelength sweep data
        measured at various bias voltages (e.g. in a demux measurement), then
        this argument will select one specific index in the bias voltage sweep.
        If 'None' is specified, 'data' must contain only one wavelength sweep.
    
    Returns:
    -result (WavelengthSweepMeasurementResult): object that stores all the
        analysis results.
    '''
    #figure out which type or result we're dealing with: measurement of 
    #transmitted power (e.g. with Agilent 8164B mainframe) or measurement of
    #insertion loss (e.g. with CT400 after calibration)
    if hasattr(data, 'VoltageSweep'):
        #the data node contains data measured at various bias voltages
        if voltage_sweep_idx is None:
            voltage_sweep_idx = 0
        if hasattr(data.VoltageSweep[0].MeasurementResult, 'P'):
            mt = 'P'
        elif hasattr(data.VoltageSweep[0].MeasurementResult, 'IL'):
            mt = 'IL'
        else:
            raise Exception("No node named 'P' or 'IL' found!")
    else:
        #the node contains just one wavelength sweep
        if not voltage_sweep_idx in [None, 0]:
            raise Exception("Data node does not contain a bias voltage sweep!")
        if hasattr(data.MeasurementResult, 'P'):
            mt = 'P'
        elif hasattr(data.MeasurementResult, 'IL'):
            mt = 'IL'
        else:
            raise Exception("No node named 'P' or 'IL' found!")
    result = WavelengthSweepAnalysisResult()
    if not voltage_sweep_idx is None:
        L = data.VoltageSweep[voltage_sweep_idx].MeasurementResult.L.pyval
    else:
        L = data.MeasurementResult.L.pyval
    L = np.genfromtxt(StringIO(L), delimiter=',')
    if mt == 'P':
        if not voltage_sweep_idx is None:
            P = data.VoltageSweep[voltage_sweep_idx].MeasurementResult.P.pyval
        else:
            P = data.MeasurementResult.P.pyval
        P = np.genfromtxt(StringIO(P), delimiter=',')
    elif mt == 'IL':
        if not voltage_sweep_idx is None:
            P = data.VoltageSweep[voltage_sweep_idx].MeasurementResult.IL.pyval
        else:
            P = data.MeasurementResult.IL.pyval
        P = np.genfromtxt(StringIO(P), delimiter=',')
    #xm, ym, pol, imin, imax = FindPolyFittedMax(L, P)
    #peak_wl = xm
    #ins_loss = inputpower-ym-outputloss
    bwdata = BandWidth.BandWidth(L, P, bws=[2,6],  plot=False,
                                 fitorder=5, fitrange=0.75)
    result.PeakWL = bwdata.x_pk
    if not mt == 'IL':
        result.InsLoss = inputpower-bwdata.y_pk-outputloss
    else:
        result.InsLoss = bwdata.y_pk
    result.TwodBLo = bwdata.x_lo[0]
    result.TwodBHi = bwdata.x_hi[0]
    result.TwodBBW = result.TwodBHi - result.TwodBLo
    result.SixdBLo = bwdata.x_lo[1]
    result.SixdBHi = bwdata.x_hi[1]
    result.SixdBBW = result.SixdBHi - result.SixdBLo
    result.Poly = bwdata.Poly
    datetime = time.strptime(data.attrib['DateStamp'], "%a %b %d %X %Y")
    result.Date = time.strftime("%Y%m%d", datetime)
    result.Time = time.strftime("%X", datetime)
    
    result.LaserPower = float(data.MeasurementResult.attrib['Power'])
    powerunit = data.MeasurementResult.attrib['PowerUnit']
    #if 'dbm' in powerunit.lower():
    #    #convert to mW
    #    result.LaserPower = 10**(result.LaserPower/10)
    result.FiberOutput = inputpower
    
    if 'Temperature' in data.attrib.keys():
        #value written to xml file is the voltage for a 10uA current in
        #the 1kOhm PTC thermistor:
        PT1000_R = np.float(data.attrib['Temperature'])*100
        #we need to convert this resitance to a temperature value.
        #equation: R[kOhm] = 1+A*T+B*T^2 with A=3.9083 x 10 -3,
        #                                     B=-5.775 x 10 -7 
        #See: http://www.ist-ag.com/eh/ist-ag/resource.nsf/imgref/
        #Download_ATP_E2.1.3.compressed.pdf/$FILE/ATP_E2.1.3.compressed.pdf
        rts = np.polynomial.polyroots([1-PT1000_R, 3.9083e-3, 5.775e-7])
        result.Temperature = max(rts)
    
    return result

def PortLossMeasurementToCSV(pn, fn, voltage_sweep_idx=None):
    '''Store the wavelength and power array from a PortLossMeasurement node to
    a csv file.
    Args:
    -pn (XML node): PortLossMeasurement node, read from an XML file.
    -fn (str): name of the CSV file that is to be written.
    -voltage_sweep_idx (None, int): if 'pn' contains wavelength sweep data
        measured at various bias voltages (e.g. in a demux measurement), then
        this argument will select one specific index in the bias voltage sweep.
        If 'None' is specified, 'data' must contain only one wavelength sweep.
    '''
    if not voltage_sweep_idx is None:
        L = pn.VoltageSweep[voltage_sweep_idx].MeasurementResult.L.pyval
    else:
        L = pn.MeasurementResult.L.pyval
    L = np.genfromtxt(StringIO(L), delimiter=',')
    if not voltage_sweep_idx is None:
        P = pn.VoltageSweep[voltage_sweep_idx].MeasurementResult.P.pyval
    else:
        P = pn.MeasurementResult.P.pyval
    P = np.genfromtxt(StringIO(P), delimiter=',')
    a = np.transpose(np.vstack([L, P]))
    np.savetxt(fn, a, delimiter=",")

class WavelengthSweepAnalysisResult(AnalysisResult):
    '''This class represents an analysis result, obtained by running
    a AnalyseWavelengthSweepMeasurement analysis.
    
    Attributes:
    -PeakWL (float): peak wavelength in nanometer
    -InsLoss (float): insertion loss at peak wavelength
    -TwodBLo (float): lower boundary of the 2dB bandwidth (in nm)
    -TwodBHi (float): upper boundary of the 2dB bandwidth (in nm)
    -TwodBBW (float): the 2dB bandwidth (in nm)
    -SixdBLo (float): lower boundary of the 6dB bandwidth (in nm)
    -SixdBHi (float): upper boundary of the 6dB bandwidth (in nm)
    -SixdBBW (float): the 6dB bandwidth (in nm)
    -Length (float): length of this waveguide (in mm)
    -Temperature (float): ambient temperature during measurement (deg C)
    -Poly (list, np.ndarray): list with coefficients of a fitted polynomial
    '''
    def __init__(self, **kwargs):
        '''Initialise some attributes of this object.'''
        self._peakwl = 1550.0
        self._insloss = 0.0
        self._twodblo = 1550.0
        self._twodbhi = 1550.0
        self._twodbbw = 0.0
        self._sixdblo = 1550.0
        self._sixdbhi = 1550.0
        self._sixdbbw = 0.0 
        self._length = 0.0
        self._temperature = 0.0
        self._poly = []
        super(WavelengthSweepAnalysisResult, self).__init__(**kwargs)
    
    def __repr__(self):
        ''' Function for debugging help: prints out an overview of the object's 
        property values. '''
        return "%s(%0.5g_%0.3g)" % (self.__class__, self.PeakWL, self.InsLoss)

    def __str__(self):
        ''' Function for debugging help: prints out an overview of the object's 
        property values. '''
        return self.__repr__()
        
    @Property
    def PeakWL(self):
        doc = ("PeakWL (float): peak wavelength in nanometer.")
        def fget(self):
            return FrozenClass.fget_template(self, 'PeakWL')
        def fset(self, wavelength_nm):
            FrozenClass.fset_template(self, wavelength_nm, 'PeakWL', 
                                      [float, int, np.float64])
        return locals()
    
    @Property
    def InsLoss(self):
        doc = ("InsLoss (float): insertion loss at peak wavelength.")
        def fget(self):
            return FrozenClass.fget_template(self, 'InsLoss')
        def fset(self, loss_db):
            FrozenClass.fset_template(self, loss_db, 'InsLoss', 
                                      [float, int, np.float64])
        return locals()
    
    @Property
    def TwodBLo(self):
        doc = ("TwodBLo (float): lower boundary of the 2dB bandwidth (in nm).")
        def fget(self):
            return FrozenClass.fget_template(self, 'TwodBLo')
        def fset(self, wavelength_nm):
            FrozenClass.fset_template(self, wavelength_nm, 'TwodBLo', 
                                      [float, int, np.float64])
        return locals()
    
    @Property
    def TwodBHi(self):
        doc = ("TwodBHi (float): upper boundary of the 2dB bandwidth (in nm).")
        def fget(self):
            return FrozenClass.fget_template(self, 'TwodBHi')
        def fset(self, wavelength_nm):
            FrozenClass.fset_template(self, wavelength_nm, 'TwodBHi', 
                                      [float, int, np.float64])
        return locals()
    
    @Property
    def TwodBBW(self):
        doc = ("TwodBBW (float): the 2dB bandwidth (in nm).")
        def fget(self):
            return FrozenClass.fget_template(self, 'TwodBBW')
        def fset(self, wavelength_nm):
            FrozenClass.fset_template(self, wavelength_nm, 'TwodBBW', 
                                      [float, int, np.float64])
        return locals()
    
    @Property
    def SixdBLo(self):
        doc = ("SixdBLo (float): lower boundary of the 2dB bandwidth (in nm).")
        def fget(self):
            return FrozenClass.fget_template(self, 'SixdBLo')
        def fset(self, wavelength_nm):
            FrozenClass.fset_template(self, wavelength_nm, 'SixdBLo', 
                                      [float, int, np.float64])
        return locals()
    
    @Property
    def SixdBHi(self):
        doc = ("SixdBHi (float): upper boundary of the 2dB bandwidth (in nm).")
        def fget(self):
            return FrozenClass.fget_template(self, 'SixdBHi')
        def fset(self, wavelength_nm):
            FrozenClass.fset_template(self, wavelength_nm, 'SixdBHi', 
                                      [float, int, np.float64])
        return locals()
    
    @Property
    def SixdBBW(self):
        doc = ("SixdBBW (float): the 2dB bandwidth (in nm).")
        def fget(self):
            return FrozenClass.fget_template(self, 'SixdBBW')
        def fset(self, wavelength_nm):
            FrozenClass.fset_template(self, wavelength_nm, 'SixdBBW', 
                                      [float, int, np.float64])
        return locals()
    
    @Property
    def Length(self):
        doc = ("Length (float): length of this waveguide (in mm).")
        def fget(self):
            return FrozenClass.fget_template(self, 'Length')
        def fset(self, length_mm):
            FrozenClass.fset_template(self, length_mm, 'Length', 
                                      [float, int, np.float64])
        return locals()
    
    @Property
    def Temperature(self):
        doc = ("Temperature (float): ambient temperature during measurement ",
               "(deg C)")
        def fget(self):
            return FrozenClass.fget_template(self, 'Temperature')
        def fset(self, temp_degc):
            FrozenClass.fset_template(self, temp_degc, 'Temperature', 
                                      [float, int, np.float64])
        return locals()
    
    @Property
    def Poly(self):
        doc = ("Poly (list, np.ndarray): list with coefficients of a fitted ",
               "polynomial")
        def fget(self):
            return FrozenClass.fget_template(self, 'Poly')
        def fset(self, poly_coeffs):
            FrozenClass.fset_template(self, poly_coeffs, 'Poly', 
                                      [list, np.ndarray],
                                      el_dtype=[float, int, np.float64,
                                                np.int32])
        return locals()

def AnalyseMZI(data, Z0=50, figname=None, result=None):
    '''Analyse the S11 parameter measured on a Mach Zehnder 
    interferometer at a given bias voltage.
    
    Args:
    -data: the SParamMeasurement node, read from XML using lxml.objectify
    -Z0 (float): characteristic impedance of the waveguide/cable/NWA ports. 
        Default is 50Ohm.
    -figname (str): name of a file where a plot of the S param measurement will
        be saved. Default is None, in which case no plot will be made.
    -result (MZIAnalysisResult): an MZIAnalysisResult object (which could have 
        some values like name, length, guess, max_freq already filled in in
        order to help improve the accuracy of the analysis below). Default
        value is None, in which case some default values for the 'guess' will
        be used.
    
    Returns:
    -result (MZIAnalysisResult): object with attributes that store
        results from MZI analyses.
    '''
    if result is None:
        result = MZIAnalysisResult()
        result.guess = [1000, 50e-15, 25e-15, 0.08e-9, 10, 150e-15]
    swp = data.MeasurementResult.BiasAndWavelengthSweepResult
    freq = swp.F.pyval
    freq = np.genfromtxt(StringIO(freq), delimiter=',')
    if not result.max_freq is None:
        maxind = np.argmin(np.abs(freq-result.max_freq))
        freqs11 = freq[0:maxind]
    else:
        freqs11 = freq
    if (not figname is None):
        _, axs11 = plt.subplots()
        _, axs21 = plt.subplots()
        s11found = False
        s21found = False
    for sparam in swp.SParameters:
        bias = float(sparam.attrib['DCBias'])
        result.Bias.append(bias)
        datetime = time.strptime(data.MeasurementResult.attrib['DateStamp'], 
                             "%a %b %d %X %Y")
        result.Date = time.strftime("%Y%m%d", datetime)
        result.Time = time.strftime("%X", datetime)
        result.Wavelength = float(sparam.attrib['Wavelength'])
        result.LaserPower = float(sparam.attrib['OpticalPower'])
        for spm in sparam.SP:
            if spm.attrib['Name'] == 'S11':
                s11found = True
                s11_real = spm.real.pyval
                s11_real = np.genfromtxt(StringIO(s11_real), delimiter=',')
                s11_imag = spm.imag.pyval
                s11_imag = np.genfromtxt(StringIO(s11_imag), delimiter=',')
                s11 = s11_real+1j*s11_imag      
                if not result.max_freq is None:
                    s11 = s11[0:maxind]
                z11 = -Z0*(s11+1)/(s11-1)
                y11 = 1/z11
                #process s11 data:
                __mzis11__(freqs11, y11, bias, axs11, result, 
                           figname=figname)
            elif spm.attrib['Name'] == 'S41':
                s21found = True
                s41_real = spm.real.pyval
                s41_real = np.genfromtxt(StringIO(s41_real), delimiter=',')
                s41_imag = spm.imag.pyval
                s41_imag = np.genfromtxt(StringIO(s41_imag), delimiter=',')
                s41 = s41_real+1j*s41_imag      
                #if not max_freq is None:
                #    s41 = s41[0:maxind]
                #z41 = -Z0*(s41+1)/(s41-1)
                #y41 = 1/z41
                #process s11 data:
                __mzis21__(freq, s41, bias, axs21, result, figname=figname)
    showorsave = 'save'
    if (not figname is None):
        plt.sca(axs11)
        axs11.grid(True)
        handles, labels = axs11.get_legend_handles_labels()
        axs11.legend(handles, labels, loc='lower right')
        if (showorsave == 'save') and (s11found == True):
            plt.savefig(figname + '_S11.png', bbox_inches='tight')
        elif (s11found == True):
            plt.show()
        plt.sca(axs21)
        axs21.grid(True)
        handles, labels = axs21.get_legend_handles_labels()
        axs21.legend(handles, labels, loc='lower right')
        if (showorsave == 'save') and (s21found == True):
            plt.savefig(figname + '_S21.png', bbox_inches='tight')
        elif (s21found == True):
            plt.show()
    return result

def __mzis11__(freq, y11, bias, ax, result, figname=None):
    '''Analyse S11 measured on a Mach Zehnder device.
    
    Args:
    -freq (np.ndarray): a numpy array with frequencies.
    -y11 (np.ndarray): admittance data measured on a Mach Zehnder device.
    -bias (float): bias voltage at which y11 was measured.
    -ax (float): handle to a matplotlib axes object, where data will be plotted.
    -result (MZIAnalysisResult): object with attributes that store
        results from MZI analyses. This method will set some of the attributes
        of this object.
    -figname (str): name of a file where a plot of the S param measurement will
        be saved. Default is None, in which case no plot will be made.
    '''
    guess = result.guess
    Cm = guess[2]
    #we will keep Cm constant in our fitting procedure:
    guess = [guess[0], guess[1], guess[3], guess[4], guess[5]]
    modelpar, _ = scipy.optimize.leastsq(__mzis11_residuals__, guess, 
                args=(freq.copy(), y11.copy(), Cm), 
                maxfev=200000, xtol=1e-12, ftol=1e-18)
    result.CenterFreq.append(1.0/2/np.pi/np.sqrt(modelpar[2]*modelpar[4]))
    result.Rs.append(modelpar[0])
    result.Cs.append(modelpar[1])
    result.Cm.append(Cm) #modelpar[2]) #50e-15) #
    result.L1.append(modelpar[2])
    result.R1.append(modelpar[3])
    result.C1.append(modelpar[4])
    plttype = 's11'
    if (not figname is None) and (plttype == 'ax'):
        mzimod = __mzimodel__(modelpar, freq, Cm)
        ax.plot(freq, np.abs(y11), label=str(bias))
        ax.hold(True)
        ax.plot(freq, np.abs(mzimod), label=str(bias) + '_fit')
    elif (not figname is None) and (plttype == 'cplx'):
        mzimod = __mzimodel__(modelpar, freq, Cm)
        ax.plot(np.real(y11), np.imag(y11), label=str(bias))
        ax.hold(True)
        ax.plot(np.real(mzimod), np.imag(mzimod), label=str(bias) + '_fit')
    elif (not figname is None) and (plttype == 's11'):
        mzimod = __mzimodel__(modelpar, freq, Cm)
        s11_fit = (1/mzimod-50)/(1/mzimod+50)
        s11 = (1/y11-50)/(1/y11+50)
        ax.plot(freq, 20*np.log10(np.abs(s11_fit)), label=str(bias))
        ax.hold(True)
        ax.plot(freq, 20*np.log10(np.abs(s11)), label=str(bias) + '_fit')
    
def __mzis21__(freq, s21, bias, ax, result, figname=None):
    '''Analyse S21 parameter measured on modulator device.
    '''
    magndata = 20*np.log10(np.abs(s21))
    #look for maximum:
    maxind = np.nanargmax(magndata)
    max_sparam_value = magndata[maxind]
    P = np.polyfit(freq, magndata, 17)
    #norm_idx = np.argmin(np.abs(Frequencies-300e6))
    #magndata -= magndata[norm_idx]
    magndata -= np.polyval(P, 300e6)
    P[-1] -= np.polyval(P, 300e6) #normalize the polynomial fit such that data at 300Mhz is zero dB.
    P[-1] += 1 #shift fitted polynomial 1dB up, and look for the roots
    roots1 = np.roots(P)
    P[-1] += 2 #shift fitted polynomial another 2dB up, and look for the roots
    roots3 = np.roots(P)
    P[-1] -= 3
    #find the root that corresponds with the actual 1dB bandwidth:
    roots1 = np.ma.array(roots1, mask=(np.imag(roots1) <> 0.0))
    min_idx1 = np.nanargmin(np.abs(roots1 - np.average(freq)))
    #find the root that corresponds with the actual 3dB bandwidth:
    roots3 = np.ma.array(roots3, mask=(np.imag(roots3) <> 0.0))
    min_idx3 = np.nanargmin(np.abs(roots3 - np.average(freq)))
    if 1:
        ax.semilogx(freq, magndata)
        ax.hold(True)
        ax.semilogx(freq, np.polyval(P, freq))
        ax.semilogx(roots3[min_idx3], np.polyval(P, roots3[min_idx3]), '^')
    result.f1db.append(np.abs(roots1[min_idx1]))
    result.f3db.append(np.abs(roots3[min_idx3]))
    #add 1dB and 3dB bandwidth from raw data:
    index_1db = np.size(magndata)-1 if np.size(np.where(magndata < -1)) == 0 else np.where(magndata < -1)[0][0]
    result.f1db_raw.append(freq[index_1db])
    index_3db = np.size(magndata)-1 if np.size(np.where(magndata < -3)) == 0 else np.where(magndata < -3)[0][0]
    result.f3db_raw.append(freq[index_3db])
    
def __mzimodel__(p, freqs, Cm):
    '''Electrical equivalent 1-port model of the MZI device. The equivalent 
    circuit consists of three parallel branches:
    -substrate parasitics: Rs [Ohm], Cs [F]
    -metal interconnect parasitics: Cm [F]
    -resonant branch representing the device: Li [H], Ri [Ohm], 
        Ci [F]
    
        
    Args:
    -p (list): a list with the values of the 6 circuit elements that make
        up the equivalent model [Rt Rs Cs Cm Li Ri Ci].
    -freqs (np.ndarray): a numpy array with frequencies.
    -Cm (float): value of Cm (the capacitance between metal traces).
    
    Returns:
    -y11 (np.ndarray): a numpy array with the admittance values of the 
        equivalent circuit, evaluated for the circuit element values specified
        in p and at the frequencies specified in freqs.
    '''
    Rs, Cs, L1, R1, C1 = p
    #Cm = 25e-15
    #termination
    #yt = 1/(Rt*100)
    #substrate branch:
    #zs = Rs*1e3 + 1/(1j*2*np.pi*freqs*Cs*1e-12)
    zs = Rs + 1/(1j*2*np.pi*freqs*Cs)
    ys = 1/zs
    #metal interconnect branch:
    zm = 1/(1j*2*np.pi*freqs*Cm)
    ym = 1/zm
    #device branch:
    #zi = 1j*2*np.pi*freqs*Li*1e-9 + Ri*1e6 + 1/(1j*2*np.pi*freqs*Ci*1e-15)
    z1 = 1j*2*np.pi*freqs*L1 + R1 + 1/(1j*2*np.pi*freqs*C1)
    y1 = 1/z1
    #total admittance:
    return ys + y1 + ym


def __mzis11_residuals__(p, freqs, ys, Cm):
    '''Calculate the distance between the mzimodel (evaluated with
    model parameter values = p) and the measured admittance ys versus frequency
    freqs.
    
    Args:
    -p (list): a list with the values of the 6 circuit elements that make
        up the equivalent mzi model [Rs Cs Cm Li Ri Ci].
    -freqs (np.ndarray): a numpy array with frequencies.
    -ys (np.ndarray): the measured electrical admittance of the mzi device.
    -Cm (float): value of Cm in the equivalent circuit (F).
    
    Returns:
    -residual (float): the distance between model and measurement.
    '''
    non_neg_p = np.abs(p)
    fitf = __mzimodel__(non_neg_p, freqs, Cm)
    dist = fitf-ys
    if (not len(np.argwhere(np.array(p) < 0)) == 0):
        #some negative model parameter given: make residual very large
        dist *= 1e6
    return np.abs(dist)

def AnalyseDetector(data, bias=[-1], inputpower=6.0,
                               gratingloss=-4.0,
                               figname=None):
    '''Analyse the detector measurement result in 'data' and return
        -dark current at bias
        -light current at bias
        -responsivity at bias
         
    Arguments:
    -data: the PortLossMeasurement node, read from XML using lxml.objectify,
        obtained by running the DetectorMeasurement recipe.
    -bias (list): bias voltage(s) (V) at which to evaluate the device parameters
    -inputpower (float): optical power at the tip of the input fiber (dBm)
    -gratingloss (float): optical loss of the input grating (dB).
    -figname (str): name of a file where a plot of the IV measurements will
        be saved. Default is None, in which case no plot will be made.
    
    Returns:
    -result (DetectorMeasurementResult): object that stores all the
        analysis results.
    '''
    result = DetectorAnalysisResult()
    pl = data #.PortLossMeasurement
    #fill in the measurement conditions in the result object:
    result.Wavelength = float(pl.LightCurrent.attrib['Wavelength'])
    result.LaserPower = float(pl.LightCurrent.attrib['Power'])
    powerunit = pl.LightCurrent.attrib['PowerUnit']
    #if 'dbm' in powerunit.lower():
    #    #convert to mW
    #    result.LaserPower = 10**(result.LaserPower/10)
    lshift = pl.PositionerInfo.LeftFibre.attrib['Shift']
    shiftx = float(lshift[1:-1].split(',')[0])
    shifty = float(lshift[1:-1].split(',')[1])
    result.FiberShift = (shiftx**2+shifty**2)**0.5
    result.Bias = bias
    result.FiberOutput = inputpower
    result.GratingLoss = gratingloss
    pwratdetect = inputpower+gratingloss
    result.PowerAtDetector = 10**(pwratdetect/10)
    datetime = time.strptime(pl.attrib['DateStamp'], "%a %b %d %X %Y")
    result.Date = time.strftime("%Y%m%d", datetime)
    result.Time = time.strftime("%X", datetime)
    #extract the detector properties: dark current, light current,
    #responsivity
    darkv = pl.DarkCurrent.V.pyval
    darkv = np.genfromtxt(StringIO(darkv), delimiter=',')
    darki = pl.DarkCurrent.I.pyval
    darki = np.genfromtxt(StringIO(darki), delimiter=',')
    iunit = pl.DarkCurrent.I.attrib['Unit']
    if "symbol='mA'" in iunit:
        darki = darki*0.001
    lightv = pl.LightCurrent.V.pyval
    lightv = np.genfromtxt(StringIO(lightv), delimiter=',')
    lighti = pl.LightCurrent.I.pyval
    lighti = np.genfromtxt(StringIO(lighti), delimiter=',')
    iunit = pl.LightCurrent.I.attrib['Unit']
    if "symbol='mA'" in iunit:
        lighti = lighti*0.001
    for b in bias:
        #dark current: 
        P = np.polyfit(darkv, darki, 7)
        #result.DarkCurrent.append(np.polyval(P, b))
        dark_idx = np.argwhere(darkv == b)[0]
        result.DarkCurrent.append(darki[dark_idx[0]])
        if (not figname is None):
            _, ax = plt.subplots()
            ax.plot(darkv, darki, label='dark')
            ax.hold(True)
            ax.plot(darkv[dark_idx[0]], result.DarkCurrent[-1], '^')
            ax.plot(darkv, np.polyval(P,darkv), '.', label='dark_fit')
            ax.grid(True)
        #light current: 
        P = np.polyfit(lightv, lighti, 7)
        #result.LightCurrent.append(np.polyval(P, b))
        light_idx = np.argwhere(lightv == b)[0]
        result.LightCurrent.append(lighti[light_idx[0]])
        if (not figname is None):
            ax.plot(lightv, lighti, label='light')
            ax.hold(True)
            ax.plot(lightv[light_idx[0]], result.LightCurrent[-1], 's')
            ax.plot(lightv, np.polyval(P,lightv), '-.', label='light_fit')
            ax.grid(True)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='lower right')
            plt.savefig(figname, bbox_inches='tight')
        #responsivity:
        result.Responsivity.append(np.abs(result.LightCurrent[-1] - \
                                          result.DarkCurrent[-1]))
        result.Responsivity[-1] *= 1000 / result.PowerAtDetector
    return result

class DetectorAnalysisResult(AnalysisResult):
    '''This class represents an analysis result, obtained by running
    a AnalyseDetector analysis.
    
    Attributes:
    -Wavelength (float): wavelength at which light current was measured (nm)
    -Bias (list of float): bias voltages at which dark and light currents  
        have been extracted (V)
    -DarkCurrent (list of float): dark currents (A) at each of the Bias values
    -LightCurrent (list of float): light currents (A) at each of the Bias values
    -Responsivity (list of float): responsivities of the detector (A/W),
        extracted at each of the Biases
    -PowerAtDetector (float): optical power reaching the detector (mW) during
        the light current measurement
    -GratingLoss (float): estimated loss in the input grating (dB)
    -FiberShift (float): adjustment of the input fiber during electro-optical
        alignment (um).
    '''
    def __init__(self, **kwargs):
        '''Initialise some attributes of this object.'''
        self._wavelength = 1550.0
        self._bias = [0.0]
        self._darkcurrent = []
        self._lightcurrent = []
        self._responsivity = []
        self._poweratdetector = 0.0
        self._gratingloss = 0.0
        self._fibershift = 0.0
        super(DetectorAnalysisResult, self).__init__(**kwargs)
    
    def __repr__(self):
        ''' Function for debugging help: prints out an overview of the object's 
        property values. '''
        return "%s(lambda:%0.5g,bias:%0.3g,response:%0.3g)" % (self.__class__, 
                                       self.Wavelength, self.Bias, 
                                       self.Responsivity)

    def __str__(self):
        ''' Function for debugging help: prints out an overview of the object's 
        property values. '''
        return self.__repr__()
        
    @Property
    def Wavelength(self):
        doc = ("Wavelength (float): wavelength at which light current was ",
               "measured (nm).")
        def fget(self):
            return FrozenClass.fget_template(self, 'Wavelength')
        def fset(self, wavelength_nm):
            FrozenClass.fset_template(self, wavelength_nm, 'Wavelength', 
                                      [float, int, np.float64])
        return locals()
    
    @Property
    def Bias(self):
        doc = ("Bias (list): bias voltages at which dark and light currents ",
               "have been extracted (V).")
        def fget(self):
            return FrozenClass.fget_template(self, 'Bias')
        def fset(self, volts):
            FrozenClass.fset_template(self, volts, 'Bias', list,
                                      el_dtype=[float, int, np.float64])
        return locals()
    
    @Property
    def DarkCurrent(self):
        doc = ("DarkCurrent (list): dark currents (A) at each of the Bias ",
               "values.")
        def fget(self):
            return FrozenClass.fget_template(self, 'DarkCurrent')
        def fset(self, amps_list):
            FrozenClass.fset_template(self, amps_list, 'DarkCurrent', list,
                                      el_dtype=[float, int, np.float64])
        return locals()
    
    @Property
    def LightCurrent(self):
        doc = ("LightCurrent (list): light currents (A) at each of the Bias ",
               "values.")
        def fget(self):
            return FrozenClass.fget_template(self, 'LightCurrent')
        def fset(self, amps_list):
            FrozenClass.fset_template(self, amps_list, 'LightCurrent', list,
                                      el_dtype=[float, int, np.float64])
        return locals()
    
    @Property
    def Responsivity(self):
        doc = ("Responsivity (list): responsivities of the detector (A/W), ",
               "extracted at each of the Biases")
        def fget(self):
            return FrozenClass.fget_template(self, 'Responsivity')
        def fset(self, response_list):
            FrozenClass.fset_template(self, response_list, 'Responsivity', 
                                      list, el_dtype=[float, int, np.float64])
        return locals()
    
    @Property
    def PowerAtDetector(self):
        doc = ("PowerAtDetector (float): optical power reaching the detector ",
               "(mW) during the light current measurement.")
        def fget(self):
            return FrozenClass.fget_template(self, 'PowerAtDetector')
        def fset(self, milliwats):
            FrozenClass.fset_template(self, milliwats, 'PowerAtDetector', 
                                      [float, int, np.float64])
        return locals()
    
    @Property
    def GratingLoss(self):
        doc = ("GratingLoss (float): estimated loss in the input grating (dB).")
        def fget(self):
            return FrozenClass.fget_template(self, 'GratingLoss')
        def fset(self, dbs):
            FrozenClass.fset_template(self, dbs, 'GratingLoss', 
                                      [float, int, np.float64])
        return locals()
    
    @Property
    def FiberShift(self):
        doc = ("FiberShift (float): adjustment of the input fiber during ",
               "electro-optical alignment (um).")
        def fget(self):
            return FrozenClass.fget_template(self, 'FiberShift')
        def fset(self, microns):
            FrozenClass.fset_template(self, microns, 'FiberShift', 
                                      [float, int, np.float64])
        return locals()
    
class MZIAnalysisResult(AnalysisResult):
    '''This class represents an analysis result, obtained by running
    a AnalyseMZI analysis.
    
    Attributes:
    -CenterFreq (list of float): resonant frequency of the MZI device.
    -Bias (list of float): bias voltages at which the measurement was performed (V)
    -Wavelength (list of float): wavelength at which the measurement was performed (nm)
    -Power (list of float): optical power of the laser source during the measurement
        (dBm)
    -Name (str): name of the device
    -Length (float): length of the electrodes of the device.
        
    The following attributes are extracted from the S11 parameter:
    -guess (list of float): starting values for the fitting process, from
        which the model parameters below are obtained. Starting values
        should appear in this order: [Rs Cs Cm L1 R1 C1]
    -max_freq (float): upper bound of the frequency range in which the
        S11 parameter should be fitted.
    -Rs (list of float): substrate resistance from the device model (Ohm)
    -Cs (list of float): substrate capacitance from the device model (F)
    -Cm (list of float): metal interconnect capacitance (F)
    -L1 (list of float): inductance of the resonant branch (H)
    -R1 (list of float): resistance of the resonant branch (Ohm)
    -C1 (list of float): capacitance of the resonant branch (F)
    
    The following attributes are extracted from S41:
    -f1db (list of float): 1dB bandwidth
    -f3db (list of float): 3dB bandwidth
    -f1db_raw (list of float): 1dB bandwidth, obtained from raw measurement data
    -f3db_raw (list of float): 3dB bandwidth, obtained from raw measurement data
    '''
    def __init__(self, **kwargs):
        '''Initialise some attributes of this object.'''
        self._wavelength = 1550.0
        self._bias = []
        self._centerfreq = [] #20.0e9
        self._max_freq = None
        self._guess = None
        self._length = None
        self._name = ''
        self._rs = [] #50.0
        self._cs = [] #100.0e-15
        self._cm = [] #1e-25
        self._l1 = [] #0.1e-9
        self._r1 = [] #10
        self._c1 = [] #308e-15
        self._f1db = []
        self._f3db = []
        self._f1db_raw = []
        self._f3db_raw = []
        super(MZIAnalysisResult, self).__init__(**kwargs)
    
    def __repr__(self):
        ''' Function for debugging help: prints out an overview of the object's 
        property values. '''
        return "%s(freq:%0.5g)" % (self.__class__, self.CenterFreq)

    def __str__(self):
        ''' Function for debugging help: prints out an overview of the object's 
        property values. '''
        return self.__repr__()
    
    @Property
    def Wavelength(self):
        doc = ("Wavelength (float): wavelength at which the measurement was ",
               "performed (nm).")
        def fget(self):
            return FrozenClass.fget_template(self, 'Wavelength')
        def fset(self, wavelength_nm):
            FrozenClass.fset_template(self, wavelength_nm, 'Wavelength', 
                                      [float, int, np.float64])
        return locals()
    
    @Property
    def Name(self):
        doc = ("Name (str): name of the device.")
        def fget(self):
            return FrozenClass.fget_template(self, 'Name')
        def fset(self, name_str):
            FrozenClass.fset_template(self, name_str, 'Name', 
                                      [str])
        return locals()
    
    @Property
    def Length(self):
        doc = ("Length (float): length of the electrodes of the device.")
        def fget(self):
            return FrozenClass.fget_template(self, 'Length')
        def fset(self, meters):
            FrozenClass.fset_template(self, meters, 'Length', 
                                      [float, int, np.float64])
        return locals()
    
    @Property
    def Bias(self):
        doc = ("Bias (float): bias voltage at which the measurement was ",
               "performed (V).")
        def fget(self):
            return FrozenClass.fget_template(self, 'Bias')
        def fset(self, volts):
            FrozenClass.fset_template(self, volts, 'Bias', list,
                                      el_dtype=[float, int, np.float64])
        return locals()
    
    @Property
    def CenterFreq(self):
        doc = ("CenterFreq (float): resonant frequency of the MZI device.")
        def fget(self):
            return FrozenClass.fget_template(self, 'CenterFreq')
        def fset(self, freq_hz):
            FrozenClass.fset_template(self, freq_hz, 'CenterFreq', list,
                                      el_dtype=[float, int, np.float64])
        return locals()
    
    @Property
    def guess(self):
        doc = ("guess (list of float): starting values for the fitting ",
               "process, from which the model parameters below are obtained. ",
               "Starting values should appear in this order: ",
               "[Rs Cs Cm L1 R1 C1].")
        def fget(self):
            return FrozenClass.fget_template(self, 'guess')
        def fset(self, modelparamlist):
            FrozenClass.fset_template(self, modelparamlist, 'guess', list,
                                      el_dtype=[float, int, np.float64])
        return locals()
    
    @Property
    def max_freq(self):
        doc = ("max_freq (float): upper bound of the frequency range in which ",
               "the S11 parameter should be fitted.")
        def fget(self):
            return FrozenClass.fget_template(self, 'max_freq')
        def fset(self, freq_float):
            FrozenClass.fset_template(self, freq_float, 'max_freq', 
                                      [float, int, np.float64])
        return locals()
        
    @Property
    def Rs(self):
        doc = ("Rs (float): substrate resistance from the device model (Ohm).")
        def fget(self):
            return FrozenClass.fget_template(self, 'Rs')
        def fset(self, r_ohm):
            FrozenClass.fset_template(self, r_ohm, 'Rs', list,
                                      el_dtype=[float, int, np.float64])
        return locals()
    
    @Property
    def Cs(self):
        doc = ("Cs (float): substrate capacitance from the device model (F).")
        def fget(self):
            return FrozenClass.fget_template(self, 'Cs')
        def fset(self, c_farad):
            FrozenClass.fset_template(self, c_farad, 'Cs', list,
                                      el_dtype=[float, int, np.float64])
        return locals()
    
    @Property
    def Cm(self):
        doc = ("Cm (float): metal interconnect capacitance (F).")
        def fget(self):
            return FrozenClass.fget_template(self, 'Cm')
        def fset(self, c_farad):
            FrozenClass.fset_template(self, c_farad, 'Cm', list,
                                      el_dtype=[float, int, np.float64])
        return locals()
    
    @Property
    def L1(self):
        doc = ("L1 (float): inductance of the resonant branch (H).")
        def fget(self):
            return FrozenClass.fget_template(self, 'L1')
        def fset(self, l_henri):
            FrozenClass.fset_template(self, l_henri, 'L1', list,
                                      el_dtype=[float, int, np.float64])
        return locals()
    
    @Property
    def R1(self):
        doc = ("R1 (float): resistance of the resonant branch (Ohm).")
        def fget(self):
            return FrozenClass.fget_template(self, 'R1')
        def fset(self, r_ohm):
            FrozenClass.fset_template(self, r_ohm, 'R1', list,
                                      el_dtype=[float, int, np.float64])
        return locals()
    
    @Property
    def C1(self):
        doc = ("C1 (float): capacitance of the resonant branch (F).")
        def fget(self):
            return FrozenClass.fget_template(self, 'C1')
        def fset(self, c_farad):
            FrozenClass.fset_template(self, c_farad, 'C1', list,
                                      el_dtype=[float, int, np.float64])
        return locals()
    
    @Property
    def f1db(self):
        doc = ("f1db (list of float): 1dB bandwidth.")
        def fget(self):
            return FrozenClass.fget_template(self, 'f1db')
        def fset(self, Hz):
            FrozenClass.fset_template(self, Hz, 'f1db', list,
                                      el_dtype=[float, int, np.float64])
        return locals()
    
    @Property
    def f3db(self):
        doc = ("f3db (list of float): 3dB bandwidth.")
        def fget(self):
            return FrozenClass.fget_template(self, 'f3db')
        def fset(self, Hz):
            FrozenClass.fset_template(self, Hz, 'f3db', list,
                                      el_dtype=[float, int, np.float64])
        return locals()
    
    @Property
    def f1db_raw(self):
        doc = ("f1db_raw (list of float): 1dB bandwidth, obtained from raw ",
               "measurement data.")
        def fget(self):
            return FrozenClass.fget_template(self, 'f1db_raw')
        def fset(self, Hz):
            FrozenClass.fset_template(self, Hz, 'f1db_raw', list,
                                      el_dtype=[float, int, np.float64])
        return locals()
    
    @Property
    def f3db_raw(self):
        doc = ("f3db_raw (list of float): 3dB bandwidth, obtained from raw ",
               "measurement data.")
        def fget(self):
            return FrozenClass.fget_template(self, 'f3db_raw')
        def fset(self, Hz):
            FrozenClass.fset_template(self, Hz, 'f3db_raw', list,
                                      el_dtype=[float, int, np.float64])
        return locals()

if __name__ == "__main__":
    if 0:
        fldr = r'\\ict0101231\OIO\dcosterj\P140406\D17\20150203_161506'
        #fldr = r'\\ict0101231\OIO\dcosterj\P140406\D20\20150130_134803'
        #fldr = r'\\ict0101231\OIO\dcosterj\P140406\D21\20150130_125659'
        #fldr = r'\\ict0101231\OIO\dcosterj\P140406\D22\20150130_115256'
        fnme = 'P140406_D17_(0,0)_SIPP20_LUMPED_MZ.xml'
        f = open(fldr + os.sep + fnme, 'r')
        data = xmlobj.fromstring(f.read())
        f.close() 
        AnalyseMZI(data.ElectroOpticalMeasurements.SiteMeasurement.SParamMeasurement[1],
                      figname=fldr + os.sep + 'test.png',
                      max_freq=10e9)
    elif 1:
        fldr = r'D:\dcosterj\Projects\Photonics\software\pymec\OIO\scratch\sparamfitting'
        file = r'20130808_p123298d15_12dbmRFm8dbm_x0y0_PN500_1553nm_m2V.s2p'
        #f = open(fldr + os.sep + file, 'r')
        data = np.loadtxt(fldr + os.sep + file, delimiter=' ', skiprows=9)
        freq = data[:,0]
        s11real = data[:,1]
        s11imag = data[:,2] 
        s21real = data[:,3]
        s21imag = data[:,4]
        s11 = s11real + 1j*s11imag
        s21 = s21real + 1j*s21imag
        z11 = -50*(s11+1)/(s11-1)
        #z21 = -50*(s21+1)/(s21-1)
        y11 = 1/z11
        #y21 = 1/z21
        max_freq = 20e9
        maxind = np.argmin(np.abs(freq-max_freq))
        freqs11 = freq[0:maxind]
        y11 = y11[0:maxind]
        #y21 = y21[0:maxind]
        #s21 = s21[0:maxind]
        fig, ax = plt.subplots()
        result = MZIAnalysisResult()
        __mzis11__(freqs11, y11, 'PN500', -2, ax, result, figname=fldr + os.sep + 's11_m2V')
        fig, ax2 = plt.subplots()
        __mzis21__(freq, s21, -2, ax2, result, figname=fldr + os.sep + 's21_m2V')
        plt.sca(ax)
        plt.savefig(fldr + os.sep + 'm2V_S11.png', bbox_inches='tight')
        plt.sca(ax2)
        plt.savefig(fldr + os.sep + 'm2V_S21.png', bbox_inches='tight')
        print 'ok'