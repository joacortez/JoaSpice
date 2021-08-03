""" Library for simulating stuff for ELNW """
import csv
import functools
import os
import warnings
from abc import get_cache_token

import ltspice
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import _label_from_arg
from order_of_magnitude import order_of_magnitude as ofm
from PyLTSpice.LTSpiceBatch import SimCommander
from scipy.signal import argrelextrema
import control


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func


class SimulationHelper:

    def __init__(self, path, filename, iterations):
        """ pass path using __file__ """
        path = os.path.dirname(path)
        self.path = path + "\\" + filename
        self.filename = filename
        self.LT = SimCommander(self.path + ".asc")
        self.static_comps = []
        self.dynamic_comps = []
        self.iterations = iterations
        self.intructions = ""

    @property
    def runno(self):
        return self.LT.runno

    @property
    def oknum(self):
        return self.LT.okSim

    @property
    def netlist(self):
        return self.LT.netlist

    def add_static_comp2list(self, name, config):
        self.static_comps.append([name, config])

    def add_dynamic_comp2list(self, name, start, var, end=")"):
        configs = []
        for config in var:
            configs.append(start + config + end)

        self.dynamic_comps.append([name, configs])

    def set_static_comps(self):
        for comp in self.static_comps:
            self.LT.set_component_value(comp[0], comp[1])

    def set_component_value(self, comp: str, config):
        self.LT.set_component_value(comp, config)

    def set_dynamic_comps(self, index=0):
        for comp in self.dynamic_comps:
            self.LT.set_component_value(comp[0], comp[1][index])

    def set_parameters(self, **kwargs):
        self.LT.set_parameters(kwargs=kwargs)

    def add_instructions(self, instructions):
        self.intructions = instructions
        self.LT.add_instructions(instructions)

    def add_component(self, comp: str) -> None:
        index = self.LT._getline_startingwith(".")
        self.LT.netlist.insert(index, comp + "\n")

    def run(self, getList: bool = True) -> list:
        self.LT.run()
        self.LT.wait_completion()
        if getList:
            return read_LTspice(self.path + "_" + str(self.oknum) + ".raw")
        return None

    def reset_netlist(self):
        self.LT.reset_netlist()

    def write_netlist(self):
        self.LT.write_netlist(self.LT.netlist_file)

    def reset_all(self):
        self.LT.reset_netlist()
        self.static_comps.clear()
        self.dynamic_comps.clear()


def test_import():
    print("it's working")


def parse_imag(data):
    """returns real and imaginary components"""
    real = [n.real for n in data]
    imag = [n.imag for n in data]

    return real, imag


def nyquist(data, **kwargs):
    """ plots data in nyquist format """
    x, y = parse_imag(data)
    plt.plot(x, y, kwargs=kwargs)
    plt.legend()
    return


def get_operation(x, y, operation):
    """
    return x[index], y[index], index
    operation np.something
    """
    # return argrelextrema(y, operation)
    index = argrelextrema(y, operation)
    print('getting index')
    print(index)
    _x = x[index]
    _y = y[index]
    return _x, _y, index


def format_scientific(data, precision=3):
    """ data must be type float """
    # assert str(data.dtype).__contains__("float"), "array type is not float"
    for i in range(len(data)):
        data[i] = np.format_float_scientific(
            data[i], precision=precision, unique=True)
    return data


def format_positional(data, precision=3):
    """parses array to float with precision decimals"""
    # assert str(data.dtype).__contains__("float"), "array type is not float"
    for i in range(len(data)):
        data[i] = np.format_float_positional(
            data[i], precision=precision, trim='k')
    return data


def format_print(data, precision: int = 3):
    """ parses array to array of strings with precision decimals """
    string = "{:." + str(precision) + "f}"
    formatter = string.format
    parsedData = []
    for n in data:
        parsedData.append(formatter(n))

    return parsedData


def quick_scientific(num, precision=3):
    """ num must be of type float """
    # assert str(type(num)).__contains__("float"), "num type is not float, it's " + str(type(num))
    return np.format_float_scientific(num, precision=precision, unique=True)


def quick_positional(num, precision=3):
    """ use quick_format_print instead
    num must be of type float """
    # assert str(type(num)).__contains__("float"), "num type is not float, it's " + str(type(num))
    return np.format_float_positional(num, precision=precision, trim='k')


def quick_format_print(data, precision=3):
    """ parses array to string with precision decimals """
    string = "{:." + str(precision) + "f}"
    formatter = string.format

    return formatter(data)


def get_magnitude(num, precision=3, returnAll=False, scale=None):
    """ num must be of type float """
    # assert str(type(num)).__contains__("float"), "num type is not float, it's " + str(type(num))
    data = ofm.symbol(num, decimals=precision, scale=scale)

    return data if returnAll else data[2]


def get_max(x, y, returnIndex=False, format=False, precision=3):
    """return x[index], y[index],
        index np.greaster """
    index = argrelextrema(y, np.greater)[0]
    X = x[index]
    Y = y[index]

    if format:
        X = format_scientific(X, precision=precision)
        Y = format_scientific(Y, precision=precision)

    return (X, Y, index) if returnIndex else (X, Y)


def get_min(x, y, returnIndex=False, format=False, precision=3):
    """return x[index], y[index],
        index np.less """
    index = argrelextrema(y, np.less)[0]

    X = x[index]
    Y = y[index]

    if format:
        X = format_scientific(X, precision=precision)
        Y = format_scientific(Y, precision=precision)

    return X, Y, index if returnIndex else X, Y

# @NotImplemented
# def get_avr(x, y):
#     """
#     return x[index], y[index], index
#     np.avr
#     """
#     index = argrelextrema(y, np.average)

#     return x[index], y[index], index


def get_peak_to_peak(data: list):
    """ returns the peak to peak measurment """
    max = argrelextrema(data, np.greater)[0]
    min = argrelextrema(data, np.less)[0]

    pp = abs(data[max[1]]) + abs(data[min[1]])

    return pp


def get_delta_t(data1: list, data2: list, time: list):
    """ calculates delta t between data1, data2 """
    # time1 = get_max(data1, time)[1]
    # time2 = get_max(data2, time)[1]

    time1 = time[argrelextrema(data1, np.greater)[0]]
    time2 = time[argrelextrema(data2, np.greater)[0]]

    # for shit in time1:
    #     print(shit)
    delta_t = abs(time1[1] - time2[1])

    return delta_t


def get_phi(data1, data2, time, freq, returnDelta: bool = False, deg: bool = False):
    """ return the phasenverschiebung in rad and Delta t from data"""
    delta_t = get_delta_t(data1, data2, time)

    omega = 360 * freq if deg else 2 * np.pi * freq

    phi = delta_t * omega

    return phi, delta_t if returnDelta else phi


def mag2db(mag):
    return control.mag2db(mag)


def db2mag(db):
    return control.db2mag(db)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    return x, y


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    return rho, phi


def calculate_delta_phi(delta_t, freq, deg: bool = False):
    """ calculates phi from delta t and frequency\n
    phi = delta t * omega"""
    return 360 * freq * delta_t if deg else 2 * np.pi * freq * delta_t


def get_omega(freq, deg=False):
    """ calculates omega,
    omega = 2 pi * freq """
    omega = 2 * np.pi * freq
    return np.rad2deg(omega) if deg else omega


def get_complex(data, freq, delta_t=0) -> np.complex64:
    return data * np.exp(1j * calculate_delta_phi(delta_t, freq))


def read_csv(path, delimiter=',', lineterminator=None, skip=0):
    """ reads data from file and returns a dictionary """
    print("reading data from csv")

    data = {}

    with open(path, mode="r") as file:
        reader = csv.reader(file, delimiter=delimiter)
        headers = next(reader, None)

        for h in headers:
            data[h] = []

        if (skip != 0):
            for _ in range(skip):
                next(reader)
        if lineterminator is None:
            for row in reader:
                for h, v in zip(headers, row):
                    data[h].append(v)
        else:
            for row in reader:
                for h, v in zip(headers, row):
                    data[h].append(v.strip(lineterminator))

    return data


def dictConverter(dict, type=np.float64, skip=True, keyList=[]):
    """ parses data from list in dictionary to type """
    if len(keyList) > 0:
        dict = clean_dict(dict, skip=skip, keyList=keyList)
    for k, v in dict.items():
        dict[k] = np.array(v, type)

    return dict


def dict2rows(dict, skip=True, keyList=[]):
    """ parses dictionary into an array """
    if len(keyList) > 0:
        dict = clean_dict(dict, skip=skip, keyList=keyList)
    keys = list(dict.keys())
    length = len(dict[keys[0]])
    data = []
    for i in range(length):
        row = []
        for k in keys:
            row.append(dict[k][i])

        data.append(row)
    return data


def clean_dict(dicti, skip=True, keyList=[]):
    """ removes unwanted keys from dict """
    assert len(keyList) > 0, "Keylist empty"

    dicti = dict(dicti)

    unwantedK = keyList if skip else set(list(dicti.keys())) - set(keyList)

    for k in unwantedK:
        dicti.pop(k, None)

    return dicti


def dict2csv(dict, path, skip=True, keyList=[], delimiter=",", lineterminator="\n", pHeaders=True):
    if len(keyList) > 0:
        dict = clean_dict(dict, skip=skip, keyList=keyList)
    headers = list(dict.keys())
    rows = dict2rows(dict)

    with open(path, mode='w') as file:
        writer = csv.writer(file, delimiter=delimiter,
                            lineterminator=lineterminator)
        if pHeaders:
            writer.writerow(headers)

        for row in rows:
            writer.writerow(format_print(row))


def dict2latex(dict, path, skip=True, keyList=[]):
    dict2csv(dict=dict, path=path, skip=skip, keyList=keyList,
             delimiter="&", lineterminator="\\\\\n", pHeaders=False)


def read_LTspice(path) -> dict:
    l = ltspice.Ltspice(path)
    l.parse()
    data = {}
    keys = l.variables
    if l._mode == 'Transient':
        data[keys[0]] = l.get_time()
        keys = l.variables[1:]
    elif l._mode == 'FFT' or l._mode == 'AC' or l._mode == 'Noise':
        data[keys[0]] = l.get_frequency()
        keys = l.variables[1:]

    for k in keys:
        data[k] = l.get_data(k)

    return data


def get_keys(dict) -> list:
    return (list(dict.keys()))


def change_keys(data: dict, keys: list) -> dict:
    new_data = {}
    old_keys = get_keys(data)

    for i in range(len(data)):
        new_data[keys[i]] = data[old_keys[i]]

    return new_data


def _legend_helper(label, color=None, marker='x'):
    return mlines.Line2D([], [], color=color, marker=marker, label=label)


def plot(data, ideal, xlabel, ylabel, path,  marker='x', outside=False, scale=["linear", "linear"], locs=[
        "lower left", "upper right"]):

    plt.rc("text", usetex=True)
    plt.rcParams.update({'font.size': 12})
    ax = plt.gca()
    colors = cm.rainbow(np.linspace(0, 1, len(data) - 1))
    ax.set_xscale(scale[0])
    ax.set_yscale(scale[1])

    locs = [
        (0.6, 1.05), (0, 1.05)] if outside else locs

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.grid(True)

    legends = []
    leg_ideal = mlines.Line2D([], [], color='0', label='Theoretischer Verlauf')
    leg_data = mlines.Line2D(
        [], [], color='0', marker=marker, linewidth=0, label='Messdaten')

    xdata = data.pop(get_keys(data)[0])
    xideal = ideal.pop(get_keys(ideal)[0])

    for k, c in zip(data, colors):
        ax.scatter(xdata, data[k], color=c, marker=marker)
        ax.plot(xideal, ideal[k], color=c)
        legends.append(_legend_helper(k, color=c))

    _leg = ax.legend(handles=[leg_ideal, leg_data],
                     loc=locs[0])
    ax.add_artist(_leg)

    leg = ax.legend(handles=legends,
                    loc=locs[1])
    ax.add_artist(leg)

    # plt.savefig(path, bbox_extra_artists=(
    #     _leg, ), bbox_inches='tight')
    plt.savefig(path, bbox_extra_artists=(
        _leg, leg, ), bbox_inches='tight')

    plt.close()


def plot_einzel(data, xlabel, ylabel, path, scale=["linear", "linear"],):

    plt.rc("text", usetex=True)
    plt.rcParams.update({'font.size': 12})
    ax = plt.gca()
    ax.set_xscale(scale[0])
    ax.set_yscale(scale[1])

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.grid(True)

    xdata = data.pop(get_keys(data)[0])

    for k in data:
        print(k)
        ax.plot(xdata, data[k], label=k)

    plt.legend()
    plt.savefig(path)

    plt.close()


def URI(U=None, I=None, R=None):
    """ Automatisches Ohmsches Gesetz """
    if(U == None):
        return I * R
    elif (I == None):
        return U / R
    elif (R == None):
        return U / I
    else:
        return np.nan


def calculate_resistanz(U=None, I=None, R=None):
    """ allgemeiner Resistanz Rechner """
    return URI(U, R, I)


# def calculate_Induktanz(U=None, I=None, Z=None, L=None, omega=None):
    """ allgemeiner Induktanzrechner """
    # if(omega == None):
