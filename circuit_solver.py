# -*- coding: utf-8 -*-


# --- Implementierung Testat --------------------------------------------------
# Implementieren Sie hierdie Lösung für die zweite Testataufgabe.

# --- Testcode ----------------------------------------------------------------
# Fügen Sie hier Ihren Testcode ein. Nutzen Sie dafür unter anderem die
# Code-Beispiele in den grauen Boxen der Aufgabenstellung sowie auch die
# Angaben in den *.results Files im abgegebenen Ordner.

import numpy as np


def netlist_parser(fname):
    """
    Parse the txt file of the circuit's components into
    a dictionary containing the previous and next node,
    as well as the value from the component.

    Parameters
    ----------
    fname : str
        relative path of the txt file.

    Returns
    -------
    elements : dict
        dictionary containing passive and active
        components of the circuit.

    """
    _prefix = {'y': 1e-24,  # yocto
               'z': 1e-21,  # zepto
               'a': 1e-18,  # atto
               'f': 1e-15,  # femto
               'p': 1e-12,  # pico
               'n': 1e-9,   # nano
               'u': 1e-6,   # micro
               'm': 1e-3,   # mili
               'c': 1e-2,   # centi
               'd': 1e-1,   # deci
               'k': 1e3,    # kilo
               'meg': 1e6,    # mega
               'g': 1e9,    # giga
               't': 1e12,   # tera
               '': 1
               }
    elements = {}
    with open(fname, encoding='utf-8') as file:
        for line in file:
            if '*' in line:
                continue
            else:
                parameters = line.split(';')[0].split(' ')
                if parameters[3].isalpha():
                    parameters[3] = parameters[4]
                parameters[3] = parameters[3].lower()
                try:
                    value = float(parameters[3])
                except ValueError:
                    f_part = parameters[3].rstrip('megunkyzafpcdt\n')
                    p_part = parameters[3][len(f_part):]
                    p_part = p_part.rstrip('\n')
                    value = float(f_part) * _prefix[p_part]
                elements[parameters[0]] = {'n+': 'Vn' + parameters[1],
                                           'n-': 'Vn' + parameters[2],
                                           'value': value}
    return elements


def inventory(elements):
    """
    From dict elements, return the components aggregated by
    function in the circuit.

    Parameters
    ----------
    elements : dict
        dictionary containing passive and active
        components of the circuit.

    Returns
    -------
    s : str
        string to be printed with components
        from the circuit, separated by function.

    """
    resistors = []
    vol_sources = []
    curr_sources = []
    for element in elements.keys():
        if element.startswith('R'):
            resistors.append(element)
        elif element.startswith('V'):
            vol_sources.append(element)
        elif element.startswith('I'):
            curr_sources.append(element)
    s = 'Resistors: ' + ', '.join(resistors) + '\n' +\
        'Voltage Sources: ' + ', '.join(vol_sources) + '\n' +\
        'Current Sources: ' + ', '.join(curr_sources)
    return s


def mna_build(elements):
    """
    Build components M and y from the linear equations
    to calculate the value of the unknowns array.

    Parameters
    ----------
    elements : dict
        dictionary containing passive and active
        components of the circuit.

    Returns
    -------
    unknowns : list
        list of variables whose values are unknown by this point.
    M : np.ndarray
        Matrix with the resistance values.
    y : np.ndarray
        array with sources of voltage and current to be multiplied by M^-1.

    """
    unknowns = []
    for key in elements:
        if key.startswith('V'):
            unknowns.append('I'+key.lower())
        unknowns.append(elements[key]['n+'])
        unknowns.append(elements[key]['n-'])
    unknowns = list(set(unknowns))
    unknowns.remove('Vn0')

    M = np.zeros((len(unknowns), len(unknowns)))
    y = np.zeros((len(unknowns)))

    for key, value in elements.items():
        if key.startswith('R'):
            np_id = unknowns.index(value['n+']) if value['n+'] != 'Vn0' else -1
            nm_id = unknowns.index(value['n-']) if value['n-'] != 'Vn0' else -1
            if np_id != -1 and nm_id != -1:
                M[np_id, np_id] += 1 / (value['value'])
                M[nm_id, nm_id] += 1 / (value['value'])
                M[nm_id, np_id] -= 1 / (value['value'])
                M[np_id, nm_id] -= 1 / (value['value'])
            else:
                if np_id == -1:
                    M[nm_id, nm_id] += 1 / (value['value'])
                elif nm_id == -1:
                    M[np_id, np_id] += 1 / (value['value'])

        elif key.startswith('V'):
            np_id = unknowns.index(value['n+']) if value['n+'] != 'Vn0' else -1
            nm_id = unknowns.index(value['n-']) if value['n-'] != 'Vn0' else -1
            n_v = unknowns.index('I' + key.lower())
            if np_id != -1 and nm_id != -1:
                M[np_id, n_v] += 1
                M[nm_id, n_v] -= 1
                M[n_v, np_id] += 1
                M[n_v, nm_id] -= 1
            else:
                if np_id == -1:
                    M[nm_id, n_v] -= 1
                    M[n_v, nm_id] -= 1
                elif nm_id == -1:
                    M[np_id, n_v] += 1
                    M[n_v, np_id] += 1
            y[unknowns.index('I'+key.lower())] += value['value']

        else:
            np_id = unknowns.index(value['n+']) if value['n+'] != 'Vn0' else -1
            nm_id = unknowns.index(value['n-']) if value['n-'] != 'Vn0' else -1
            if np_id != -1 and nm_id != -1:
                y[np_id] -= value['value']
                y[nm_id] += value['value']
            else:
                if np_id == -1:
                    y[nm_id] += value['value']
                elif nm_id == -1:
                    y[np_id] -= value['value']
    return unknowns, M, y


def mna_solve(unknowns, M, y):
    """
    Solve linear equations of the circuit, using
    parameters M and y, by multiplying M^-1@y through
    simple matrix algebra.

    Parameters
    ----------
    unknowns : list
        list of variables whose values are unknown by this point.
    M : np.ndarray
        Matrix with the resistance values.
    y : np.ndarray
        array with sources of voltage and current to be multiplied by M^-1.

    Returns
    -------
    solution : dict
        values from the unknowns variables.

    """
    M_inv = np.linalg.inv(M)
    erg = M_inv @ y
    solution = dict({k for k in zip(unknowns, erg)})
    solution['Vn0'] = 0
    return solution


def mna_report(elements, solution):
    """
    Formatting the solution into a report with
    the components of the circuits receiving the
    calculated values of Voltate between nodes and
    Current passing through the component.

    Parameters
    ----------
    elements : dict
        dictionary containing passive and active
        components of the circuit.
    solution : dict
        values from the unknowns variables.

    Returns
    -------
    report : dict
        prettyfied dictionary with calculated values of Voltage
        and Current for each of the circuit's components.

    """
    report = {}
    for k, value in elements.items():
        if k.startswith('R'):
            report[k] = {'V': solution[value['n+']] - solution[value['n-']],
                         'I': (solution[value['n+']] -
                               solution[value['n-']]) / value['value']}
        elif k.startswith('V'):
            report[k] = {'V': value['value'],
                         'I': solution['I'+k.lower()]}
        else:
            report[k] = {'V': solution[value['n+']] - solution[value['n-']],
                         'I': value['value']}
    return report


if __name__ == '__main__':
    elements = netlist_parser('netlist.cir')
    # elements = {'R1': {'n+': 'Vn1', 'n-': 'Vn2', 'value': 2.2},
    #             'R2': {'n+': 'Vn2', 'n-': 'Vn0', 'value': 4000.0},
    #             'R3': {'n+': 'Vn2', 'n-': 'Vn3', 'value': 100000.0}}
    s = inventory(elements)
    unknowns, M, y = mna_build(elements)
    solution = mna_solve(unknowns, M, y)
    report = mna_report(elements, solution)
    print(report)
