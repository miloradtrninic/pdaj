import sys
import numpy as np
import csv
import argparse
import logging
from scipy.integrate import odeint


# The gravitational acceleration (m.s-2).
g = 9.81

def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
         raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def deriv(y, t, L1, L2, m1, m2):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    theta1, z1, theta2, z2 = y

    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

    theta1dot = z1
    z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
             (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
    theta2dot = z2
    z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) +
             m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
    return theta1dot, z1dot, theta2dot, z2dot

def solve(L1, L2, m1, m2, tmax, dt, y0):
    t = np.arange(0, tmax+dt, dt)

    # Do the numerical integration of the equations of motion
    y = odeint(deriv, y0, t, args=(L1, L2, m1, m2))
    theta1, theta2 = y[:, 0], y[:, 2]

    # Convert to Cartesian coordinates of the two bob positions.
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    return theta1, theta2, x1, y1, x2, y2

def simulate_pendulum(theta_resolution, results_file_name, tmax, dt):
    # Pendulum rod lengths (m), bob masses (kg).
    L1, L2 = 1.0, 1.0
    m1, m2 = 1.0, 1.0

    # Maximum time, time point spacings (all in s).
    with open(results_file_name, 'w') as resultsfile:
        fieldnames = ['theta1_init', 'theta2_init', 'theta1', 'theta2', 'x1', 'x2', 'y1', 'y2']
        writer = csv.DictWriter(resultsfile, fieldnames=fieldnames)
        writer.writeheader()
        for theta1_init in np.linspace(0, 2*np.pi, theta_resolution):
            for theta2_init in np.linspace(0, 2*np.pi, theta_resolution):
                # Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
                y0 = np.array([
                    theta1_init,
                    0.0,
                    theta2_init,
                    0.0
                ])

                theta1, theta2, x1, y1, x2, y2 = solve(L1, L2, m1, m2, tmax, dt, y0)
                writer.writerow({'theta1_init': theta1_init,
                                 'theta2_init': theta2_init,
                                 'theta1' : theta1[-1],
                                 'theta2' : theta2[-1],
                                 'x1' : x1[-1],
                                 'x2' : x2[-1],
                                 'y1' : y1[-1],
                                 'y2' : y2[-1]})
                # print theta1_init, theta2_init, theta1[-1], theta2[-1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r',
        '--resolution',
        type = check_positive,
        default = 2
    )
    parser.add_argument(
        '-n',
        '--name',
        type = str,
        default = 'results.csv'
    )
    parser.add_argument(
        '-t',
        '--timemax',
        type = float,
        default = 30.0
    )
    parser.add_argument(
        '-d',
        '--dtime',
        type = float,
        default = 0.01
    )

    args = parser.parse_args()
    simulate_pendulum(args.resolution, args.name, args.timemax, args.dtime)

if __name__ == "__main__":
    main()