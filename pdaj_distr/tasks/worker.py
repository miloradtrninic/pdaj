from scipy.integrate import odeint
import numpy as np
from ..app import app


SCALE_FACTOR_HELPER = 2.


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

    return theta1, theta2, x1, y1, x2, y2, y0


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
@app.task
def simulate_pendulum_instance(L1, L2, m1, m2, tmax, dt, theta1_init, theta2_init):
    y0 = np.array([
        theta1_init,
        0.0,
        theta2_init,
        0.0
    ])

    return solve(L1, L2, m1, m2, tmax, dt, y0)
