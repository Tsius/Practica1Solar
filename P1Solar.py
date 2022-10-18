import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

data_file = r'C:/Users/guill/Desktop/Practica1Solar/model_jcd.dat'

f = open(data_file, 'r')
lines = f.readlines()
vars = lines[0]
n_lines = len(lines)
height, p, rho, t = map(np.zeros, [n_lines-1] * 4)
for i in range(n_lines-1):
    data = lines[1:][i].split()
    height[i], p[i], rho[i], t[i] = data

height = height[::-1]
p = p[::-1]
rho = rho[::-1]
t = t[::-1]
f.close()

gamma = 5/3
g = 273.7
H = p/(rho*g)
cs = np.sqrt(gamma*p / rho)
omega_c = gamma*g/(2*cs)
N = np.sqrt((g/H) * ((gamma-1)/gamma))

z, kx, kz, omega = sym.symbols('z kx kz w')
sound = sym.Function('cs')
omega_corte = sym.Function('wc')
brunt = sym.Function('N')
F = (sound(z)**2*(kx**2 + kz**2) + omega_corte(z)**2)/2 + \
    (1/2)*sym.sqrt((sound(z)**2*(kx**2 + kz**2) + omega_corte(z)**2)**2 -
                   4 * sound(z)**2 * brunt(z)**2 * kx**2) - omega**2

dkx_ds = 0
dkz_ds = -sym.diff(F, z)
dx_ds = sym.diff(F, kx)
dz_ds = sym.diff(F, kz)



plt.figure(figsize=(15,7))
plt.subplot(1,3,1)
plt.plot(height, cs**2, label=r'$c_s^2$')
plt.yscale('log')
plt.legend()
plt.subplot(1,3,2)
plt.plot(height, omega_c, label=r'$\omega_c$')
plt.yscale('log')
plt.legend()
plt.subplot(1,3,3)
plt.plot(height, N**2, label=r'$N^2$')
plt.yscale('log')
plt.legend()
plt.show()
plt.close()


"""
def rk4(f, u0, t0, tf, n):
    t = np.linspace(t0, tf, n+1)
    u = np.array((n+1)*[u0])
    h = t[1]-t[0]
    for i in range(n):
        k1 = h * f(u[i], t[i])
        k2 = h * f(u[i] + 0.5 * k1, t[i] + 0.5*h)
        k3 = h * f(u[i] + 0.5 * k2, t[i] + 0.5*h)
        k4 = h * f(u[i] + k3, t[i] + h)
        u[i+1] = u[i] + (k1 + 2*(k2 + k3 ) + k4) / 6
    return u, t
"""