import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

data_file = r'D:/AstrophysicsMasterULL/Solar/model_jcd.dat'

f = open(data_file, 'r')
lines = f.readlines()
vars = lines[0]
n_lines = len(lines)
height, p, rho, te = map(np.zeros, [n_lines-1] * 4)
for i in range(n_lines-1):
    data = lines[1:][i].split()
    height[i], p[i], rho[i], te[i] = data

height = height
f.close()

gamma = 5/3
g = 273.7 * (100)
H = p/(rho*g)
cs = np.sqrt(gamma*p / rho)
omega_c = gamma*g/(2*cs)
N = np.sqrt((g/H) * ((gamma-1)/gamma))
csz = np.gradient(cs, (height[1]-height[0]))
wcz = np.gradient(omega_c, (height[1]-height[0]))
Nz = np.gradient(N, (height[1]-height[0]))


def dkzds(kx, kz, x, z):
    return 2.0 * N * Nz * cs**2 * kx**2 / np.sqrt(-4 * N**2 * cs**2 * kx**2 + (cs**2 * (kx**2 + kz**2) + omega_c**2)**2) - \
           csz * (cs * (kx**2 + kz**2) + 0.5 * (-4 * N**2 * cs * kx**2 + 2 * cs * (kx**2 + kz**2) * (cs**2 * (kx**2 + kz**2) + omega_c**2))
           / np.sqrt(-4 * N**2 * cs**2 * kx**2 + (cs**2 * (kx**2 + kz**2) + omega_c**2)**2)) \
           - wcz * (1.0 * omega_c * (cs**2 * (kx**2 + kz**2) + omega_c**2) /
           np.sqrt(-4 * N**2 * cs**2 * kx**2 + (cs**2 * (kx**2 + kz**2) + omega_c**2)**2) + omega_c)

def dkxds(kx, kz):
    return 0

def dxds(kx, kz):
    return cs**2*kx + 0.5*(-4*N**2*cs**2*kx + 2*cs**2*kx*(cs**2*(kx**2 + kz**2) + omega_c**2))/ \
           np.sqrt(-4*N**2*cs**2*kx**2 + (cs**2*(kx**2 + kz**2) + omega_c**2)**2)

def dzds(kx, kz):
    return 1.0*cs**2*kz*(cs**2*(kx**2 + kz**2) + omega_c**2)/ \
           np.sqrt(-4*N**2*cs**2*kx**2 + (cs**2*(kx**2 + kz**2) + omega_c**2)**2) + cs**2*kz

def System(kz, kx, x, z, t):
    index = np.argmin(np.abs(height-z))
    return np.array([2.0 * N[index] * Nz[index] * cs[index]**2 * kx**2 / np.sqrt(-4 * N[index]**2 * cs[index]**2 * kx**2 + (cs[index]**2 * (kx**2 + kz**2) + omega_c[index]**2)**2) - \
           csz[index] * (cs[index] * (kx**2 + kz**2) + 0.5 * (-4 * N[index]**2 * cs[index] * kx**2 + 2 * cs[index] * (kx**2 + kz**2) * (cs[index]**2 * (kx**2 + kz**2) + omega_c[index]**2))
           / np.sqrt(-4 * N[index]**2 * cs[index]**2 * kx**2 + (cs[index]**2 * (kx**2 + kz**2) + omega_c[index]**2)**2)) \
           - wcz[index] * (+1.0 * omega_c[index] * (cs[index]**2 * (kx**2 + kz**2) + omega_c[index]**2) /
           np.sqrt(-4 * N[index]**2 * cs[index]**2 * kx**2 + (cs[index]**2 * (kx**2 + kz**2) + omega_c[index]**2)**2) + omega_c[index]),
           0,
           cs[index]**2*kx + 0.5*(-4*N[index]**2*cs[index]**2*kx + 2*cs[index]**2*kx*(cs[index]**2*(kx**2 + kz**2) + omega_c[index]**2))/ np.sqrt(-4*N[index]**2*cs[index]**2*kx**2 + (cs[index]**2*(kx**2 + kz**2) + omega_c[index]**2)**2),
           (1.0*cs[index]**2*kz*(cs[index]**2*(kx**2 + kz**2) + omega_c[index]**2) / np.sqrt(-4*N[index]**2*cs[index]**2*kx**2 + (cs[index]**2*(kx**2 + kz**2) + omega_c[index]**2)**2) + cs[index]**2*kz)])
def rk4(f, initial, t0, tf, n):
    t = np.linspace(t0, tf, n+1)
    vars = np.zeros(shape=(4, n+1))
    vars[0, 0] = initial[0]
    vars[1, 0] = initial[1]
    vars[2, 0] = initial[2]
    vars[3, 0] = initial[3]
    h = t[1]-t[0]
    for i in range(1, n-1):
        k1 = h * f(*vars[:, i-1], t[i-1])
        k2 = h * f(*vars[:, i-1] + 0.5 * k1, t[i-1] + 0.5*h)
        k3 = h * f(*vars[:, i-1] + 0.5 * k2, t[i-1] + 0.5*h)
        k4 = h * f(*vars[:, i-1] + k3, t[i-1] + h)
        vars[:, i] = vars[:, i-1] + (k1 + 2*(k2 + k3) + k4) / 6
    return vars, t

#Initial conditions
w = 2*np.pi*3*10**(-3)
kz = w**2/cs**2 * (N**2/w**2 - 1) * (omega_c**2/w**2 - 1)
reflection = np.argwhere(kz<0)[0][0]
print(reflection)
initial_conds = np.array([(w/cs[reflection]), 0, -5000, height[reflection]])

u, t = rk4(System, initial_conds, 0, 1, 10000)


plt.figure()
plt.plot(u[2, :-1000], u[3, :-1000])
plt.show()
plt.close()
















"""plt.figure(figsize=(15,7))
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
plt.close()"""


"""
kx, kz, omega, sound, omega_corte, brunt = sym.symbols('kx kz w cs wc N')
sound_z, omega_corte_z, brunt_z = sym.symbols('csz wcz Nz')

F = (sound**2 * (kx**2 + kz**2) + omega_corte**2)/2 + (1/2) * \
    sym.sqrt((sound**2 * (kx**2 + kz**2) + omega_corte**2)**2 - 4*sound**2 * brunt**2 * kx**2) \
    - omega**2

dkx_ds = 0
dkz_ds = -(sym.diff(F, sound)*sound_z + sym.diff(F, omega_corte)*omega_corte_z + sym.diff(F, brunt)*brunt_z)
dx_ds = sym.diff(F, kx)
dz_ds = sym.diff(F, kz)

print(dkz_ds)
print(dx_ds)
print(dz_ds)


def System(kx, kz, x, z, t):
    return np.array([2.0 * N * Nz * cs**2 * kx**2 / np.sqrt(-4 * N**2 * cs**2 * kx**2 + (cs**2 * (kx**2 + kz**2) + omega_c**2)**2) - \
           csz * (cs * (kx**2 + kz**2) + 0.5 * (-4 * N**2 * cs * kx**2 + 2 * cs * (kx**2 + kz**2) * (cs**2 * (kx**2 + kz**2) + omega_c**2))
           / np.sqrt(-4 * N**2 * cs**2 * kx**2 + (cs**2 * (kx**2 + kz**2) + omega_c**2)**2)) \
           - wcz * (1.0 * omega_c * (cs**2 * (kx**2 + kz**2) + omega_c**2) /
           np.sqrt(-4 * N**2 * cs**2 * kx**2 + (cs**2 * (kx**2 + kz**2) + omega_c**2)**2) + omega_c), 0, cs**2*kx + 0.5*(-4*N**2*cs**2*kx + 2*cs**2*kx*(cs**2*(kx**2 + kz**2) + omega_c**2))/ \
           np.sqrt(-4*N**2*cs**2*kx**2 + (cs**2*(kx**2 + kz**2) + omega_c**2)**2), 1.0*cs**2*kz*(cs**2*(kx**2 + kz**2) + omega_c**2)/ \
           np.sqrt(-4*N**2*cs**2*kx**2 + (cs**2*(kx**2 + kz**2) + omega_c**2)**2) + cs**2*kz])
"""