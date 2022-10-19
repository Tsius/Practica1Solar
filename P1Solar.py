import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

data_file_PC = r'D:/AstrophysicsMasterULL/Solar/model_jcd.dat'
data_file_port = r'C:\Users\guill\Desktop\Practica1Solar\model_jcd.dat'

f = open(data_file_port, 'r')
lines = f.readlines()
var_names = lines[0]
n_lines = len(lines)
height, p, rho, te = map(np.zeros, [n_lines-1] * 4)
for i in range(n_lines-1):
    data = lines[1:][i].split()
    height[i], p[i], rho[i], te[i] = data

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

def System(kx, kz, x, z, t):
    index = np.argmin(np.abs(height-z))
    return np.array([0,
           2.0 * N[index] * Nz[index] * cs[index]**2 * kx**2 / np.sqrt(-4 * N[index]**2 * cs[index]**2 * kx**2 + (cs[index]**2 * (kx**2 + kz**2) + omega_c[index]**2)**2) - \
           csz[index] * (cs[index] * (kx**2 + kz**2) + 0.5 * (-4 * N[index]**2 * cs[index] * kx**2 + 2 * cs[index] * (kx**2 + kz**2) * (cs[index]**2 * (kx**2 + kz**2) + omega_c[index]**2))
           / np.sqrt(-4 * N[index]**2 * cs[index]**2 * kx**2 + (cs[index]**2 * (kx**2 + kz**2) + omega_c[index]**2)**2)) \
           - wcz[index] * (+1.0 * omega_c[index] * (cs[index]**2 * (kx**2 + kz**2) + omega_c[index]**2) /
           np.sqrt(-4 * N[index]**2 * cs[index]**2 * kx**2 + (cs[index]**2 * (kx**2 + kz**2) + omega_c[index]**2)**2) + omega_c[index]),
           cs[index]**2*kx + 0.5*(-4*N[index]**2*cs[index]**2*kx + 2*cs[index]**2*kx*(cs[index]**2*(kx**2 + kz**2) + omega_c[index]**2))/ np.sqrt(-4*N[index]**2*cs[index]**2*kx**2 + (cs[index]**2*(kx**2 + kz**2) + omega_c[index]**2)**2),
           (1.0*cs[index]**2*kz*(cs[index]**2*(kx**2 + kz**2) + omega_c[index]**2) / np.sqrt(-4*N[index]**2*cs[index]**2*kx**2 + (cs[index]**2*(kx**2 + kz**2) + omega_c[index]**2)**2) + cs[index]**2*kz)])

def rk4(f, initial, t0, tf, n):
    t = np.linspace(t0, tf, n)
    vars = np.zeros(shape=(4, n))
    vars[0, 0] = initial[0]
    vars[1, 0] = initial[1]
    vars[2, 0] = initial[2]
    vars[3, 0] = initial[3]
    h = t[1]-t[0]
    for i in range(1, n):
        k1 = h * f(*vars[:, i-1], t[i-1])
        k2 = h * f(*vars[:, i-1] + 0.5 * k1, t[i-1] + 0.5*h)
        k3 = h * f(*vars[:, i-1] + 0.5 * k2, t[i-1] + 0.5*h)
        k4 = h * f(*vars[:, i-1] + k3, t[i-1] + h)
        vars[:, i] = vars[:, i-1] + (k1 + 2*(k2 + k3) + k4) / 6
    return vars, t

#Initial conditions
w = 2*np.pi*10**(-3)*np.array([2, 3, 3.5, 5])
z_ini_index = 200
initial_conds = np.zeros(shape=(4,4))
initial_conds[0,:] = np.array([(w[0]/cs[z_ini_index]), 0, 0, height[z_ini_index]])
initial_conds[1,:] = np.array([(w[1]/cs[z_ini_index]), 0, 0, height[z_ini_index]])
initial_conds[2,:] = np.array([(w[2]/cs[z_ini_index]), 0, 0, height[z_ini_index]])
initial_conds[3,:] = np.array([(w[3]/cs[z_ini_index]), 0, 0, height[z_ini_index]])

s2, t = rk4(System, initial_conds[0,:], 0, 2.5, 30000)
s3, t = rk4(System, initial_conds[1,:], 0, 1, 30000)
s3_5, t = rk4(System, initial_conds[2,:], 0, 1, 30000)
s5, t = rk4(System, initial_conds[3,:], 0, 0.5, 30000)


plt.figure()
plt.plot(s2[2, :]/1000, s2[3, :], 'r--', label=r'$\nu = 2 mHz$')
plt.axhline(np.max(s2[3, :]), color='r')
plt.plot(s3[2, :]/1000, s3[3, :], 'c--', label=r'$\nu = 3 mHz$')
plt.axhline(np.max(s3[3, :]), color='c')
plt.plot(s3_5[2, :]/1000, s3_5[3, :], 'y--', label=r'$\nu = 3.5 mHz$')
plt.axhline(np.max(s3_5[3, :]), color='y')
plt.plot(s5[2, :]/1000, s5[3, :], 'k--', label=r'$\nu = 5 mHz$')
plt.axhline(np.max(s5[3, :]), color='k')
plt.ylim(height[z_ini_index]-250, height[0]+100)
plt.xlim(0, 30)
plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2)
plt.xlabel('X-direction [Mm]')
plt.ylabel('Height [km]')
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