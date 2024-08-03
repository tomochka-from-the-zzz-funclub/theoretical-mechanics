import math
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint 

def SystemOfEquations(y, t, m1, m2, l, g, alpha):
    #y массив с начальными значениями переменных системы ([S, Phi, dS/dt, dPhi/dt]).
    #t массив с временными точками, на которых нужно вычислить значения переменных.
    yt = np.zeros_like(y)

    yt[0] = y[2]
    yt[1] = y[3]
    # коэффициент для матрицы системы уравнений вида A * [S, Phi]' = B, где S - смещение
    #Phi - угол, A - матрица коэффициентов, а B - вектор правой части.
    #a11 * S' + a12 * Phi' = b1
    #a21 * S' + a22 * Phi' = b2
    a11 = m1 + m2 
    a12 = -m2 * l * np.cos(y[1] - alpha)
    a21 = -np.cos(y[1] - alpha)
    a22 = l

    b1 = g * np.sin(alpha) * (m1 + m2) - m2 * l * ((y[3]) ** 2) * np.sin(y[1] - alpha)
    b2 = -g * np.sin(y[1])

    # по крамеру
    yt[2] = (b1 * a22 - a12 * b2) / (a11 * a22 - a12 * a21)
    yt[3] = (b2 * a11 - a21 * b1) / (a11 * a22 - a12 * a21)

    return yt


def rotate(x, y, alpha): #умножаем стольбец координат на матрицу поворота
    xr = math.cos(alpha) * x - math.sin(alpha) * y
    yr = (math.sin(alpha)) * x + math.cos(alpha) * y
    return xr, yr


def Film(i):#обновляет анимацию на каждом кадре
    Point_A.set_data(X_A[i], Y_A[i])
    Point_B.set_data(X_B[i], Y_B[i])
    Line_AB.set_data([X_A[i], X_B[i]], [Y_A[i], Y_B[i]])
    Box.set_data(X_A[i] + X_Box, Y_A[i] + Y_Box)
    return [Point_A, Point_B, Box, Line_AB]


m1 = 150
m2 = 500
L = 3  
Phi0 = np.pi / 6
alpha = np.pi / 6

g = 9.81
S0 = 3
DS0 = 0
DPhi0 = 20
y0 = [S0, Phi0, DS0, DPhi0]
Tfin = 10#время окончания интегрирования.
NT = 1000#количество временных точек для интегрирования
t = np.linspace(Tfin, 0, NT)#от Tfin до 0 с равными промежутками.

Y = odeint(SystemOfEquations, y0, t, (m1, m2, L, g, alpha))

x = Y[:, 0]
phi = Y[:, 1]
Dx = Y[:, 2]
Dphi = Y[:, 3]

DDx = [SystemOfEquations(y, t, m1, m2, L, g, alpha)[2] for y, t in zip(Y, t)]
DDphi = [SystemOfEquations(y, t, m1, m2, L, g, alpha)[3] for y, t in zip(Y, t)]


fig_for_graphs = plt.figure(figsize=[13, 7])
ax_for_graphs = fig_for_graphs.add_subplot(2, 3, 1)
ax_for_graphs.plot(t, x, color='blue')
ax_for_graphs.set_title("x(t)")
ax_for_graphs.set(xlim=[0, Tfin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 3, 4)
ax_for_graphs.plot(t, phi, color='red')
ax_for_graphs.set_title('phi(t)')
ax_for_graphs.set(xlim=[0, Tfin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 3, 2)
ax_for_graphs.plot(t, Dx, color='green')
ax_for_graphs.set_title("x'(t)")
ax_for_graphs.set(xlim=[0, Tfin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 3, 5)
ax_for_graphs.plot(t, Dphi, color='black')
ax_for_graphs.set_title('phi\'(t)')
ax_for_graphs.set(xlim=[0, Tfin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 3, 3)
ax_for_graphs.plot(t, DDx, color='blue')
ax_for_graphs.set_title('x\'\'(t)')
ax_for_graphs.set(xlim=[0, Tfin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 3, 6)
ax_for_graphs.plot(t, DDphi, color='red')
ax_for_graphs.set_title('phi\'\'(t)')
ax_for_graphs.set(xlim=[0, Tfin])
ax_for_graphs.grid(True)


fig = plt.figure(figsize=[20, 8])
ax = fig.add_subplot(1, 1, 1)
ax.axis('equal')
ax.set(xlim=[-3, 10], ylim=[-3, 10])

BoxX = 3
BoxY = 2
bx = BoxX / 2
by = BoxY / 2
l = L
Pi = np.pi
tg = np.tan(alpha)

# Отрисовка земли
xMax = x[0] + (math.fabs(x[-1] - x[0]))/10
xMin = x[-1]*(-1)
coef = by*1.1
X_Ground = [xMax - BoxX*2*tg, xMax, xMin, xMax]
Y_Ground = [xMax*tg - coef + BoxX*2, xMax *
            tg - coef, xMin*tg - coef, xMin*tg - coef]
ax.plot(X_Ground, Y_Ground, color='black', linewidth=3)
#

# Коробки
x1, y1 = rotate(-bx, by, alpha)
x2, y2 = rotate(bx, by, alpha)
x3, y3 = rotate(bx, -by, alpha)
x4, y4 = rotate(-bx, -by, alpha)
X_Box = np.array([x1, x2, x3, x4, x1])
Y_Box = np.array([y1, y2, y3, y4, y1])
#

# Точки А и В
X_A = BoxX / 2 - x + 10
Y_A = X_A * tg
X_B = X_A + l * np.sin(phi)
Y_B = Y_A - l * np.cos(phi)
#

# Плоты
Point_A = ax.plot(X_A[0], Y_A[0], marker='o')[0]
Point_B = ax.plot(X_B[0], Y_B[0], marker='o', markersize=20, color='black')[0]
Box = ax.plot(X_A[0] + X_Box, Y_A[0] + Y_Box, color='black', linewidth='2')[0]
Line_AB = ax.plot([X_A[0], X_B[0]], [Y_A[0], Y_B[0]],
                  color='black', linewidth='2')[0]
#

anima = FuncAnimation(fig, Film, frames=NT, interval=10)

plt.show()