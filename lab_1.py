import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.animation import FuncAnimation

t, T = sp.Symbol('t'), np.linspace(1, 15, 1000)

r, phi = 2 + sp.cos(6 * t), 7 * t + 1.2 * sp.cos(6 * t)
x, y = r * sp.cos(phi), r * sp.sin(phi)

Vx, Vy = sp.diff(x, t), sp.diff(y, t)  # скорость
Wx, Wy = sp.diff(Vx, t), sp.diff(Vy, t)  # ускорение
V, W = sp.sqrt(Vx ** 2 + Vy ** 2), sp.sqrt(Wx ** 2 + Wy ** 2)  # квадратный корень из составляющих по координатам

F_func = [sp.lambdify(t, i) for i in [x, y, Vx, Vy, Wx, Wy]]
[X, Y, Vx, Vy, Wx, Wy] = [func(T) for func in F_func]

fig = plt.figure() 
ax = fig.add_subplot(1, 1, 1)
ax.axis('equal'), ax.set_title("Модель движения точки"), ax.set_xlabel('x'), ax.set_ylabel('y'), ax.plot(X, Y), ax.set(xlim=[-5, 5], ylim=[-5, 5])

P = ax.plot(X[0], Y[0], marker='*')[0]
kf, flag = 0.1, True


def animation(i):
    P.set_data(X[i], Y[i])
    VLine = ax.arrow(X[i], Y[i], kf * Vx[i], kf * Vy[i], width=0.02, color='darkred', label='- скорость')  # Вектор скорости
    WLine = ax.arrow(X[i], Y[i], kf * Wx[i], kf * Vy[i], width=0.02, color='indianred', label='- ускорение')  # Вектор ускорения

    CVector = ax.arrow(X[i], Y[i], - kf * ((Vy[i] * (Vx[i] ** 2 + Vy[i] ** 2)) / (Vx[i] * Wy[i] - Wx[i] * Vy[i])),
                       kf * ((Vx[i] * (Vx[i] ** 2 + Vy[i] ** 2)) / (Vx[i] * Wy[i] - Wx[i] * Vy[i])),
                       width=0.03, color="r", label='- кривизна')  # Вектор кривизны
    global flag
    if flag:
        ax.legend(ncol=2,  # количество столбцов
                  facecolor='oldlace',  # цвет области
                  edgecolor='r',  # цвет крайней линии
                  title='обозначение векторов',  # заголовок
                  title_fontsize='10')  # размер заголовка
        flag = False

    return P, VLine, WLine, CVector

anim = FuncAnimation(fig, animation, frames=len(T), interval=10, blit=True)

plt.show()