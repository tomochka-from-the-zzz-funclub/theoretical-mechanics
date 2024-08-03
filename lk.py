import matplotlib.pyplot as plt # для отрисовки
import numpy as np 
import math
from matplotlib.animation import FuncAnimation #подгрузим одну функцию из 
import sympy as sp


def Rot2D(X, Y, Alpha):#поворот в 2-хмерной плоскости(начальные координаты, угол )
    RX = X * np.cos(Alpha) - Y * np.sin(Alpha)#коорднаты после поворота(из матрицы поворота)
    RY = X * np.sin(Alpha) + Y * np.cos(Alpha)#
    return RX, RY#


def anima(j):#функция измения кадров для анимации(подается значение, у обекта меняются координаты на соответствующие житые элементы массивов координат)
    P.set_data(X[j], Y[j])#обновление значений координат
    Vline.set_data([X[j], X[j] + VX[j]], [Y[j], Y[j] + VY[j]])#по аналогии обновляем координаты для скорости 
    #Vline2.set_data([X[j], X[j] + WX[j]], [Y[j], Y[j] + WY[j]])#
    #Vline3.set_data([X_[j], X[j]], [Y_[j], Y[j]])#
    #Vline4.set_data([X[j], X[j] - (Y[j] + VY[j]) * Ro[j]/((Y[j] + VY[j])**2 +
                   # (X[j] + VX[j])**2)**0.5], [Y[j], Y[j] + (X[j] + VX[j]) *
                    # Ro[j]/((Y[j] + VY[j])**2 + (X[j] + VX[j])**2)**0.5])#
    RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[j], VX[j]))#анимация для стрелки(в каждый момент времени разный тангенс)
    VArrow.set_data(RArrowX + X[j] + VX[j], RArrowY + Y[j] + VY[j])#заменяем координаты
    #RArrowWX, RArrowWY = Rot2D(ArrowWX, ArrowWY, math.atan2(WY[j], WX[j]))#
    #WArrow.set_data(RArrowWX + X[j] + WX[j], RArrowWY + Y[j] + WY[j])#
    #RArrowRX, RArrowRY = Rot2D(ArrowRX, ArrowRY, math.atan2(Y[j], X[j]))#
    #RArrow.set_data(RArrowRX + X[j], RArrowRY + Y[j])#
    return P, Vline, VArrow,#Vline2, WArrow, Vline3, Vline4,#

T = np.linspace(1, 15, 1000)#создаем массив времени, линспейс создает последовательность данных: от какого значения, до какого значния пробегает массив, на сколько элементов
t = sp.Symbol('t')#вводим время как символьную переменную
R_ = 4 #дан постоянный радиус 
Omega = 1#постоянная угловая скорость по условию рад/с

#functions
r = 2 + sp.sin(8 * t)#
phi = t + 0.2 * sp.cos(6 * t)#
x = r * sp.cos(phi)#ввели функции 
y = r * sp.sin(phi)#ввели функции 
Vx = sp.diff(x, t)#берем производную от икс по параметру т
#Wx = sp.diff(Vx, t)
Vy = sp.diff(y, t)#аналогично
#Wy = sp.diff(Vy, t)#
#W_ =  sp.sqrt(Wx * Wx + Wy * Wy)#
#W_t = sp.diff(sp.sqrt(Vx**2 + Vy**2),t)#
#ro = (Vx**2 + Vy**2)/sp.sqrt((Wx * Wx + Wy * Wy) - sp.diff(sp.sqrt(Vx**2 + Vy**2), t)**2)

#filling arrays with zeros
#R = np.zeros_like(T)#
#PHI = np.zeros_like(T)#
X = np.zeros_like(T)#для хранения координаты в каждый момент времени, создаем массив из 0 с типом из скобок
Y = np.zeros_like(T)#по аналогии
VX = np.zeros_like(T)#проекции скоростей на оси в каждый момент времени
VY = np.zeros_like(T)#аналогично
#WX = np.zeros_like(T)#
#WY = np.zeros_like(T)#
#W = np.zeros_like(T)#
#W_T = np.zeros_like(T)#
#o = np.zeros_like(T)#
#X_ = [0 for i in range(1000)]#
#Y_ = [0 for i in range(1000)]#


#filling arrays
for i in np.arange(len(T)):#пробегаем длину Т(эрендж возвращает одномерный массив с равномерно разнесенными значениями внутри него, на вход подаем количество элементов)
    #R[i] = sp.Subs(r, t, T[i])#
    #PHI[i] = sp.Subs(phi, t, T[i])#
    X[i] = sp.Subs(x, t, T[i])#заполняем на вход подаем функцию, где будем заменять параметр, какой параметр заменять,и на что будем менять
    Y[i] = sp.Subs(y, t, T[i])#аналогично
    VX[i] = sp.Subs(Vx, t, T[i])#в функции проекции скорости параметр т меняем на т итый
    VY[i] = sp.Subs(Vy, t, T[i])#аналогично
    #WX[i] = sp.Subs(Wx, t, T[i])#
    #WY[i] = sp.Subs(Wy, t, T[i])#
    #W[i] = sp.Subs(W_, t, T[i])#
    #W_T[i] = sp.Subs(W_t, t, T[i])#
    #Ro[i] = sp.Subs(ro, t, T[i])#

#drawing
fig = plt.figure()#создаем область для отрисовки(окно)
ax1 = fig.add_subplot(1, 1, 1)# создаем в окне ячейку(в первую строку, в первый столбец, в единственную ячейку)
ax1.axis('equal')#определяем равенство осей
ax1.set(xlim=[-R_, R_], ylim=[-R_, R_])# 
ax1.plot(X, Y)#отрисуем траекторию(указываем, где будем отрисовывать)б передаем координаты точки в каждый момент времени
P, = ax1.plot(X[0], Y[0], 'r', marker='o')#создаем объект отрисовки точки в начальный момент времени(координаты точки,задаем цвет ,задаем вид точки)
Vline, = ax1.plot([X[0], X[0] + VX[0]], [Y[0], Y[0] + VY[0]], 'r')  # vector of speed линия скорости(передаем коордиаты начала и конци по икс и игрик, линия  красного цвета)
#Vline2, = ax1.plot([X[0], X[0] + WX[0]], [Y[0], Y[0] + WY[0]], 'g')  # vector of acceleration
#Vline3, = ax1.plot([X_[0], X[0]], [Y_[0], Y[0]], 'b')  # vector of radius vector
#Vline4, = ax1.plot([X[0], X[0] - (Y[0] + VY[0]) * Ro[0]/((Y[0] + VY[0])**2 +
        # (X[0] + VX[0])**2)**0.5], [Y[0], Y[0] + (X[0] + VX[0]) * Ro[0]/
          #                          ((Y[0] + VY[0])**2 + (X[0] + VX[0])**2)**0.5], 'm')  # vector of radius of curvature

ArrowX = np.array([-0.2 * R_, 0, -0.2 * R_])  # arrow of speed коордиаты для стрелки
ArrowY = np.array([0.1 * R_, 0, -0.1 * R_])  
#ArrowWX = np.array([-R_, 0, -R_])  # arrow of acceleration
#ArrowWY = np.array([R_, 0, -R_])#
#ArrowRX = np.array([-0.1 * R_, 0, -0.1 * R_])  # arrow of radius vector
#ArrowRY = np.array([0.05 * R_, 0, -0.05 * R_])

# drawing an arrow at the end of a vector
RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[0], VX[0]))#кординаты стрелки после поворота, функция вернет нам 2 массива
#RArrowWX, RArrowWY = Rot2D(ArrowWX, ArrowWY, math.atan2(WY[0], WX[0]))#
#RArrowRX, RArrowRY = Rot2D(ArrowRX, ArrowRY, math.atan2(Y[0], X[0]))#
VArrow, = ax1.plot(RArrowX + X[0] + VX[0], RArrowY + Y[0] + VY[0], 'r')#отрисовка конца стрелочки
#WArrow, = ax1.plot(RArrowWX + X[0] + WX[0], RArrowY + Y[0] + WY[0], 'g')#
#RArrow, = ax1.plot(ArrowRX + X[0], ArrowRY + Y[0], 'b')#

anim = FuncAnimation(fig, anima, frames=1000, interval=2, blit=True, repeat=False)#(где будет отображаться анимация, функция обновления кадров, количество кадров, интервал в милисекундах(насколько быстро будет меняться), цикличность выполнения(включаем повторение))

plt.show()#покзать отрисовку