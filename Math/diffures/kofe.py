import sympy as sp
from sympy import symbols, lambdify, exp
import numpy as np

# Определяем все символы (теперь c и m разделены)
theta, c1, m1, c, m, T1, T0, s, l, t, n = symbols('theta c1 m1 c m T1 T0 s l t n')

# Формула Анатолия с раздельными c и m
exprA = theta + (((T1*(c1*m1)/(c*m + c1*m1)) + (T0*(c*m)/(c*m + c1*m1)) - theta) * exp( t*(-n*s)/(l*(c*m + c1*m1)) ))

# Формула Владимира (вручную переведенная из LaTeX в SymPy)
exprV = (c1 * m1) / (c * m + c1 * m1) * T1 + (c * m) / (c * m + c1 * m1) * (
            theta + (T0 - theta) * exp(-n*s / (l * c * m) * t))

# Создаем функции
funcA = lambdify((theta, c1, m1, c, m, T1, T0, s, l, t, n), exprA, modules='numpy')
funcV = lambdify((c1, m1, c, m, T1, theta, T0, s, l, t, n), exprV, modules='numpy')

# Тестовые параметры (c и m заданы отдельно)
params = {
    'theta': 20,      # температура воздуха в кафе
    'c1': 3.9,        # удельная теплоемкость сливок [кДж/(кг·K)]
    'm1': 0.02,        # масса сливок [кг]
    'c': 4.2,         # удельная теплоемкость воды [кДж/(кг·K)]
    'm': 0.08,         # масса воды [кг]
    'T1': 20,         # температура сливок  [°C]
    'T0': 80,         # начальная температура воды [°C]
    's': 0.011,        # площадь боковой поверхности [м²]
    'l': 0.002,         # толщина стенок [м]
    'n': 0.6,         # теплопроводность материала
    't': 1          # время [с]
}

# Вычисляем
try:
    resultA = funcA(
        params['theta'], params['c1'], params['m1'], params['c'], params['m'],
        params['T1'], params['T0'], params['s'], params['l'], params['t'], params['n']
    )
    resultV = funcV(
        params['c1'], params['m1'], params['c'], params['m'], params['T1'],
        params['theta'], params['T0'], params['s'], params['l'], params['t'], params['n']
    )

    print(f"Результат формулы Анатолия: {resultA:.2f} °C")
    print(f"Результат формулы Владимира: {resultV:.2f} °C")

except Exception as e:
    print(f"Ошибка при вычислении: {e}")