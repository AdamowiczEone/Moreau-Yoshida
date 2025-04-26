import numpy as np

def lp_norm(v, p):
    if p <= 0:
        raise ValueError("Параметр p должен быть положительным числом.")

    sum_abs_p = np.sum(np.abs(v) ** p)

    return sum_abs_p ** (1 / p)

def total_variation(u, p):
    if p < 1:
        raise ValueError("Параметр p должен быть больше или равен 1.")

    dx = np.abs(np.diff(u, axis=1))[:-1, :]
    dy = np.abs(np.diff(u, axis=0))[:, :-1]

    tv_sum = np.sum((dx ** p) + (dy ** p))

    return tv_sum ** (1 / p)

if __name__ == "__main__":
    vector = np.array([3.0, -1.0, 2.0, 1.0])

    image = np.random.rand(10, 7)

    l1_norm = lp_norm(vector, 1)
    print(f"L1-норма: {l1_norm}")

    l2_norm = lp_norm(vector, 2)
    print(f"L2-норма: {l2_norm}")

    tv_l1 = total_variation(image, 1)
    print(f"Полная вариация с L1-нормой: {tv_l1}")

    tv_l2 = total_variation(image, 2)
    print(f"Полная вариация с L2-нормой: {tv_l2}")