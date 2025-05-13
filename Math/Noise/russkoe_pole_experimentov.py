import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import numpy as np
from scipy.optimize import minimize

from PIL import Image
import numpy as np


def add_noise_to_image(image_array, noise_type='gaussian', **kwargs):
    """
    Добавляет шум к изображению.

    Параметры:
    - image_array: numpy array изображения (нормализовано [0, 1])
    - noise_type: тип шума ('gaussian' или 'salt_pepper')
    - kwargs: параметры шума:
        - sigma (для гауссовского шума)
        - prob (для шума соль-перец)

    Возвращает:
    - Зашумлённое изображение (numpy array)
    """
    if noise_type == 'gaussian':
        sigma = kwargs.get('sigma', 0.1)
        noise = np.random.normal(0, sigma, image_array.shape)
        noisy_image = image_array + noise
        return np.clip(noisy_image, 0, 1)

    elif noise_type == 'salt_pepper':
        prob = kwargs.get('prob', 0.05)
        noisy_image = np.copy(image_array)
        mask = np.random.random(image_array.shape) < prob
        noisy_image[mask] = np.random.randint(0, 2, size=np.sum(mask))
        return noisy_image

    else:
        raise ValueError("Поддерживаемые типы шума: 'gaussian', 'salt_pepper'")


def image_to_ascii(image_data, width=100, chars=" .,:;+*?%S#@"):
    """
    Преобразует изображение (numpy array или путь к файлу) в ASCII-арт.

    Параметры:
    - image_data: numpy.ndarray (нормализованный [0,1]) ИЛИ путь к изображению
    - width: ширина ASCII-арта
    - chars: градации яркости
    """
    # Если передан путь к файлу
    if isinstance(image_data, str):
        img = Image.open(image_data).convert("L")
        img_array = np.array(img, dtype=np.float32) / 255.0
    # Если передан numpy array
    elif isinstance(image_data, np.ndarray):
        img_array = image_data.copy()
        # Нормализуем, если значения не в [0,1]
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
    else:
        raise ValueError("Неподдерживаемый тип данных. Ожидается путь или numpy array")

    # Пропорции и масштабирование
    aspect_ratio = img_array.shape[0] / img_array.shape[1]
    height = int(width * aspect_ratio * 0.5)

    # Для масштабирования используем PIL
    img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
    img_pil = img_pil.resize((width, height))
    pixels = np.array(img_pil, dtype=np.float32) / 255.0

    # Генерация ASCII
    normalized = (pixels * (len(chars) - 1)).astype(int)
    ascii_art = "\n".join("".join(chars[pixel] for pixel in row) for row in normalized)

    print(ascii_art)

# Пример использования

image_file_path = "logos.png"
image_path = image_file_path
original_image = Image.open(image_path).convert("L").resize((100,100))  # Чёрно-белое
original_array = np.array(original_image, dtype=np.float32) / 255.0  # Нормализуем [0, 1]
image_to_ascii(original_image, width=100)
# Добавляем разные типы шума
noisy_gaussian = add_noise_to_image(original_array,
                                   noise_type='gaussian',
                                   sigma=0.1)

noisy_salt_pepper = add_noise_to_image(original_array,
                                      noise_type='salt_pepper',
                                      prob=0.05)
image_to_ascii(noisy_gaussian)
def total_variation_denoising(v, lambda_param, max_iter=1000, tol=1e-6):
    """
    Реализация ROF модели (Rudin-Osher-Fatemi) для удаления шума с progress bar.

    Параметры:
    v : numpy.ndarray
        Зашумленное изображение (2D массив).
    lambda_param : float
        Параметр регуляризации (λ).
    max_iter : int, optional
        Максимальное количество итераций (по умолчанию 1000).
    tol : float, optional
        Критерий остановки по изменению значения функционала (по умолчанию 1e-6).

    Возвращает:
    numpy.ndarray
        Очищенное изображение (u).
    """
    u = v.copy()
    height, width = v.shape

    # Инициализация progress bar
    pbar = tqdm(total=max_iter, desc='TV Denoising Progress', unit='iter')

    # Функция для вычисления градиента
    def gradient(u):
        grad_x = np.roll(u, -1, axis=1) - u
        grad_y = np.roll(u, -1, axis=0) - u
        return grad_x, grad_y

    # Callback функция для обновления progress bar
    def callback(xk):
        pbar.update(1)
        pbar.refresh()  # Принудительное обновление

    # Функционал для минимизации
    def functional(u_flat):
        u = u_flat.reshape((height, width))
        grad_x, grad_y = gradient(u)
        tv_term = np.sum(np.sqrt(grad_x**2 + grad_y**2))
        fidelity_term = 0.5 * np.sum((v - u)**2)
        return tv_term + lambda_param * fidelity_term

    try:
        # Оптимизация с callback
        result = minimize(functional,
                        u.flatten(),
                        method='L-BFGS-B',
                        callback=callback,
                        options={'maxiter': max_iter, 'gtol': tol})
    finally:
        pbar.close()  # Закрываем progress bar в любом случае

    return result.x.reshape((height, width))


print("\n\n\n\n")
image_to_ascii(total_variation_denoising(noisy_gaussian, lambda_param=0.5, max_iter=5, tol=1e-6), width=100)

