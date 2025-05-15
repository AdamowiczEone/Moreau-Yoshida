import numpy as np
import cv2
from dataclasses import dataclass
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim

# Проверка входных изображений
def validate_images(original, compressed):
    if original is None or compressed is None:
        raise ValueError("Одно или оба изображения не загружены")
    if original.size == 0 or compressed.size == 0:
        raise ValueError("Изображения не должны быть пустыми")
    if original.dtype != np.uint8 or compressed.dtype != np.uint8:
        raise TypeError("Оба изображения должны быть 8-битными")
    if original.shape != compressed.shape:
        raise ValueError("Изображения должны быть одинакового размера")

# Функция преобразования в оттенки серого
def to_grayscale(image):
    if image.ndim == 3 and image.shape[2] == 3:
        return rgb2gray(image)
    else:
        return image  # уже ч/б

def PSNR(original, compressed):
    validate_images(original, compressed)
    
    # Вычисление MSE
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    
    # Вычисление PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def SSIM(original, compressed):
    validate_images(original, compressed)
    
    # Преобразование в оттенки серого
    gray1 = to_grayscale(original)
    gray2 = to_grayscale(compressed)
    
    # Вычисление SSIM
    score, _ = ssim(gray1, gray2, full=True, data_range=1.0)
    return score







RGB_TO_YUV = np.float32([
    [0.2126, 0.7152, 0.0722],
    [-0.09991, -0.33609, 0.436],
    [0.615, -0.55861, -0.05639],
])

YUV_TO_RGB = np.linalg.inv(RGB_TO_YUV)

def f_abs(x):
  return np.abs(x)

# 5. Функция Хубера (смесь L₁ и L₂)
def f_huber(x, delta=1.0):
    """Функция Хубера (устойчива к выбросам)"""
    abs_x = np.abs(x)
    return np.sum(np.where(abs_x <= delta, 0.5 * x**2, delta * (abs_x - 0.5 * delta)))

# 6. Логистическая функция
def f_logistic(x):
    """Логистическая функция (для классификации)"""
    return np.sum(np.log(1 + np.exp(-x)))

# 7. Функция квадрата с порогом
def f_square_threshold(x, threshold=0.5):
    """Квадратичная функция с порогом активации"""
    return np.sum(np.where(np.abs(x) < threshold, x**2, 2*threshold*np.abs(x) - threshold**2))

# 8. Нулевая функция (для тестирования)
def f_zero(x):
    """Просто возвращает 0 (проксимальный оператор = вход)"""
    return 0

# 9. "Гребневая" функция
def f_ridge(x, epsilon=0.1):
    """Модифицированная L₂-норма с добавкой для устойчивости"""
    return np.sum(x**2 / (np.abs(x) + epsilon))
#%%


def lp_norm(image,orig_image, l, p):
    if p < 1:
        raise ValueError("Параметр p должен быть больше или равен 1.")
    grad = image - orig_image
    # Вычисляем сумму абсолютных значений элементов, возведённых в степень p
    loss = (np.sum(np.abs(image) ** p))**(l/p)
    # Возвращаем результат, возводя сумму в степень 1/p
    return loss, grad







#loss- используем как метрику, grad - градиент 
def tv_norm(image, eps=1e-8):
  
    x_diff = image[:-1, :-1, ...] - image[:-1, 1:, ...]
    y_diff = image[:-1, :-1, ...] - image[1:, :-1, ...]
    grad_mag = np.sqrt(x_diff**2 + y_diff**2 + eps) # магнитуды, градиент в контексте изображения
    loss = np.sum(grad_mag)
    dx_diff = x_diff / grad_mag
    dy_diff = y_diff / grad_mag
    grad = np.zeros_like(image)
    grad[:-1, :-1, ...] = dx_diff + dy_diff
    grad[:-1, 1:, ...] -= dx_diff
    grad[1:, :-1, ...] -= dy_diff
    return loss, grad


def eval_loss_and_grad(image, orig_image, strength_luma=0.9, strength_chroma=0.9):
   
    tv_loss_y, tv_grad_y = tv_norm(image[:, :, 0])
    tv_loss_uv, tv_grad_uv = tv_norm(image[:, :, 1:])
    tv_grad = np.zeros_like(image)
    tv_grad[..., 0] = tv_grad_y * strength_luma 
    tv_grad[..., 1:] = tv_grad_uv * strength_chroma
    l2_loss, l2_grad = lp_norm(image, orig_image,1,2)
    loss = tv_loss_y * strength_luma + tv_loss_uv * strength_chroma + l2_loss
    grad = tv_grad + l2_grad
    return loss, grad





def tv_denoise_gradient_descent(image, strength_luma, strength_chroma, step_size=1e-2, tol=3.2e-3):
    image = image @ RGB_TO_YUV.T
    orig_image = image.copy()
    momentum = np.zeros_like(image)
    momentum_beta = 0.9
    loss_smoothed = 0
    loss_smoothing_beta = 0.9
    i = 0
    while True:
        i += 1
        loss, grad = eval_loss_and_grad(image, orig_image, strength_luma, strength_chroma)
        loss_smoothed = loss_smoothed * loss_smoothing_beta + loss * (1 - loss_smoothing_beta)
        loss_smoothed_debiased = loss_smoothed / (1 - loss_smoothing_beta**i)
        if i > 1 and loss_smoothed_debiased / loss < tol + 1:
            break
        step_size_luma = step_size / (strength_luma + 1)
        step_size_chroma = step_size / (strength_chroma + 1)
        step_size_arr = np.float32([[[step_size_luma, step_size_chroma, step_size_chroma]]])
        momentum *= momentum_beta
        momentum += grad * (1 - momentum_beta)
        image -= step_size_arr / (1 - momentum_beta**i) * momentum
    return image @ YUV_TO_RGB.T


if __name__ == "__main__":
    image_path = "shum.jpg"
    output_path = "denoised_image.jpg"
    image = cv2.imread(image_path).astype(np.float32) / 255
    denoised_image = tv_denoise_gradient_descent(image, 0.1, 0.1)
    cv2.imwrite(output_path, (denoised_image * 255).clip(0, 255).astype(np.uint8))
    print(f"Denoised image saved to {output_path}")
