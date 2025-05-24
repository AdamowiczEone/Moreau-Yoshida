import numpy as np
import cv2
from dataclasses import dataclass
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim

# Проверка изображений
def validate_images(original, compressed):
    if original is None or compressed is None:
        raise ValueError("Одно или оба изображения не загружены")
    if original.size == 0 or compressed.size == 0:
        raise ValueError("Изображения не должны быть пустыми")
    if original.dtype != np.uint8 or compressed.dtype != np.uint8:
        raise TypeError("Оба изображения должны быть 8-битными")
    if original.shape != compressed.shape:
        raise ValueError("Изображения должны быть одинакового размера")

def to_grayscale(image):
    if image.ndim == 3 and image.shape[2] == 3:
        return rgb2gray(image)
    else:
        return image

def PSNR(original, compressed):
    validate_images(original, compressed)
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def SSIM(original, compressed):
    validate_images(original, compressed)
    gray1 = to_grayscale(original)
    gray2 = to_grayscale(compressed)
    score, _ = ssim(gray1, gray2, full=True, data_range=1.0)
    return score

# Прямая TV-норма
def tv_norm(image, eps=1e-8, l=1, p=2):
    x_diff = image[:-1, :-1, ...] - image[:-1, 1:, ...]
    y_diff = image[:-1, :-1, ...] - image[1:, :-1, ...]
    grad_mag = (abs(x_diff**p) + abs(y_diff**p) + eps)**(l/p)
    loss = np.sum(grad_mag)
    dx_diff = x_diff / grad_mag
    dy_diff = y_diff / grad_mag
    grad = np.zeros_like(image, dtype=np.float64)
    grad[:-1, :-1, ...] = dx_diff + dy_diff
    grad[:-1, 1:, ...] -= dx_diff
    grad[1:, :-1, ...] -= dy_diff
    return loss, grad

# Lp-норма
def lp_norm(image, orig_image, l, p):
    grad = image - orig_image
    loss = (np.sum(grad ** p)) ** (l/p)
    return loss, grad

# Проксимальный оператор TV (итеративно)
def prox_tv(u, weight, iter=20):
    v = u.copy()
    for _ in range(iter):
        loss, grad = tv_norm(v,p=1)
        v = v - weight * grad
    return v

# Оболочка Мороу-Йошиды от TV-нормы
def my_tv_envelope(u, mu=0.01, prox_iter=20):
    v = prox_tv(u, weight=mu, iter=prox_iter)
    loss, _ = tv_norm(v)
    envelope = loss + (1 / (2 * mu)) * np.sum((v - u) ** 2)
    grad = (u - v) / mu
    return envelope, grad

# Основная функция потерь
def eval_loss_and_grad(image, orig_image, strength_luma=0.9, strength_chroma=0.9, mu=0.01, use_morozov=False):
    if use_morozov:
        tv_loss_y, tv_grad_y = my_tv_envelope(image[:, :, 0], mu)
        tv_loss_uv, tv_grad_uv = my_tv_envelope(image[:, :, 1:], mu)
    else:
        tv_loss_y, tv_grad_y = tv_norm(image[:, :, 0], l=1, p=2)
        tv_loss_uv, tv_grad_uv = tv_norm(image[:, :, 1:], l=1, p=2)

    tv_grad = np.zeros_like(image)
    tv_grad[..., 0] = tv_grad_y * strength_luma
    tv_grad[..., 1:] = tv_grad_uv * strength_chroma

    lp_loss, lp_grad = lp_norm(image, orig_image, 1, 2)
    loss = tv_loss_y * strength_luma + tv_loss_uv * strength_chroma + lp_loss
    grad = tv_grad + lp_grad
    return loss, grad

# Основной градиентный спуск
def tv_denoise_gradient_descent(image, strength_luma, strength_chroma,
                                step_size=1e-2, tol=3.2e-3, iter=0,
                                mu=0.01, use_morozov=False):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    orig_image = image.copy()
    momentum = np.zeros_like(image)
    momentum_beta = 0.9
    loss_smoothed = 0
    loss_smoothing_beta = 0.9
    i = 0

    if iter == 0:
        while True:
            i += 1
            loss, grad = eval_loss_and_grad(image, orig_image, strength_luma, strength_chroma, mu, use_morozov)
            loss_smoothed = loss_smoothed * loss_smoothing_beta + loss * (1 - loss_smoothing_beta)
            loss_smoothed_debiased = loss_smoothed / (1 - loss_smoothing_beta ** i)
            if i > 1 and loss_smoothed_debiased / loss < tol + 1:
                break
            step_size_luma = step_size / (strength_luma + 1)
            step_size_chroma = step_size / (strength_chroma + 1)
            step_size_arr = np.float32([[[step_size_luma, step_size_chroma, step_size_chroma]]])
            momentum *= momentum_beta
            momentum += grad * (1 - momentum_beta)
            image -= step_size_arr / (1 - momentum_beta ** i) * momentum

    else:
        for i in range(1, iter):
            loss, grad = eval_loss_and_grad(image, orig_image, strength_luma, strength_chroma, mu, use_morozov)
            step_size_luma = step_size / (strength_luma + 1)
            step_size_chroma = step_size / (strength_chroma + 1)
            step_size_arr = np.float32([[[step_size_luma, step_size_chroma, step_size_chroma]]])
            momentum *= momentum_beta
            momentum += grad * (1 - momentum_beta)
            image -= step_size_arr / (1 - momentum_beta ** i) * momentum

    return cv2.cvtColor(image, cv2.COLOR_YUV2RGB)

# Основной запуск
if __name__ == "__main__":
    image_path = "cam.png"
    output_tv = "denoised_tv.jpg"
    output_my = "denoised_my.jpg"

    image = cv2.imread(image_path).astype(np.float32) / 255

   # denoised_tv = tv_denoise_gradient_descent(image.copy(), 0.1, 0.1, iter=100, use_morozov=False)
    denoised_my = tv_denoise_gradient_descent(image.copy(), 0.1, 0.1, iter=300, mu=0.009, use_morozov=True)

  #  cv2.imwrite(output_tv, (denoised_tv * 255).clip(0, 255).astype(np.uint8))
    cv2.imwrite(output_my, (denoised_my * 255).clip(0, 255).astype(np.uint8))

  #  print("Saved denoised_tv.jpg and denoised_my.jpg")

  #  original = (image * 255).astype(np.uint8)
  #  tv_out = (denoised_tv * 255).astype(np.uint8)
    my_out = (denoised_my * 255).astype(np.uint8)

   # print(f"TV → PSNR: {PSNR(original, tv_out):.2f}, SSIM: {SSIM(original, tv_out):.4f}")
    print(f"Moreau–Yosida → PSNR: {PSNR(original, my_out):.2f}, SSIM: {SSIM(original, my_out):.4f}")
