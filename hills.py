import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import numpy as np
import cv2
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim



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
    if image.ndim == 3:
        return rgb2gray(image)
    return image

def PSNR(original, compressed):
    validate_images(original, compressed)
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def SSIM(original, compressed):
    validate_images(original, compressed)
    gray1 = to_grayscale(original)
    gray2 = to_grayscale(compressed)
    score, _ = ssim(gray1, gray2, full=True) #data_range убрана, могут изменится результаты
    return score

def tv_norm(image, eps=1e-8, l=1, p=2):
    x_diff = image[:-1, :-1] - image[:-1, 1:]
    y_diff = image[:-1, :-1] - image[1:, :-1]
    grad_mag = (abs(x_diff**p) + abs(y_diff**p) + eps)**(l/p)
    loss = np.sum(grad_mag)
    dx_diff = x_diff / grad_mag
    dy_diff = y_diff / grad_mag
    grad = np.zeros_like(image, dtype=np.float64)
    grad[:-1, :-1] += dx_diff + dy_diff
    grad[:-1, 1:] -= dx_diff
    grad[1:, :-1] -= dy_diff
    return loss, grad

def lp_norm(image, orig_image, l, p):
    grad = (l/p)*(image - orig_image)
    loss = (np.sum(grad ** p)) ** (l/p)
    return loss, grad

def prox_tv(u, weight, iter=10):
    v = tv_norm(u.copy(),p=1)[1]
    for _ in range(iter):
        loss, grad = tv_norm(v, p=1)
        v = v - weight * grad
    return v

def my_tv_envelope(u, mu=0.001, prox_iter=20):
    v = np.array(prox_tv(u, weight=mu, iter=prox_iter))
    loss, _ = tv_norm(v, p=2)
    envelope = loss + (1 / (2 * mu)) * np.sum((v - u) ** 2)
    grad = v
    return envelope, grad

def eval_loss_and_grad(image, orig_image, strength=0.9, mu=0.01, useMorea=False):
    lp_loss, lp_grad = lp_norm(image, orig_image, 1, 2)
    if useMorea: # if true => MoreaYosida
        tv_loss, tv_grad = my_tv_envelope(image, mu)
        loss =  tv_loss + lp_loss
        grad =  tv_grad + lp_grad
    else: # if false => TV
        tv_loss, tv_grad = tv_norm(image, l=1, p=2)
        loss =  strength *tv_loss + lp_loss
        grad =  strength*tv_grad + lp_grad
    
    return loss, grad

def tv_denoise_gradient_descent(image, strength=0.1, step_size=1e-2, tol=3.2e-3, iter=0,
                                mu=0.01, use=False):
    orig_image = image.copy()
    momentum = np.zeros_like(image)
    momentum_beta = 0.9
    loss_smoothed = 0
    loss_smoothing_beta = 0.9
    i = 0

    if iter == 0:
        while True:
            i += 1
            loss, grad = eval_loss_and_grad(image, orig_image, strength, mu, use)
            loss_smoothed = loss_smoothed * loss_smoothing_beta + loss * (1 - loss_smoothing_beta)
            loss_smoothed_debiased = loss_smoothed / (1 - loss_smoothing_beta ** i)
            if i > 1 and loss_smoothed_debiased / loss < tol + 1:
                break
            step = step_size / (strength + 1)

            # momentum импульс
            # momentum_beta коэфицент к импульсу
            momentum = momentum * momentum_beta + grad * (1 - momentum_beta)
            image -= step / (1 - momentum_beta ** i) * momentum
    else:
        for i in range(1, iter):
            loss, grad = eval_loss_and_grad(image, orig_image, strength, mu, use)
            step = step_size / (strength + 1)
            momentum = momentum * momentum_beta + grad * (1 - momentum_beta)
            image -= step / (1 - momentum_beta ** i) * momentum

    return image


def setup_live_plot(image_shape_tuple):
    """Инициализация окна для живой отрисовки"""
    plt.ion() 
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    stride = 4
    # Правильно извлекаем размеры, даже если картинка не квадратная
    h, w = image_shape_tuple
    z = np.zeros((h // stride, w // stride))
    
    ax.set_zlim(0, 1)
    ax.set_title("TV Denoising Process (Live)")
    ax.axis('off')
    # Возвращаем None в качестве начального объекта surf
    return fig, ax, None, stride

def update_plot(fig, ax, surf, image_data, stride):
    """Обновление данных на графике"""
    # 1. Удаляем старую поверхность, если она существует
    if surf is not None:
        surf.remove()
    
    # 2. Подготавливаем новые данные
    z_low_res = image_data[::stride, ::stride]
    nrows, ncols = z_low_res.shape
    x, y = np.meshgrid(np.arange(ncols), np.arange(nrows))
    
    # 3. Рисуем новую поверхность
    # antialiased=False сильно ускоряет отрисовку в реальном времени
    new_surf = ax.plot_surface(x, y, z_low_res, cmap='terrain', 
                               linewidth=0, antialiased=False)
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001) 
    
    return new_surf # Возвращаем новую поверхность для следующей итерации

def tv_denoise_gradient_descent_live(image, strength=0.1, step_size=1e-2, iter=100, mu=0.01, use=False):
    orig_image = image.copy()
    momentum = np.zeros_like(image)
    momentum_beta = 0.9
    
    fig, ax, surf, stride = setup_live_plot(image.shape)

    for i in range(1, iter + 1):
        loss, grad = eval_loss_and_grad(image, orig_image, strength, mu, use)
        step = step_size / (strength + 1)
        momentum = momentum * momentum_beta + grad * (1 - momentum_beta)
        image -= step / (1 - momentum_beta ** i) * momentum
        
        if i % 2 == 0:
            # ОБЯЗАТЕЛЬНО сохраняем результат в surf, чтобы удалить его на след. шаге
            surf = update_plot(fig, ax, surf, image, stride)
            if i % 10 == 0:
                print(f"Итерация {i}, Loss: {loss:.4f}")

    plt.ioff()
    plt.show()
    return image


# Основной запуск
if __name__ == "__main__":
    image_path = "ships.png"
    
    # Загружаем
    raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if raw_img is None:
        # Если нет файла, создадим тестовый шум
        print("Файл не найден, создаю тестовый шум...")
        image = np.ones((200, 200)) * 0.5
        image += np.random.normal(0, 0.1, image.shape)
    else:
        image = raw_img.astype(np.float32) / 255.0

    # Запускаем живую деноизацию
    # Попробуй поменять use=True на use=False, чтобы увидеть разницу в "плато"
    denoised = tv_denoise_gradient_descent_live(image.copy(), strength=0.1, iter=1000, mu=0.1, use=False)
    
