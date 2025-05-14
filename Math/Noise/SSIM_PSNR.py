import numpy as np
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

# Загрузка и вычисление
original = imread("original_image.png") 
compressed = imread("compressed_image1.png")

print(f"PSNR value is {PSNR(original, compressed)} dB") 
print(f"SSIM value is {SSIM(original, compressed)}") 