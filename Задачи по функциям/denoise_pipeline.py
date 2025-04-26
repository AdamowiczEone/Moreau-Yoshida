import numpy as np
import matplotlib.pyplot as plt
import cv2

def PSNR(original, compressed):
    # diff = (original - compressed) ** 2
    # mse = diff.sum() / diff.size
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# def main(): 
#     original = cv2.imread("img_with_noise.jpg") 
#     compressed = cv2.imread("img_without_noise.jpg", 1) 
#     compressed = cv2.resize(compressed, (original.shape[1], original.shape[0]))
#     value = PSNR(original, compressed) 
#     #  print(original.shape)
#     #  print(compressed.shape)
#     print(f"PSNR value is {value} dB")  
       
# if __name__ == "__main__": 
#     main() 

def run_denoise_pipeline(img, noise_function, denoise_function, accurate_metric):
    # Добавление шума
    noisy_img = noise_function(img)
    
    # Применение алгоритма удаления шума
    denoised_img = denoise_function(noisy_img)
    
    # Вычисление метрики
    metric_value = accurate_metric(img, denoised_img)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB))
    plt.title("Noisy Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB))
    plt.title("Denoised Image")
    plt.axis("off")
    
    plt.suptitle(f"Metric Value: {metric_value:.2f}")
    plt.show()


def add_noise(img):
    noise = np.random.normal(0, 2, img.shape).astype(np.uint8) #random return float => uint8
    noisy_img = cv2.add(img, noise)
    return noisy_img


# ?? какой именно алгоритм удаления шума использовать?
def denoise(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

if __name__ == "__main__":
    img = cv2.imread("img_without_noise.jpg")
    run_denoise_pipeline(
        img=img,
        noise_function=add_noise,
        denoise_function=denoise,
        accurate_metric=PSNR
    )

    """
    Универсальная функция прогона удаления шума 
    (np.ndarray img, function noise_function,  function denoise_function, function accurate_metric) 
    Функция вызывает прогон действий 
    1) Применение шума к img 
    2) Алгоритм удаления шума 
    3) показ результата, с 3 картинками(исходная img, img+noise_function, img после denoise_function ) 
    и метрикой результат
    """