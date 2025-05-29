import os
import cv2
import numpy as np
from tqdm import tqdm


def add_gaussian_noise(image, amplitude):
    """
    Добавляет гауссов шум к изображению.
    amplitude: уровень шума в % (от 0 до 100)
    """
    row, col, ch = image.shape
    mean = 0
    sigma = (amplitude / 100) * 255  # Преобразуем амплитуду в диапазон 0-255
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)


def add_salt_pepper_noise(image, amplitude):
    """
    Добавляет шум "соль-перец" к изображению.
    amplitude: уровень шума в % (от 0 до 100)
    """
    row, col, ch = image.shape
    s_vs_p = 0.5
    amount = amplitude / 200  # Делим на 200, так как суммарно соль и перец
    noisy = np.copy(image)

    # Соль
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
    noisy[coords[0], coords[1], :] = 255

    # Перец
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
    noisy[coords[0], coords[1], :] = 0

    return noisy


def process_images(input_folder, output_folder):
    # Создаем папку для результатов, если ее нет
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Получаем список изображений в папке
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.ppm', '.pgm'))]

    for image_file in tqdm(image_files, desc="Обработка изображений"):
        # Загружаем изображение
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Не удалось загрузить изображение: {image_file}")
            continue

        # Создаем папку для текущего изображения
        image_name = os.path.splitext(image_file)[0]
        image_output_folder = os.path.join(output_folder, image_name)
        if not os.path.exists(image_output_folder):
            os.makedirs(image_output_folder)

        # Применяем шумы с разными амплитудами
        for amplitude in range(5, 101, 5):
            # Гауссов шум
            gaussian_noisy = add_gaussian_noise(image, amplitude)
            gaussian_output_path = os.path.join(image_output_folder, f"{image_name}_gaussian_{amplitude}.png")
            cv2.imwrite(gaussian_output_path, gaussian_noisy)

            # Шум "соль-перец"
            saltpepper_noisy = add_salt_pepper_noise(image, amplitude)
            saltpepper_output_path = os.path.join(image_output_folder, f"{image_name}_soltpepper_{amplitude}.png")
            cv2.imwrite(saltpepper_output_path, saltpepper_noisy)


if __name__ == "__main__":
    input_folder = 'clear_images'  # Папка с исходными изображениями
    output_folder = "result_noise"  # Папка для результатов

    process_images(input_folder, output_folder)
    print("Обработка завершена!")