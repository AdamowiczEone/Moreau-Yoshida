import os
import numpy as np
import cv2
from skimage.io import imread
import pandas as pd
import glob
from MoreaYoshida import tv_denoise_gradient_descent, PSNR, SSIM

def add_gaussian_noise(image, mean=0, sigma=0.05):
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1).astype(np.float32)


def add_salt_pepper_noise(image, amount=0.05):
    out = np.copy(image)
    s_vs_p = 0.5
    num_salt = np.ceil(amount * image.size * s_vs_p).astype(int)
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p)).astype(int)
    
    coords = [np.random.randint(0, i, num_salt) for i in image.shape]
    out[coords[0], coords[1], :] = 1.0
    
    coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
    out[coords[0], coords[1], :] = 0.0
    
    return out.astype(np.float32)


# img .bmp, .ppm, .pgm
def load_image(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    image = imread(image_path)
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image.astype(np.float32) / 255


def save_denoised_image(image, output_path):
    cv2.imwrite(output_path, (image * 255).clip(0, 255).astype(np.uint8))


def test_denoising(dataset_dir, output_dir, luma_values, chroma_values):
    os.makedirs(output_dir, exist_ok=True)
    
    extensions = ['*.bmp', '*.ppm', '*.pgm']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(dataset_dir, ext)))
    
    results = []
    
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        original_image = load_image(image_path)
        
        noise_types = ['salt_pepper', 'gaussian']
        noisy_images = {
            'salt_pepper': add_salt_pepper_noise(original_image),
            'gaussian': add_gaussian_noise(original_image)
        }
        
        for noise_type in noise_types:
            noisy_image = noisy_images[noise_type]
            
            for strength_luma in luma_values:
                for strength_chroma in chroma_values:
                    # TV denoising
                    denoised_tv_image = tv_denoise_gradient_descent(
                        noisy_image.copy(), strength_luma, strength_chroma, use_morozov=False
                    )
                    
                    original_uint8 = (original_image * 255).astype(np.uint8)
                    denoised_tv_uint8 = (denoised_tv_image * 255).clip(0, 255).astype(np.uint8)
                    
                    psnr_tv_value = PSNR(original_uint8, denoised_tv_uint8)
                    ssim_tv_value = SSIM(original_uint8, denoised_tv_uint8)
                    
                    output_filename_tv = (
                        f"{os.path.splitext(image_name)[0]}_{noise_type}"
                        f"_luma{strength_luma:.2f}_chroma{strength_chroma:.2f}_TV.png"
                    )
                    output_path_tv = os.path.join(output_dir, output_filename_tv)
                    save_denoised_image(denoised_tv_image, output_path_tv)
                    
                    results.append({
                        'image': image_name,
                        'noise_type': noise_type,
                        'denoising_method': 'TV',
                        'strength_luma': strength_luma,
                        'strength_chroma': strength_chroma,
                        'psnr': psnr_tv_value,
                        'ssim': ssim_tv_value,
                        'output_path': output_path_tv
                    })

                    # Moreau-Yosida denoising
                    denoised_my_image = tv_denoise_gradient_descent(
                        noisy_image.copy(), strength_luma, strength_chroma, use_morozov=True
                    )
                    
                    denoised_my_uint8 = (denoised_my_image * 255).clip(0, 255).astype(np.uint8)
                    
                    psnr_my_value = PSNR(original_uint8, denoised_my_uint8)
                    ssim_my_value = SSIM(original_uint8, denoised_my_uint8)
                    
                    output_filename_my = (
                        f"{os.path.splitext(image_name)[0]}_{noise_type}"
                        f"_luma{strength_luma:.2f}_chroma{strength_chroma:.2f}_MoreauYosida.png"
                    )
                    output_path_my = os.path.join(output_dir, output_filename_my)
                    save_denoised_image(denoised_my_image, output_path_my)
                    
                    results.append({
                        'image': image_name,
                        'noise_type': noise_type,
                        'denoising_method': 'Moreau-Yosida',
                        'strength_luma': strength_luma,
                        'strength_chroma': strength_chroma,
                        'psnr': psnr_my_value,
                        'ssim': ssim_my_value,
                        'output_path': output_path_my
                    })
    
    results_df = pd.DataFrame(results)
    # округление psnr и ssim до 4 знаков после запятой
    # results_df['psnr'] = results_df['psnr'].round(4)
    # results_df['ssim'] = results_df['ssim'].round(4)

    csv_path = os.path.join(output_dir, 'denoising_results.csv')
    results_df.to_csv(csv_path, index=False)
    
    return results_df


if __name__ == "__main__":
    dataset_dir = "dataset/STI/..."
    output_dir = "output/..."
    luma_values = [0.1, 0.9]   # val for tests
    chroma_values = [0.1, 0.9] # val for tests
    
    results = test_denoising(dataset_dir, output_dir, luma_values, chroma_values)
    
    print("\nСводка результатов:")
    print(results.groupby(['denoising_method', 'noise_type', 'strength_luma', 'strength_chroma'])[['psnr', 'ssim']]
          .mean().reset_index())
