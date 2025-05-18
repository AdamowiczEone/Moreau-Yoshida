import os
import numpy as np
import cv2
from skimage.io import imread
import pandas as pd
import glob
from MoreaYoshida import tv_denoise_gradient_descent, PSNR, SSIM

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
        image = load_image(image_path)
        
        for strength_luma in luma_values:
            for strength_chroma in chroma_values:
                denoised_image = tv_denoise_gradient_descent(
                    image, strength_luma, strength_chroma
                )
                
                original_uint8 = (image * 255).astype(np.uint8)
                denoised_uint8 = (denoised_image * 255).clip(0, 255).astype(np.uint8)
                
                psnr_value = PSNR(original_uint8, denoised_uint8)
                ssim_value = SSIM(original_uint8, denoised_uint8)
                
                output_filename = (
                    f"{os.path.splitext(image_name)[0]}_luma{strength_luma:.2f}"
                    f"_chroma{strength_chroma:.2f}.png"
                )
                output_path = os.path.join(output_dir, output_filename)
                save_denoised_image(denoised_image, output_path)
                
                results.append({
                    'image': image_name,
                    'strength_luma': strength_luma,
                    'strength_chroma': strength_chroma,
                    'psnr': psnr_value,
                    'ssim': ssim_value,
                    'output_path': output_path
                })
    

    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'denoising_results.csv')
    results_df.to_csv(csv_path, index=False)
    
    return results_df

if __name__ == "__main__":
    dataset_dir = "dataset/STI/Classic"
    output_dir = "output"
    luma_values = [0.1, 0.5, 0.9]   # val for tests
    chroma_values = [0.1, 0.5, 0.9] # val for tests
    
    results = test_denoising(dataset_dir, output_dir, luma_values, chroma_values)
    
    print("\nСводка результатов:")
    print(results.groupby(['strength_luma', 'strength_chroma'])[['psnr', 'ssim']]
          .mean().reset_index())
