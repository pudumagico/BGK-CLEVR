import os
import numpy as np
import cv2
from skimage import util, img_as_ubyte
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

def add_gaussian_noise(image_path, mean=0, std=0.1):
    image = cv2.imread(image_path)
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = np.clip(image + noise * 255, 0, 255).astype(np.uint8)
    return noisy_image

def add_salt_and_pepper_noise(image_path, amount=0.05):
    image = cv2.imread(image_path)
    noisy_image = util.random_noise(image, mode='s&p', amount=amount)
    return img_as_ubyte(noisy_image)

def add_poisson_noise(image_path):
    image = cv2.imread(image_path)
    noisy_image = util.random_noise(image, mode='poisson')
    return img_as_ubyte(noisy_image)

def add_speckle_noise(image_path, mean=0, std=0.1):
    image = cv2.imread(image_path)
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = np.clip(image + image * noise, 0, 255).astype(np.uint8)
    return noisy_image

def add_uniform_noise(image_path, low=-0.1, high=0.1):
    image = cv2.imread(image_path)
    noise = np.random.uniform(low, high, image.shape)
    noisy_image = np.clip(image + noise * 255, 0, 255).astype(np.uint8)
    return noisy_image

def add_periodic_noise(image_path, freq=20):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    rows, cols = image.shape
    x = np.linspace(0, 1, cols)
    y = np.linspace(0, 1, rows)
    X, Y = np.meshgrid(x, y)
    periodic_noise = np.sin(2 * np.pi * freq * X) * 127 + 128
    noisy_image = np.clip(image + periodic_noise, 0, 255).astype(np.uint8)
    return noisy_image

def add_rayleigh_noise(image_path, scale=0.1):
    image = cv2.imread(image_path)
    noise = np.random.rayleigh(scale, image.shape)
    noisy_image = np.clip(image + noise * 255, 0, 255).astype(np.uint8)
    return noisy_image

def add_gamma_noise(image_path, shape=2.0, scale=1.0):
    image = cv2.imread(image_path)
    noise = np.random.gamma(shape, scale, image.shape)
    noisy_image = np.clip(image + noise * 255, 0, 255).astype(np.uint8)
    return noisy_image

def add_correlated_noise(image_path, mean=0, std=0.1, correlation=0.5):
    image = cv2.imread(image_path)
    noise = np.random.normal(mean, std, image.shape)
    correlated_noise = gaussian_filter(noise, sigma=correlation)
    noisy_image = np.clip(image + correlated_noise * 255, 0, 255).astype(np.uint8)
    return noisy_image

def add_color_noise(image_path, mean=0, std=0.1):
    image = cv2.imread(image_path)
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = np.clip(image + noise * 255, 0, 255).astype(np.uint8)
    return noisy_image

def add_adversarial_noise(image, epsilon=0.01):
    # Placeholder for adversarial noise. Typically requires a pre-trained model and specific attack implementation.
    noise = epsilon * np.sign(np.random.randn(*image.shape))
    noisy_image = np.clip(image + noise * 255, 0, 255).astype(np.uint8)
    return noisy_image

def add_structured_noise(image_path, pattern='grid', intensity=0.5):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    rows, cols = image.shape
    if pattern == 'grid':
        structured_noise = np.indices((rows, cols)).sum(axis=0) % 2 * 255 * intensity
    elif pattern == 'checkerboard':
        structured_noise = np.indices((rows, cols)).sum(axis=0) % 2 * 255 * intensity
        structured_noise[1::2, :] = 255 - structured_noise[1::2, :]
    else:
        raise ValueError("Unsupported pattern type.")
    noisy_image = np.clip(image + structured_noise, 0, 255).astype(np.uint8)
    return noisy_image

def add_random_occlusions(image_path, occlusion_size=50, occlusion_count=300):
    image = cv2.imread(image_path)
    for _ in range(occlusion_count):
        x = np.random.randint(0, image.shape[1] - occlusion_size)
        y = np.random.randint(0, image.shape[0] - occlusion_size)
        image[y:y+occlusion_size, x:x+occlusion_size] = np.random.choice([0, 255])
    return image

# # Example usage:
# noisy_image_gaussian = add_gaussian_noise('image.png')
# cv2.imwrite('noisy_image_gaussian.png', noisy_image_gaussian)

# noisy_image_salt_and_pepper = add_salt_and_pepper_noise('image.png')
# cv2.imwrite('noisy_image_salt_and_pepper.png', noisy_image_salt_and_pepper)

# noisy_image_poisson = add_poisson_noise('image.png')
# cv2.imwrite('noisy_image_poisson.png', noisy_image_poisson)

# noisy_image_speckle = add_speckle_noise('image.png')
# cv2.imwrite('noisy_image_speckle.png', noisy_image_speckle)

# noisy_image_uniform = add_uniform_noise('image.png')
# cv2.imwrite('noisy_image_uniform.png', noisy_image_uniform)

# noisy_image_periodic = add_periodic_noise('image.png')
# cv2.imwrite('noisy_image_periodic.png', noisy_image_periodic)

# noisy_image_rayleigh = add_rayleigh_noise('image.png')
# cv2.imwrite('noisy_image_rayleigh.png', noisy_image_rayleigh)

# noisy_image_gamma = add_gamma_noise('image.png')
# cv2.imwrite('noisy_image_gamma.png', noisy_image_gamma)

# # Example usage:
# noisy_image_correlated = add_correlated_noise('image.png')
# cv2.imwrite('noisy_image_correlated.png', noisy_image_correlated)

# noisy_image_color = add_color_noise('image.png')
# cv2.imwrite('noisy_image_color.png', noisy_image_color)

# # Adversarial noise typically requires a pre-trained model, so this is a placeholder.
# image = cv2.imread('image.png')
# noisy_image_adversarial = add_adversarial_noise(image)
# cv2.imwrite('noisy_image_adversarial.png', noisy_image_adversarial)

# noisy_image_structured = add_structured_noise('image.png')
# cv2.imwrite('noisy_image_structured.png', noisy_image_structured)

# noisy_image_occlusions = add_random_occlusions('image.png')
# cv2.imwrite('noisy_image_occlusions.png', noisy_image_occlusions)

def apply_noise_to_folder(folder_path, noise_type):
    noise_functions = {
        'gaussian': add_gaussian_noise,
        'salt_and_pepper': add_salt_and_pepper_noise,
        'poisson': add_poisson_noise,
        'speckle': add_speckle_noise,
        'uniform': add_uniform_noise,
        'periodic': add_periodic_noise,
        'rayleigh': add_rayleigh_noise,
        'gamma': add_gamma_noise,
        'correlated': add_correlated_noise,
        'color': add_color_noise,
        'adversarial': add_adversarial_noise,
        'structured': add_structured_noise,
        'random_occlusions': add_random_occlusions
    }

    parent_folder, folder_name = os.path.split(folder_path)
    output_folder = os.path.join(f"{parent_folder}_{noise_type}", folder_name)
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            # if noise_type == 'periodic' or noise_type == 'structured':
            #     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            noisy_image = noise_functions[noise_type](image_path)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, noisy_image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply noise to images in a folder.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing images.")
    parser.add_argument("noise_type", type=str, help="Type of noise to apply.")
    args = parser.parse_args()

    apply_noise_to_folder(args.folder_path, args.noise_type)