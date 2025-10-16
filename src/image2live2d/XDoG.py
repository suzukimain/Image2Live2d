import cv2
import numpy as np
from PIL import Image

# Difference of Gaussians
def dog(img, size=(0,0), k=1.6, sigma=0.5, gamma=1):
    img1 = cv2.GaussianBlur(img, size, sigma)
    img2 = cv2.GaussianBlur(img, size, sigma*k)
    return (img1 - gamma * img2)

# XDoG filter
def xdog(img, sigma=0.5, k=1.6, gamma=1, epsilon=-0.1, phi=10):
    aux = dog(img, sigma=sigma, k=k, gamma=gamma) / 255.0
    # Nonlinear transformation
    for i in range(aux.shape[0]):
        for j in range(aux.shape[1]):
            if aux[i, j] < epsilon:
                aux[i, j] = 1.0
            else:
                aux[i, j] = 1.0 + np.tanh(phi * (aux[i, j] - epsilon))
    return (aux * 255).astype(np.uint8)

# Load image → Apply XDoG for lineart → Return PIL.Image
def convert_img(path, sigma=0.5, k=1.6, gamma=0.98, epsilon=-0.1, phi=10):
    """
    path: Input image path
    sigma, k, gamma, epsilon, phi: XDoG parameters
    return: Lineart image (PIL.Image)
    """
    # Load as grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")

    # Apply XDoG
    lineart = xdog(img, sigma=sigma, k=k, gamma=gamma,
                   epsilon=epsilon, phi=phi)

    # Convert NumPy array to PIL.Image
    pil_img = Image.fromarray(lineart)

    return pil_img
