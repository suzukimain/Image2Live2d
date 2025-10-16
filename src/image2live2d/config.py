import cv2
from PIL import Image

def convert_image(path: str):
    """
    Converts an input image to a processed PIL Image highlighting its edges.
    The function performs the following steps:
    1. Loads the image from the specified path in grayscale mode.
    2. Applies Gaussian blur to reduce noise.
    3. Detects edges using the Canny edge detection algorithm.
    4. Converts the resulting NumPy array to a PIL Image.
    Args:
      path (str): The file path to the input image.
    Returns:
      PIL.Image.Image: The processed image with edges highlighted.
    Raises:
      FileNotFoundError: If the image cannot be found at the specified path.
    """
    # Load the image in grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
      raise FileNotFoundError(f"Image not found: {path}")

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Convert NumPy array to PIL.Image
    pil_img = Image.fromarray(edges)

    return pil_img