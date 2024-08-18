#%%
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2
from matplotlib import pyplot as plt

# %%
def compare_images(image_path1, image_path2):
    """
    Compare two images and highlight the differences.

    Parameters:
    - image_path1: str, path to the first image.
    - image_path2: str, path to the second image.
    - Input images must have the same dimensions.

    Returns:
    - score: float, the SSIM score between the two images.
    - diff: ndarray, the difference image.
    """
    # Load the two input images
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    # Convert the images to grayscale
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute the Structural Similarity Index (SSIM) between the two images
    score, diff = ssim(gray_image1, gray_image2, full=True)
    diff = (diff * 255).astype("uint8")

    print(f"SSIM: {score:.4f}")

    # Threshold the difference image to get the regions with significant differences
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Find contours of the regions that differ
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around the regions of difference on the original images
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the images with the differences highlighted
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.title('Image 1')
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 3, 2)
    plt.title('Image 2')
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 3, 3)
    plt.title('Difference')
    plt.imshow(diff, cmap='gray')

    plt.tight_layout()
    plt.show()

    return score, diff
