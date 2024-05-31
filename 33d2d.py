import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(filepath):
    """Load an image from file."""
    image = cv2.imread(filepath)
    return image


def preprocess_image(image):
    """Convert image to grayscale and apply edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges


def extract_features(edges):
    """Extract contours from the edges."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_floor_plan(contours, image_shape):
    """Draw the floor plan based on contours."""
    floor_plan = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(floor_plan, contours, -1, (255, 255, 255), 2)
    return floor_plan


def main():
    filepath = 'C:\\Users\\AKSHAY\\PycharmProjects\\pythonProject3\\Screenshot 2024-05-31 223145.png'
    image = load_image(filepath)
    edges = preprocess_image(image)
    contours = extract_features(edges)
    floor_plan = draw_floor_plan(contours, image.shape[:2])

    plt.figure(figsize=(10, 10))
    plt.imshow(floor_plan, cmap='gray')
    plt.title('Generated Floor Plan')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
