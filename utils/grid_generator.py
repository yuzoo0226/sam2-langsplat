import cv2
import numpy as np


def generate_uniform_grid(height, width, num_points, padding=20):
    effective_width = width - 2 * padding
    effective_height = height - 2 * padding

    num_x = int(np.sqrt(num_points))
    num_y = num_x

    x = np.linspace(padding, padding + effective_width - 1, num_x)
    y = np.linspace(padding, padding + effective_height - 1, num_y)
    xx, yy = np.meshgrid(x, y)

    points = np.vstack([xx.ravel(), yy.ravel()]).T
    return points


def draw_points_on_image(image, points, color=(0, 0, 255), radius=3):
    output_image = image.copy()
    for (x, y) in points.astype(int):
        cv2.circle(output_image, (x, y), radius, color, -1)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGBA)
    return output_image

if __name__ == "__main__":
    height, width = 480, 640
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    num_points = 32
    points = generate_uniform_grid(height, width, num_points)

    print(points)

    output_image = draw_points_on_image(image, points)

    cv2.imshow("Points on Image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()