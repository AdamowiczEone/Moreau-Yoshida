import cv2
import numpy as np
imC = cv2.imread("img/robot.png", cv2.IMREAD_COLOR)
imG = cv2.imread("img/robot.png", cv2.IMREAD_GRAYSCALE)
print(imG.shape)

# # Создаём красный квадрат (в BGR)
# red_image = np.zeros((160, 160, 3), dtype=np.uint8)
# red_image[:, :] = [0, 0, 255]  # Заполняем красным (BGR)
#
# cv2.imshow("Red Image", red_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()