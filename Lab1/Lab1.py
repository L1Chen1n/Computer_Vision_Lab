import cv2
import numpy as np
import matplotlib.pyplot as plt

t1_path = "COMP9517_25T1_Lab1_Images/Task1.jpg"
t2_path = "COMP9517_25T1_Lab1_Images/Task2.jpg"
t3_path = "COMP9517_25T1_Lab1_Images/Task3.jpg"

t1 = cv2.imread(t1_path,cv2.IMREAD_GRAYSCALE)
t2 = cv2.imread(t2_path,cv2.IMREAD_GRAYSCALE)
t3 = cv2.imread(t3_path,cv2.IMREAD_GRAYSCALE)

t3_gauss = cv2.GaussianBlur(t3, (3, 3), 0)

def gamma_correction(image, gamma):
    normalized = image / 255.0
    corrected = np.power(normalized, gamma)

    return np.uint8(corrected * 255)

h_fre = np.clip(t3 - t3_gauss, 0, 255)
a = 2
res = np.clip(t3 + a * h_fre, 0, 255)
ans = gamma_correction(res, -0.5)

plt.imshow(ans, cmap="gray")
plt.axis("off")
plt.show()

# res = t3.astype(np.float32) - t3_gauss.astype(np.float32)

# a = 5
# t3_unsharp = np.clip(t3.astype(np.float32) + a * res, 0, 255).astype(np.uint8)


# t3_lap = cv2.Laplacian(t3_unsharp, cv2.CV_64F)
# t3_lap = np.clip(t3_lap, 0, 255)

# b = 0.5
# sharpened_t3 = np.clip(t3_unsharp.astype(np.float32) - b * t3_lap, 0, 255).astype(np.uint8)




# temp = cv2.Laplacian(t3, cv2.CV_64F)
# temp = np.clip(temp, 0, 255)

# a = 1
# t3_lap = np.clip(t3.astype(np.float32) - a * temp, 0, 255).astype(np.uint8)

# t3_gauss = cv2.GaussianBlur(t3_lap, (5, 5), 0)
# res = t3_lap.astype(np.float32) - t3_gauss.astype(np.float32)

# b = 4
# t3_unsharp = np.clip(t3_lap.astype(np.float32) + b * res, 0, 255).astype(np.uint8)


# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.imshow(t3, cmap="gray")
# plt.title("Original")

# plt.subplot(1, 2, 2)
# plt.imshow(t3_unsharp, cmap="gray")
# plt.title("Result")
# plt.show()