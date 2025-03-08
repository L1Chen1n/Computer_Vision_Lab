import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io, transform

def draw_custom_keypoints(img, keypoints, color):
    image_with_keypoints = img.copy()
    for kp in keypoints:
        X, Y = int(kp.pt[0]), int(kp.pt[1])
        radius = int(kp.size * 2)
        cv2.circle(image_with_keypoints, (X, Y), radius, color, 2)
    
    return image_with_keypoints

def rotate_image(img, degree):
    (h, w) = img.shape[: 2]
    img_center = (w//2, h//2)
    rotation_matrix = cv2.getRotationMatrix2D(img_center, degree, 1.0)
    cos_val = np.abs(rotation_matrix[0, 0])
    sin_val = np.abs(rotation_matrix[0, 1])

    new_width = int((h * sin_val) + (w * cos_val))
    new_height = int((h * cos_val) + (w * sin_val))

    rotation_matrix[0, 2] += (new_width / 2) - img_center[0]
    rotation_matrix[1, 2] += (new_height / 2) - img_center[1]

    rotated_img = cv2.warpAffine(img, rotation_matrix, (new_width, new_height))
    return rotated_img

img1 = cv2.imread("Lab2\img3.jpg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread("Lab2\img4.jpg")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

height1, width1 = img1.shape[:2]
height2, width2 = img2.shape[:2]

ratio1 = 500 / width1
new_height1 = int(ratio1 * height1)

ratio2 = 500 / width2
new_height2 = int(ratio2 * height2)

img1 = cv2.resize(img1, (500, new_height1))
img2 = cv2.resize(img2, (500, new_height2))

# plt.imshow(img2)
# plt.show()

sift = cv2.SIFT.create(nfeatures=30)

img1_keypoints, img1_descriptors = sift.detectAndCompute(img1, None)
img2_keypoints, img2_descriptors = sift.detectAndCompute(img2, None)

img1_sift = draw_custom_keypoints(img1, img1_keypoints, (255, 0, 0))
img2_sift = draw_custom_keypoints(img2, img2_keypoints, (255, 0, 0))

# plt.subplot(1,2,1)
# plt.imshow(img1_sift)

# plt.subplot(1,2,2)
# plt.imshow(img2_sift)
# plt.show()

# image_with_keypoints = cv2.drawKeypoints(img1, keypoints, None, color=(0,0,255))

# plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
# plt.title('Output Image')
# plt.axis('off')
# plt.show()


scale_factor = 1.2
t2_img1 = cv2.resize(img1, (int(500 * scale_factor), int(new_height1 * scale_factor)))
t2_img2 = cv2.resize(img2, (int(500 * scale_factor), int(new_height2 * scale_factor)))
rotated_img = transform.rotate(t2_img1, -60, resize=True)
angle = -60
t2_img1 = rotate_image(t2_img1, -60)
t2_img2 = rotate_image(t2_img2, -60)

plt.subplot(1,2,1)
plt.imshow(rotated_img)

plt.subplot(1,2,2)
plt.imshow(t2_img2)
plt.show()
