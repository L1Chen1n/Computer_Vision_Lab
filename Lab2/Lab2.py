import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io, transform

# def draw_custom_keypoints(img, keypoints, color):
#     image_with_keypoints = img.copy()
#     for kp in keypoints:
#         X, Y = int(kp.pt[0]), int(kp.pt[1])
#         radius = int(kp.size * 2)
#         cv2.circle(image_with_keypoints, (X, Y), radius, color, 2)
    
#     return image_with_keypoints

# def rotate_image(img, degree):
#     (h, w) = img.shape[: 2]
#     img_center = (w//2, h//2)
#     rotation_matrix = cv2.getRotationMatrix2D(img_center, degree, 1.0)
#     cos_val = np.abs(rotation_matrix[0, 0])
#     sin_val = np.abs(rotation_matrix[0, 1])

#     new_width = int((h * sin_val) + (w * cos_val))
#     new_height = int((h * cos_val) + (w * sin_val))

#     rotation_matrix[0, 2] += (new_width / 2) - img_center[0]
#     rotation_matrix[1, 2] += (new_height / 2) - img_center[1]

#     rotated_img = cv2.warpAffine(img, rotation_matrix, (new_width, new_height))
#     return rotated_img

# img1 = cv2.imread("Lab2\img3.jpg")
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

# img2 = cv2.imread("Lab2\img4.jpg")
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# height1, width1 = img1.shape[:2]
# height2, width2 = img2.shape[:2]

# ratio1 = 500 / width1
# new_height1 = int(ratio1 * height1)

# ratio2 = 500 / width2
# new_height2 = int(ratio2 * height2)

# img1 = cv2.resize(img1, (500, new_height1))
# img2 = cv2.resize(img2, (500, new_height2))

# # plt.imshow(img2)
# # plt.show()

# sift = cv2.SIFT.create(nfeatures=30)

# img1_keypoints, img1_descriptors = sift.detectAndCompute(img1, None)
# img2_keypoints, img2_descriptors = sift.detectAndCompute(img2, None)

# img1_sift = draw_custom_keypoints(img1, img1_keypoints, (255, 0, 0))
# img2_sift = draw_custom_keypoints(img2, img2_keypoints, (255, 0, 0))

# # plt.subplot(1,2,1)
# # plt.imshow(img1_sift)

# # plt.subplot(1,2,2)
# # plt.imshow(img2_sift)
# # plt.show()

# # image_with_keypoints = cv2.drawKeypoints(img1, keypoints, None, color=(0,0,255))

# # plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
# # plt.title('Output Image')
# # plt.axis('off')
# # plt.show()


# scale_factor = 1.2
# t2_img1 = cv2.resize(img1, (int(500 * scale_factor), int(new_height1 * scale_factor)))
# t2_img2 = cv2.resize(img2, (int(500 * scale_factor), int(new_height2 * scale_factor)))
# rotated_img = transform.rotate(t2_img1, -60, resize=True)
# angle = -60
# t2_img1 = rotate_image(t2_img1, -60)
# t2_img2 = rotate_image(t2_img2, -60)

# plt.subplot(1,2,1)
# plt.imshow(rotated_img)

# plt.subplot(1,2,2)
# plt.imshow(t2_img2)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import transform, util, img_as_ubyte


img1 = cv2.imread("Lab2\img3.jpg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread("Lab2\img4.jpg")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

height1, width1 = img1.shape[:2]
height2, width2 = img2.shape[:2]

ratio = 1000 / width1


img1 = cv2.resize(img1, (1000, int(height1 * ratio)))
img2 = cv2.resize(img2, (1000, int(height2 * ratio)))

sift = cv2.SIFT.create()

gray_img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

img1_kp, img1_dp = sift.detectAndCompute(gray_img1, None)
img2_kp, img2_dp = sift.detectAndCompute(gray_img2, None)

img1_sift = cv2.drawKeypoints(img1, img1_kp, img1.copy(), (255, 0, 0))
img2_sift = cv2.drawKeypoints(img2, img2_kp, img2.copy(), (255, 0, 0))

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

matches = bf.knnMatch(img1_dp, img2_dp, k=2)
good_matches = []
for m, n in matches:
    if m.distance < n.distance * 0.75:
        good_matches.append(m)

img_matches = cv2.drawMatches(img1, img1_kp, img2, img2_kp, good_matches, (255, 0, 0), (255, 0, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

img1_pts = np.float32([img1_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
img2_pts = np.float32([img2_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)

h1, w1, _ = img1.shape
h2, w2, _ = img2.shape

pts_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
pts_transformed = cv2.perspectiveTransform(pts_img1, H)


pts_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
pts_combined = np.concatenate((pts_transformed, pts_img2), axis=0)

[x_min, y_min] = np.int32(pts_combined.min(axis=0).ravel())
[x_max, y_max] = np.int32(pts_combined.max(axis=0).ravel())

translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

stitched_img = cv2.warpPerspective(img1, translation @ H, (x_max - x_min, y_max - y_min))

stitched_img[-y_min:h2 - y_min, -x_min:w2 - x_min] = img2

# y = 0
# for m in stitched_img:
#     if sum(m[-1]) != 0:
#         break
#     y += 1
# print(y)
x_start, y_start = 860, 1074
x_end, y_end = 3477, 1074 + 1333
cropped_image = stitched_img[y_start:y_end, x_start:x_end]

plt.figure(figsize=(12, 6))
plt.imshow(cropped_image)
plt.show()
print(1074 + 1333)
# 860  1074

#((860, 1074), (3477, 1074 + 1333))