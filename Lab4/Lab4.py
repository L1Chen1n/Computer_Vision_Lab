import cv2
import numpy as np
import matplotlib.pyplot as plt

def morphology(img, type):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    mask = cv2.morphologyEx(img, type, kernel)
    return mask

def fillArea(img_mask, thresh):
    inverted_mask = cv2.bitwise_not(img_mask)

    contours, _ = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < thresh:
            cv2.drawContours(inverted_mask, [cnt], -1, (255), thickness=cv2.FILLED)
    return cv2.bitwise_not(inverted_mask)

def countNum(img_mask, size):
    contours, _ = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    count = 0

    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > size:
            count += 1
    return count

def plot(img):
    plt.imshow(img, cmap="gray")
    plt.show()

def detectBorder(img_dir, thresh_lb, thresh_ub):
    img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)

    width, height = img.shape[: 2]
    thresh_img = cv2.inRange(img, thresh_lb, thresh_ub)

    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    temp = np.zeros_like(thresh_img)

    for contour in contours:
        touches_border = False
        for point in contour:
            x, y = point[0]
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                touches_border = True
                break
        
        # If the object does NOT touch the border, draw it on the mask
        if not touches_border:
            cv2.drawContours(temp, [contour], -1, 255, thickness=cv2.FILLED)
    return temp

def segment(img_dir, thresh_lb, thresh_ub, morpho_type, filled_thresh, fill):
    new_mask = detectBorder(img_dir, thresh_lb, thresh_ub)

    bin_mask = morphology(new_mask, morpho_type)

    if fill:
        h, w = bin_mask.shape[:2]

        mask = np.zeros((h+2, w+2), np.uint8)

        filled_img = bin_mask.copy()

        cv2.floodFill(filled_img, mask=mask, seedPoint=(0,0), newVal=255)

        img_floodfill_inv = cv2.bitwise_not(filled_img)

        img_flood = bin_mask | img_floodfill_inv
    else:
        img_flood = bin_mask
    plot(img_flood)

    filled_mask = fillArea(img_flood, filled_thresh)
    plot(filled_mask)



segment('COMP9517_25T1_Lab4_Images\Leaves.jpg', 100, 140, cv2.MORPH_OPEN, 1600, True)