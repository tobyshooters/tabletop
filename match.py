import pygame
from pygame.locals import *
import cv2
import numpy as np

def pygame_to_np(surface):
    arr = np.array(pygame.surfarray.array3d(surface))
    return np.transpose(arr, (1, 0, 2))

src = pygame.image.load('calibration.jpg')
src = pygame_to_np(src)

screenshot = pygame.image.load('screenshot.jpg')
screenshot_np = pygame_to_np(screenshot)

detector = cv2.xfeatures2d_SURF.create(hessianThreshold=300)
kp1, des1 = detector.detectAndCompute(src, None)
kp2, des2 = detector.detectAndCompute(screenshot_np, None)

matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
knn_matches = matcher.knnMatch(des1, des2, 2)

good = []
for m, n in knn_matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

H = max(src.shape[0], screenshot_np.shape[0])
W = src.shape[1] + screenshot_np.shape[1]
canvas = np.empty((H, W, 3), dtype=np.uint8)

cv2.drawMatches(
    src, kp1,
    screenshot_np, kp2,
    good,
    canvas,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
#-- Show detected matches
cv2.imshow('Good Matches', canvas)
cv2.waitKey()

# H, _ = cv2.findHomography(origin, target)

# warped = cv2.warpPerspective(src, H, (win_w, win_h))
# projection = np_to_pygame(warped)
