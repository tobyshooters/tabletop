import sys
import pygame
from pygame.locals import *
import cv2
import numpy as np

def pygame_to_np(surface):
    arr = np.array(pygame.surfarray.array3d(surface))
    return np.transpose(arr, (1, 0, 2))

def np_to_pygame(arr):
    arr = np.transpose(arr, (1, 0, 2))
    return pygame.surfarray.make_surface(arr)


if len(sys.argv) != 3:
    print("Usage: python3 match.py calibrate.jpg screenshot.jpg")
    exit()

path_cal = sys.argv[1]
path_shot = sys.argv[2]

calibration = pygame.image.load(path_cal)
calibration = pygame_to_np(calibration)

screenshot = pygame.image.load(path_shot)
screenshot = pygame_to_np(screenshot)

detector = cv2.xfeatures2d_SURF.create(hessianThreshold=300)
kp1, des1 = detector.detectAndCompute(screenshot, None)
kp2, des2 = detector.detectAndCompute(calibration, None)

matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
knn_matches = matcher.knnMatch(des1, des2, 2)

good = []
for m, n in knn_matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if False:
    H = max(calibration.shape[0], screenshot.shape[0])
    W = calibration.shape[1] + screenshot.shape[1]
    canvas = np.empty((H, W, 3), dtype=np.uint8)

    cv2.drawMatches(
        screenshot,
        kp1,
        calibration,
        kp2,
        good,
        canvas,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imshow('Good Matches', canvas)
    cv2.waitKey()

origin = [kp1[m.queryIdx].pt for m in good]
origin = np.array(origin, dtype=np.float32).reshape(-1, 1, 2)

target = [kp2[m.trainIdx].pt for m in good]
target = np.array(target, dtype=np.float32).reshape(-1, 1, 2)

H_shot2cal, _ = cv2.findHomography(origin, target)
H_cal2table = np.load('data/cal2table.npy')
H = H_cal2table @ H_shot2cal


# Init pygame screen
pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((0, 0), RESIZABLE)
font = pygame.font.SysFont('Comic Sans MS', 10)
pygame.display.set_caption("Tabletop Match")
win_w, win_h = pygame.display.get_surface().get_size()

running = True
while running:
    screen.fill((255, 255, 255))

    # Draw warped screenshot
    warped = cv2.warpPerspective(screenshot, H, (win_w, win_h))
    projection = np_to_pygame(warped)
    screen.blit(projection, (0, 0))

    # Draw save button
    pygame.draw.rect(screen, (255, 255, 0), (win_w - 50, win_h - 20, 50, 20))
    label = font.render("Save", False, (0, 0, 0))
    screen.blit(label, (win_w - 48, win_h - 18))

    # Handle events
    events = pygame.event.get()
    for event in events:
        if event.type == QUIT:
            running = False

        if event.type == MOUSEBUTTONUP:
            # Save button
            with open('data/shot2cal.npy', 'wb') as f:
                np.save(f, H_shot2cal)

    pygame.display.update()


pygame.quit()
