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


if len(sys.argv) != 2:
    print("Usage: python3 render.py screenshot.jpg")
    exit()

screenshot = pygame.image.load(sys.argv[1])
screenshot = pygame_to_np(screenshot)

H_shot2cal = np.load('data/shot2cal.npy')
H_cal2table = np.load('data/cal2table.npy')
H = H_cal2table @ H_shot2cal

# Init pygame screen
pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((0, 0), RESIZABLE)
font = pygame.font.SysFont('Comic Sans MS', 10)
pygame.display.set_caption("Tabletop Match")
win_w, win_h = pygame.display.get_surface().get_size()

warped = cv2.warpPerspective(screenshot, H, (win_w, win_h))
projection = np_to_pygame(warped)

running = True
while running:
    screen.fill((255, 255, 255))

    # Draw warped screenshot
    warped = cv2.warpPerspective(screenshot, H, (win_w, win_h))
    projection = np_to_pygame(warped)
    screen.blit(projection, (0, 0))

    # Handle events
    events = pygame.event.get()
    for event in events:
        if event.type == QUIT:
            running = False

    pygame.display.update()


pygame.quit()
