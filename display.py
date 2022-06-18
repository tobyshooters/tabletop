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


pygame.init()
screen = pygame.display.set_mode((0, 0), RESIZABLE)
pygame.display.set_caption("Tabletop Collage")

# The mini-map we are using to select points
display = pygame.image.load('test.jpg')
w, h = display.get_size()
display = pygame.transform.scale(display, (400 * w / h, 400))
dsp_w, dsp_h = display.get_size()

# The high-resolution image we are warping
src = pygame.image.load('test.jpg')
src = pygame_to_np(src)
src_h, src_w, _ = src.shape

# State for calculating homography
projection = None
image = []  # [[0, 0], [1, 0], [0, 1], [1, 1]]
table = []  # [[0, 0], [1, 0], [0, 1], [1, 1]]

running = True
while running:
    screen.fill((255, 255, 255))

    mouse = pygame.mouse.get_pos()
    events = pygame.event.get()
    win_w, win_h = pygame.display.get_surface().get_size()

    for event in events:
        if event.type == QUIT:
            running = False

        if event.type == MOUSEBUTTONUP:

            # Establish point on image
            if len(image) == len(table):
                image.append(mouse)

            # Find pair on the table
            else:
                table.append(mouse)

                if len(table) >= 4:

                    # Calculate homography
                    origin = np.array(image, dtype=np.float32)
                    origin = np.flip(origin, axis=0)
                    origin *= np.array([src_h / dsp_h, src_w / dsp_w])

                    target = np.array(table, dtype=np.float32)
                    target = np.flip(target, axis=0)

                    H, _ = cv2.findHomography(origin, target)

                    # Update projection
                    warped = cv2.warpPerspective(src, H, (win_w, win_h))
                    print("output cv2", warped.shape)
                    projection = np_to_pygame(warped)
                    print("projection", projection.get_size())


    # Draw projection
    if projection:
        screen.blit(projection, (0, 0))

    # Draw source image
    screen.blit(display, (0, 0))

    # Draw mouse
    cursor = pygame.Surface((30, 30))
    cursor.set_alpha(128)
    pygame.draw.rect(cursor, (255, 0, 255), (0, 0, 30, 30))
    pygame.draw.line(cursor, (0, 0, 0), (0, 15), (30, 15))
    pygame.draw.line(cursor, (0, 0, 0), (15, 0), (15, 30))
    screen.blit(cursor, (mouse[0] - 15, mouse[1] - 15))

    # Draw correspondence
    for i, (mx, my) in enumerate(image):
        pygame.draw.rect(screen, (255, 0, 0), (mx - 5, my - 5, 10, 10))

        if i < len(table):
            mx, my = table[i]
            pygame.draw.rect(screen, (0, 255, 0), (mx - 5,  my - 5, 10, 10))

    pygame.display.update()

pygame.quit()
