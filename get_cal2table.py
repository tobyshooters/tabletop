import sys
import json
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


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python3 get_cal2table.py [calibration.png]")
        exit()

    path = sys.argv[1]

    # Init pygame screen
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((0, 0), RESIZABLE)
    font = pygame.font.SysFont('Comic Sans MS', 10)
    pygame.display.set_caption("Tabletop Collage")

    # The mini-map we are using to select points
    display = pygame.image.load(path)
    w, h = display.get_size()
    display = pygame.transform.scale(display, (900 * w / h, 900))
    dsp_w, dsp_h = display.get_size()

    # The high-resolution image we are warping
    src = pygame.image.load(path)
    src = pygame_to_np(src)
    src_h, src_w, _ = src.shape

    # State for calculating homography
    projection = None
    image = []  # [[0, 0], [1, 0], [0, 1], [1, 1]]
    table = []  # [[0, 0], [1, 0], [0, 1], [1, 1]]

    # Modes:
    # 0. TABULA_RASA: show nothing
    # 1. CALIBRATION_SOURCE: request point on source image
    # 2. CALIBRATION_TABLE: request point on table
    # 3. CALIBRATION_INSPECT: just show calibrated image

    mode = "TABULA_RASA"

    running = True
    while running:
        screen.fill((255, 255, 255))

        mouse = pygame.mouse.get_pos()
        events = pygame.event.get()
        win_w, win_h = pygame.display.get_surface().get_size()

        # Drawing thre screen
        if mode == "CALIBRATION_SOURCE":
            # Draw the calibration image
            screen.blit(display, (0, 0))
            
            # Draw existing points
            for i, (mx, my) in enumerate(image):
                pygame.draw.rect(screen, (255, 0, 0), (mx - 5, my - 5, 10, 10))

        elif mode == "CALIBRATION_TABLE":
            # Draw existing points
            for i, (mx, my) in enumerate(table):
                pygame.draw.rect(screen, (0, 255, 0), (mx - 5, my - 5, 10, 10))

        elif mode == "CALIBRATION_INSPECT":
            # Draw the calibration when re-projected
            assert projection
            screen.blit(projection, (0, 0))

        # Draw toggle button
        pygame.draw.rect(screen, (255, 255, 0), (win_w - 50, win_h - 20, 50, 20))
        label = font.render("Calibrate", False, (0, 0, 0))
        screen.blit(label, (win_w - 48, win_h - 18))

        if projection:
            pygame.draw.rect(screen, (255, 0, 255), (win_w - 50, win_h - 40, 50, 20))
            label = font.render("Inspect", False, (0, 0, 0))
            screen.blit(label, (win_w - 48, win_h - 38))

            pygame.draw.rect(screen, (0, 255, 255), (win_w - 50, win_h - 60, 50, 20))
            label = font.render("Save", False, (0, 0, 0))
            screen.blit(label, (win_w - 48, win_h - 58))

        # Draw mouse
        cursor = pygame.Surface((30, 30))
        cursor.set_alpha(128)
        pygame.draw.rect(cursor, (255, 0, 255), (0, 0, 30, 30))
        pygame.draw.line(cursor, (0, 0, 0), (0, 15), (30, 15))
        pygame.draw.line(cursor, (0, 0, 0), (15, 0), (15, 30))
        screen.blit(cursor, (mouse[0] - 15, mouse[1] - 15))

        for event in events:
            if event.type == QUIT:
                running = False

            if event.type == MOUSEBUTTONUP:

                # Handle button clicks to transition states
                if mouse[0] > win_w - 50 and mouse[1] > win_h - 20:
                    if mode != "CALIBRATION_TABLE":
                        mode = "CALIBRATION_SOURCE"
                
                elif mouse[0] > win_w - 50 and mouse[1] > win_h - 40:
                    if mode == "TABULA_RASA":
                        mode = "CALIBRATION_INSPECT"
                    elif mode != "CALIBRATION_TABLE":
                        mode = "TABULA_RASA"

                elif mouse[0] > win_w - 50 and mouse[1] > win_h - 60:
                    with open('cal2table.npy', 'wb') as f:
                        np.save(f, H)

                # Collect correspondences for homography
                elif mode == "CALIBRATION_SOURCE":
                    if mouse[0] < dsp_w and mouse[1] < dsp_h:
                        image.append(mouse)
                        mode = "CALIBRATION_TABLE"

                elif mode == "CALIBRATION_TABLE":
                    table.append(mouse)

                    if len(table) < 4:
                        # Need more points!
                        mode = "CALIBRATION_SOURCE"

                    else:
                        # Calculate homography
                        origin = np.array(image, dtype=np.float32)
                        origin = np.flip(origin, axis=0)
                        origin *= np.array([src_h / dsp_h, src_w / dsp_w])

                        target = np.array(table, dtype=np.float32)
                        target = np.flip(target, axis=0)

                        H, _ = cv2.findHomography(origin, target)

                        # Update projection
                        warped = cv2.warpPerspective(src, H, (win_w, win_h))
                        projection = np_to_pygame(warped)

                        mode = "CALIBRATION_INSPECT"


        pygame.display.update()

    pygame.quit()
