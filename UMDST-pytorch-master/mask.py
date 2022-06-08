import numpy as np
import cv2

def show_mask_on_image(img, mask):
    img = np.float32(img)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)

    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    # cam = (1 - mask) * cam + mask * img
    return cam