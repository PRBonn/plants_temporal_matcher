# MIT License

# Copyright (c) 2023 Luca Lobefaro, Meher V. R. Malladi, Olga Vysotska, Tiziano Guadagnino, Cyrill Stachniss

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from PIL import Image, ImageDraw
import cv2
from typing import Tuple, List
import numpy as np


def visualize_visual_matches(
    img1: Image.Image,
    kp1: Tuple[cv2.KeyPoint],
    img2: Image.Image,
    kp2: Tuple[cv2.KeyPoint],
    matches: Tuple[cv2.DMatch],
    inliers_mask: List[int],
    space_between_images: int = 100,
) -> None:
    # Convert the images into numpy
    img1_np = np.asarray(img1)
    img2_np = np.asarray(img2)

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1_np.shape[0]
    cols1 = img1_np.shape[1]
    rows2 = img2_np.shape[0]
    cols2 = img2_np.shape[1]
    out = (
        np.ones(
            (rows1 + rows2 + space_between_images, max([cols1, cols2]), 3),
            dtype="uint8",
        )
        * 255
    )

    # Place the second image to the left
    out[:rows2, :cols2, :] = img2_np

    # Place the next image to the right of it
    out[
        rows2 + space_between_images : rows2 + rows1 + space_between_images, :cols1, :
    ] = img1_np

    # Convert out to PIL image
    out = Image.fromarray(out)
    out_draw = ImageDraw.Draw(out)

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat_idx, mat in enumerate(matches):
        # Filter out outliers
        if not inliers_mask[mat_idx]:
            continue

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1 + rows2 + space_between_images)
        y2 = int(y2)

        # Draw a small circle at both co-ordinates
        circle_color = (0, 0, 255)
        r = 2
        out_draw.ellipse((x2 - r, y2 - r, x2 + r, y2 + r), fill=circle_color)
        out_draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill=circle_color)

        # Draw a line in between the two points
        line_color = (90, 180, 255)
        line_width = 1
        out_draw.line(
            (x2, y2, x1, y1),
            fill=line_color,
            width=line_width,
        )

    out = out.rotate(-90, expand=True)
    out.show()
