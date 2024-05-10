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
from dataclasses import dataclass
from PIL import Image
from typing import Tuple
import numpy as np
import cv2
import open3d as o3d


@dataclass
class Frame:
    rgb_img: Image.Image
    depth_img: Image.Image
    keypoints: Tuple[cv2.KeyPoint]
    descriptors: np.ndarray
    camera_intrinsics: o3d.camera.PinholeCameraIntrinsic
    camera_extrinsics: np.ndarray
