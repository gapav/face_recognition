from blur_implementations.blur_2 import Blur_2
import cv2
import numpy as np


def blur_image(input_filename, output_filename = None):
    """

    Args:
        input_filename(string) : filename of original image
        output_filename = None(string)(optional) : filename of output image

    Return:
        Integer 3D array of a blurred image of input filename.
    """
    if output_filename is None:
        new_image_to_blur = Blur_2(input_filename)
        return new_image_to_blur.get_3D_array_blurred()

    else:
        new_image_to_blur = Blur_2(input_filename, output_filename)
        return new_image_to_blur.get_3D_array_blurred()
