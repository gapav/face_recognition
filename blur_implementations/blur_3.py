from numba import jit
import cv2
import numpy as np
import time
from . cv2_format_tester import format_test, failed_format_test_print

class Blur_3:
    """ Class containing (almost)pure python function, decorated with
    @jit(Numba)

    Args:
        param1(string) : filename of original image
        param2(string) : filename of output image
    """

    def __init__(self, input_filename, output_file_name):
        self.input_filename = input_filename
        self.output_filename = output_file_name


    def pure_python(self):
        """
        (Almost)Pure pythonic function, decorated with @jit(Numba). Uses Numpy
        for padding, and  storing.

        Args:
            self

        Returns:
            Void, creates blurred version of param1,
            saved as blurred_image_2.jpg in same folder as original file

        Raises:
            TypeError: If format of input_filname is not supported by cv2 module.
        """
        try:
            if format_test(self.input_filename) and not None:
                raise TypeError
        except TypeError:
            failed_format_test_print(self.input_filename)
            return

        try:
            if format_test(self.output_filename) and not None:
                raise TypeError
        except TypeError:
            failed_format_test_print(self.output_filename)
            return

        src_unpadded = cv2.imread(self.input_filename)
        src_unpadded = src_unpadded.astype("uint32")
        src = np.pad(src_unpadded, 1, mode="edge")
        dst = np.zeros_like(src_unpadded)
        t0 = time.time()

        self.generate_blurred_array(dst,src)

        t1 = time.time()

        print("Runtime for blur_3: {}".format(t1-t0))
        dst = dst.astype("uint8")
        cv2.imwrite(self.output_filename, dst)

    @jit
    def generate_blurred_array(self,dst,src):
        height, width, channel = dst.shape

        for c in range(channel):
            for h in range(height):
                for w in range(width):
                    dst[h, w, c] = (src[h, w, c]
                                    + src[h - 1, w, c]
                                    + src[h + 1, w, c]
                                    + src[h, w - 1, c]
                                    + src[h, w + 1, c]
                                    + src[h - 1, w - 1, c]
                                    + src[h - 1, w + 1, c]
                                    + src[h + 1, w - 1, c]
                                    + src[h + 1, w + 1, c]) / 9
