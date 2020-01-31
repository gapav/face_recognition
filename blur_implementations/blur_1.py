import cv2
import numpy as np
from . cv2_format_tester import format_test, failed_format_test_print
import time

class Blur_1:
    """ Class containing function for blurring images.

    Args:
        input_filename(string) [optional] : filename of original image
        output_filename(string) [optional] : filename of output image

    """


    def __init__(self, input_filename, output_filename):

        self.input_filename = input_filename
        self.output_filename = output_filename


    def pure_python(self):
        """ Pure pythonic function, blurring and writing an image

        Args:
            param1(string) : filename of original image

        Returns:
            Void, creates blurred version of param1, saved as param2 in same folder as original file

        """

        try:
            if format_test(self.input_filename):
                raise TypeError
        except TypeError:
            failed_format_test_print(self.input_filename)
            return

        try:
            if format_test(self.output_filename):
                raise TypeError
        except TypeError:
            failed_format_test_print(self.output_filename)
            return



        src_unpadded = cv2.imread(self.input_filename)
        src_unpadded = src_unpadded.astype("uint32")

        src = np.pad(src_unpadded, 1, mode="edge")
        dst = np.zeros_like(src_unpadded)

        for c in range(dst.shape[2]):
            for h in range(dst.shape[0] - 1):
                for w in range(dst.shape[1] - 1):
                    dst[h, w, c] = (src[h, w, c]
                                    + src[h - 1, w, c]
                                    + src[h + 1, w, c]
                                    + src[h, w - 1, c]
                                    + src[h, w + 1, c]
                                    + src[h - 1, w - 1, c]
                                    + src[h - 1, w + 1, c]
                                    + src[h + 1, w - 1, c]
                                    + src[h + 1, w + 1, c]) / 9

        dst = dst.astype("uint8")
        cv2.imwrite(self.output_filename, dst)
