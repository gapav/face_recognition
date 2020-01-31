import cv2
import numpy as np
import time
from . cv2_format_tester import format_test, failed_format_test_print


class Blur_2:
    """ Class containing functions for blurring images.


    Args:
        input_filename(string) : filename of original image
        output_filename = None(string)(optional) : filename of output image
    """

    def __init__ (self, input_filename, output_filename = None):
        """ Class containing functions for blurring images.


        Args:
            input_filename(string) : filename of original image
            output_filename = None(string)(optional) : filename of output image
        """

        self.input_filename = input_filename
        self.blurred_3D_array = self.numpy_implement()

        if output_filename is not None:
            self.output_filename = output_filename
            self.write_to_image(self.output_filename, self.blurred_3D_array)



    def get_3D_array_blurred(self):
        """ function to be called from blur_image.py

        Args:

        Returns:
            Numpy integer 3D array of a blurred image of input filename
        """
        return self.blurred_3D_array



    @staticmethod
    def vectorize_blur(image):
        """ Vectorized image blur function
        Args:
            image : 3D array of original image
        Returns:
            Numpy integer 3D array of a blurred image of input filename
        """

        return (image[1:-1,1:-1,1:-1] + image[:-2,1:-1,1:-1]
                + image[2:,1:-1,1:-1] + image[1:-1,:-2,1:-1]
                + image[1:-1,2:,1:-1] + image[:-2,:-2,1:-1]
                + image[:-2,2:,1:-1]  + image[2:,:-2,1:-1]
                + image[2:,2:,1:-1])/9


    def  write_to_image(self, output_filename, blurred_image_array):
        """ Function to write to file.
        Args:
            output_filename(string) : filename of output file
            blurred_image : 3D array of blurred image
        Returns:

        """
        try:
            if format_test(self.output_filename):
                raise TypeError
        except TypeError:
            failed_format_test_print(self.output_filename)
            return

        blurred_image_array = blurred_image_array.astype("uint8")
        cv2.imwrite(self.output_filename, blurred_image_array)


    def numpy_implement(self):
        """   Vectorized version of blur_1.py

        Args:

        Returns:
            Numpy integer 3D array of a blurred image of input filename,

        Raises:
            TypeError: If format of input_filname is not supported by cv2 module.
        """
        try:
            if format_test(self.input_filename):
                raise TypeError
        except TypeError:
            failed_format_test_print(self.input_filename)
            return

        src_unpadded = cv2.imread(self.input_filename)
        src_unpadded = src_unpadded.astype("uint32")
        src = np.pad(src_unpadded, (1,1), mode="edge")

        blurred_3D_array = self.vectorize_blur(src)


        return blurred_3D_array
