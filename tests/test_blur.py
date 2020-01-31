from blur_implementations.blur_2 import Blur_2
import numpy as np



def test_max_value_decrease():
    """Generates a 3-dimensional numpy array with pixel values
        randomly chosen between 0 and 255, and test if max value
        has decreased after blurring.

        Assert:
            compares max_value_after_blurring to max_value_before_blurring
    """
    np.random.seed(0)
    random_array = np.random.randint(0, 255, size=(460, 640, 3))

    max_value_before_blurring = np.amax(random_array)
    random_array_padded = np.pad(random_array,
    (1,1), mode="edge")

    random_array_blurred = Blur_2.vectorize_blur(random_array_padded)
    max_value_after_blurring = np.amax(random_array_blurred)

    assert max_value_before_blurring > max_value_after_blurring




def test_neighbor_pixel_average():
    """Takes a pixel and assert that the pixel in the blurred image
    is the average of its neighbors in the clear image

        Assert: is

    """
    np.random.seed(0)

    random_array_unpadded = np.random.randint(0, 255, size=(460, 640, 3))
    random_array_padded = np.pad(random_array_unpadded, (1,1), mode="edge")
    vectorized_Blur_2_array = Blur_2.vectorize_blur(random_array_padded)

    h = 2
    w = 2
    c = 2

    blurred_pixel =  (random_array_unpadded[h, w, c] + random_array_unpadded[h - 1, w, c] + random_array_unpadded[h + 1, w, c]
                           + random_array_unpadded[h, w - 1, c] + random_array_unpadded[h, w + 1, c]
                           + random_array_unpadded[h - 1, w - 1, c] + random_array_unpadded[h - 1, w + 1, c]
                           + random_array_unpadded[h + 1, w - 1, c] + random_array_unpadded[h + 1, w + 1, c]) / 9

    assert vectorized_Blur_2_array[h,w,c] == blurred_pixel
