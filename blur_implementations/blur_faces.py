import cv2
import numpy as np
from blur_implementations.blur_2 import Blur_2
from . cv2_format_tester import format_test, failed_format_test_print

def face_detection(input_filename, output_filename):
    """ Detects faces, apply blur until faceCascade nolonger able to recognice faces.

    Args:
        image(string) : filename of  image

    Return:
        Nothing, writes image to file when done.
    """
    try:
        if format_test(input_filename):
            raise TypeError
    except TypeError:
        failed_format_test_print(input_filename)
        return

    try:
        if format_test(output_filename):
            raise TypeError
    except TypeError:
        failed_format_test_print(output_filename)
        return



    image = cv2.imread(input_filename)
    faces = detect_faces(image)

    print(f"Found {len(faces)} faces in {input_filename}")

    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

    blur_counter = 0

    while len(faces) > 0:
        blur_counter += 1

        for (x,y,w,h) in faces:
            image = image.astype("uint32")
            c = image.shape[2]
            h = h+y
            w = w+x
            image[y:h, x:w, 0:c] = (image[y:h,      x:w,     0:c]
                                  + image[y-1:h-1,  x:w,     0:c]
                                  + image[y+1:h+1,  x:w,     0:c]
                                  + image[y:h,      x-1:w-1, 0:c]
                                  + image[y:h,      x+1:w+1, 0:c]
                                  + image[y-1:h-1,  x-1:w-1, 0:c]
                                  + image[y-1:h-1,  x+1:w+1, 0:c]
                                  + image[y+1:h+1,  x-1:w-1, 0:c]
                                 + image[y+1:h+1,  x+1:w+1, 0:c])/9

            image = image.astype("uint8")
        faces = detect_faces(image)


    print(f"{len(faces)} faces in {input_filename} after blurring {blur_counter}"+
        " times, blurred image saved.")

    cv2.imwrite(output_filename, image)

def detect_faces(image):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(image, scaleFactor=1.025,
                                        minNeighbors=5, minSize=(30, 30))

    return faces

if __name__ == '__main__':
    face_detection("beatles.jpg","blurred_faces.jpg")
