import argparse
from blur_implementations.blur_1 import Blur_1
from blur_implementations.blur_2 import Blur_2
from blur_implementations.blur_3 import Blur_3
from blur_implementations.blur_faces import face_detection
import cv2


parser = argparse.ArgumentParser(description='Process an image file and \
                                            returns a blurred version')

parser.add_argument('input_filename', action='store', \
                    help='filename of an image to be processed')

parser.add_argument('output_filename', action='store', help='name of outfile')

parser.add_argument('implementation', action='store', type = int,
                    help='an integer for wanted implementation.\
                     \n 1: pure python, \
                     \n 2: python with NumyPy, \
                     \n 3: python with Numba')




args = parser.parse_args()

if args.implementation < 1 or args.implementation > 4:
    parser.error("\n   Implementation integer must be 1,2 or 3\n"
                    + "   For help, use -h as argument")

if args.implementation == 1:
    blur_implementation1 = Blur_1(args.input_filename, args.output_filename)
    blur_implementation1.pure_python()

elif args.implementation == 2:
    blur_implementation2 = Blur_2(args.input_filename, args.output_filename)

elif args.implementation == 3:
    blur_implementation3 = Blur_3(args.input_filename, args.output_filename)
    blur_implementation3.pure_python()

else:
    face_detection(args.input_filename, args.output_filename)
