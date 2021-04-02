import argparse

import cv2

parser = argparse.ArgumentParser()
parser.add_argument('source', help='The path of the original image.', type=str)
args = parser.parse_args()

path = args.source
img = cv2.imread(path)[:, :, 0]
cv2.imwrite(path.split('.')[0] + '.png', img)
