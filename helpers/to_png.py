import argparse
import os
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('source', help='The path of the original image.', type=str)
args = parser.parse_args()

path = args.source
img = np.uint8(Image.open(path))
f_name, f_ext = os.path.splitext(path)
Image.fromarray(img).save(f_name + '.png')
