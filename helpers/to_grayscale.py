import argparse
import os
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('source', help='The path of the original color image.', type=str)
args = parser.parse_args()

path = args.source
f_name, f_ext = os.path.splitext(path)
img = Image.open(path).convert('L')
img.save(f_name + '_gray.png')
