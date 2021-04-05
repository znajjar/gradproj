import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity

import compress
from measure import Measure
from rdh import *
from shared import bits_to_bytes, read_image

parser = argparse.ArgumentParser()
parser.add_argument('source', help='The path of the original image.', type=str)
args = parser.parse_args()

ORIGINAL_IMAGE_NAME = args.source
ORIGINAL_IMAGE_PATH = f'res/{ORIGINAL_IMAGE_NAME}'
DATA_PATH = 'res/data.txt'
RDH_ALGORITHMS = [original_algorithm, scaling_algorithm, unidirectional_algorithm]

image_name, _ = os.path.splitext(ORIGINAL_IMAGE_NAME)

original_image = read_image(ORIGINAL_IMAGE_PATH)
dims = original_image.shape
if len(dims) > 2:
    original_image = original_image[:, :, 0]

np.random.seed(2115)
data = bits_to_bytes(np.random.randint(0, 2, size=2000 * 2000) > 0)


def plot(ys, labels, x_label, y_label, name):
    plt.figure()
    for y, label in zip(ys, labels):
        plt.plot(range(1, len(y) + 1), y, '-o', label=label, markersize=3)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(f'plots/{name}.png')


ratios = []
stds = []
means = []
ssids = []
for rdh in RDH_ALGORITHMS:
    stopwatch = Measure()
    print('======================')
    print(rdh)
    print('======================')
    algorithm_rations = []
    algorithm_stds = []
    algorithm_means = []
    algorithm_ssids = []
    for iterations_count in range(1, rdh.limit + 1):
        print(f'{iterations_count} iterations:')

        processed_image, remaining_data = Measure(rdh.embed, 'embedding', print_time=True)\
            (original_image.copy(), data, iterations_count)
        if rdh.extract:
            try:
                recovered_image, extraction_iterations, extracted_data = \
                    Measure(rdh.extract, 'extraction', print_time=True)(processed_image.copy())
                is_successful = \
                    not np.any(original_image - recovered_image) and extraction_iterations == iterations_count
                hidden_data_size = len(extracted_data) * 8
            except Exception as e:
                print(e)
                is_successful = False
                hidden_data_size = 0
                recovered_image = None
        else:
            recovered_image = None
            hidden_data_size = 8 * (len(data) - len(remaining_data))
            is_successful = hidden_data_size >= 0

        if is_successful:
            print(hidden_data_size, 'bits')
            print(hidden_data_size / 8000, 'kb')
            print(round(hidden_data_size / original_image.size, 3), 'bit/pixel')
            algorithm_rations.append(hidden_data_size / original_image.size)
            algorithm_stds.append(np.std(processed_image, dtype=np.float64))
            algorithm_means.append(np.mean(processed_image, dtype=np.float64))
            algorithm_ssids.append(structural_similarity(original_image, processed_image))
        else:
            print('extraction failed')
            if recovered_image is not None:
                print('PSNR =', cv2.PSNR(original_image, recovered_image))
            break

        print('total time:', stopwatch)
        print('----------------------')

    ratios.append(algorithm_rations)
    stds.append(algorithm_stds)
    means.append(algorithm_means)
    ssids.append(algorithm_ssids)

algorithms_labels = [algo.label for algo in RDH_ALGORITHMS]
algorithms_labels.append(unidirectional_algorithm.label)

plot(ratios, algorithms_labels, 'Iterations', 'Pure hiding ratio (bpp)', f'rate_{image_name}')
plot(stds, algorithms_labels, 'Iterations', 'Standard Deviation', f'std_{image_name}')
plot(means, algorithms_labels, 'Iterations', 'Mean Brightness', f'mean_{image_name}')
plot(ssids, algorithms_labels, 'Iterations', 'Structural Similarity (SSID)', f'ssid_{image_name}')
