import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity

import compress
from measure import Measure
from rdh import *

ORIGINAL_IMAGE_NAME = 'peppers'
ORIGINAL_IMAGE_PATH = f'res/{ORIGINAL_IMAGE_NAME}.png'
DATA_PATH = 'res/data.txt'
RDH_ALGORITHMS = [original_algorithm, scaling_algorithm]
COMPRESSION_ALGORITHM = compress.Zlib()


def plot(xs, ys, labels, x_label, y_label, name):
    plt.figure()
    for x, y, label in zip(xs, ys, labels):
        plt.plot(x, y, 'o', label=label, markersize=3)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(f'plots/{name}.png')
    plt.show()


def evaluate():
    global iterations_count
    if rdh.extract:
        try:
            recovered_image, iterations_count, extracted_data = \
                Measure(rdh.extract, 'extraction')(processed_image.copy())
            is_successful = not np.any(original_image - recovered_image)
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
        algorithm_iterations.append(iterations_count)
        algorithm_rations.append(hidden_data_size / original_image.size)
        algorithm_stds.append(np.std(processed_image, dtype=np.float64))
        algorithm_means.append(np.mean(processed_image, dtype=np.float64))
        algorithm_ssids.append(structural_similarity(original_image, processed_image))
    else:
        print('extraction failed')
        if recovered_image is not None:
            print('PSNR =', cv2.PSNR(original_image, recovered_image))

    return


original_image = cv2.imread(ORIGINAL_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
data = open(DATA_PATH, 'rb').read()

ITERATIONS = np.arange(1, 65)
iterations = []
ratios = []
stds = []
means = []
ssids = []
for rdh in RDH_ALGORITHMS:
    stopwatch = Measure()
    print('======================')
    print(rdh)
    print('======================')
    algorithm_iterations = []
    algorithm_rations = []
    algorithm_stds = []
    algorithm_means = []
    algorithm_ssids = []
    for iterations_count in ITERATIONS:
        print(f'{iterations_count} iterations:')

        processed_image, remaining_data = Measure(rdh.embed, 'embedding')(original_image.copy(), data, iterations_count,
                                                                          COMPRESSION_ALGORITHM.compress)
        evaluate()

        print('total time:', stopwatch)
        print('----------------------')

    iterations.append(algorithm_iterations)
    ratios.append(algorithm_rations)
    stds.append(algorithm_stds)
    means.append(algorithm_means)
    ssids.append(algorithm_ssids)

algorithm_iterations = []
algorithm_rations = []
algorithm_stds = []
algorithm_means = []
algorithm_ssids = []
rdh = unidirectional_algorithm
processed_image = rdh.embed(original_image.copy(), data)
evaluate()
iterations.append(algorithm_iterations)
ratios.append(algorithm_rations)
stds.append(algorithm_stds)
means.append(algorithm_means)
ssids.append(algorithm_ssids)

algorithms_labels = [algo.label for algo in RDH_ALGORITHMS]
algorithms_labels.append(unidirectional_algorithm.label)

plot(iterations, ratios, algorithms_labels, 'Iterations', 'Pure hiding ratio (bpp)', ORIGINAL_IMAGE_NAME + '_rate')
plot(iterations, stds, algorithms_labels, 'Iterations', 'Standard Deviation', ORIGINAL_IMAGE_NAME + '_std')
plot(iterations, means, algorithms_labels, 'Iterations', 'Mean Brightness', ORIGINAL_IMAGE_NAME + '_mean')
plot(iterations, ssids, algorithms_labels, 'Iterations', 'Structural Similarity (SSID)', ORIGINAL_IMAGE_NAME + '_ssid')
