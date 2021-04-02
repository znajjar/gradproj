import cv2
import matplotlib.pyplot as plt
import numpy as np

import compress
from measure import Measure
from rdh import *

ORIGINAL_IMAGE_PATH = 'res/lena_gray_512.png'
DATA_PATH = 'res/data.txt'
RDH_ALGORITHMS = [scaling_algorithm, original_algorithm]
COMPRESSION_ALGORITHM = compress.Zlib()

original_image = cv2.imread(ORIGINAL_IMAGE_PATH)[:, :, 0]
data = open(DATA_PATH, 'rb').read()

iterations = np.arange(1, 64)
ratios = []
stds = []
means = []
for rdh in RDH_ALGORITHMS:
    stopwatch = Measure()
    print('======================')
    print(rdh)
    print('======================')
    print('----------------------')
    algorithm_rations = []
    algorithm_stds = []
    algorithm_means = []
    for iterations_count in iterations:
        print(f'{iterations_count} iterations:')

        processed_image, remaining_data = Measure(rdh.embed, 'embedding')(original_image, data, iterations_count,
                                                                          COMPRESSION_ALGORITHM.compress)

        if rdh.extract:
            try:
                recovered_image, _, extracted_data = \
                    Measure(rdh.extract, 'extraction')(processed_image, COMPRESSION_ALGORITHM.decompress)
                is_successful = not np.any(original_image - recovered_image)
                hidden_data_size = len(extracted_data) * 8
            except:
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
        else:
            print('extraction failed')
            if recovered_image is not None:
                print('PSNR =', cv2.PSNR(original_image, recovered_image))
            algorithm_rations.append(0)
            algorithm_stds.append(0)
            algorithm_means.append(0)
        print('total time:', stopwatch)
        print()

    ratios.append(algorithm_rations)
    stds.append(algorithm_stds)
    means.append(algorithm_means)

plt.figure(0)
for ratio, algo in zip(ratios, RDH_ALGORITHMS):
    plt.plot(iterations, ratio, label=algo.label)
plt.xlabel('Iterations')
plt.ylabel('Pure hiding ratio (bit per pixel)')
plt.legend()
plt.show()

plt.figure(1)
for mean, algo in zip(stds, RDH_ALGORITHMS):
    plt.plot(iterations, mean, label=algo.label)
plt.xlabel('Iterations')
plt.ylabel(f'Standard Deviation.')
plt.legend()
plt.show()

plt.figure(2)
for mean, algo in zip(means, RDH_ALGORITHMS):
    plt.plot(iterations, mean, label=algo.label)
plt.xlabel('Iterations')
plt.ylabel(f'Mean Brightness.')
plt.legend()
plt.show()
