import os
import traceback
from os import path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from rdh_algorithm import *
from util.measure import Measure
from util.util import bits_to_bytes, read_image, is_image
from write_data import RunStats, ImageStats, write_data

ORIGINAL_IMAGES_PATH = 'res/dataset-50/'
original_images = [f'{i}.gif' for i in range(1, 50)]  # path relative to ORIGINAL_IMAGES_PATH

if not original_images:
    for f in os.listdir(ORIGINAL_IMAGES_PATH):
        joined_path = path.join(ORIGINAL_IMAGES_PATH, f)
        if is_image(joined_path):
            original_images.append(joined_path)
else:
    original_images = [ORIGINAL_IMAGES_PATH + file for file in original_images]

print(len(original_images))

original_images = [(os.path.split(image)[1], read_image(image)) for image in original_images]

RDH_ALGORITHMS = [
    original_algorithm,
    scaling_algorithm,
    bp_scaling_algorithm,
    uni_algorithm,
    bp_uni_algorithm_improved,
    bp_uni_algorithm,
]

np.random.seed(2115)
data = bits_to_bytes(np.random.randint(0, 2, size=2000 * 2000) > 0)
for rdh_embedder, rdh_extractor, label in RDH_ALGORITHMS:
    stopwatch = Measure()
    print('======================')
    print(label)
    print('======================')
    run_stats = RunStats(label)
    for filename, original_image in original_images:
        embedder = rdh_embedder(original_image.copy(), data)
        extractor = rdh_extractor()
        iterations_count = 0

        image_stats = ImageStats(filename)

        for embedded_image, iterations_count, _ in embedder:
            print(f'{iterations_count} iterations:')

            try:
                recovered_image, extraction_iterations, extracted_data = Measure(extractor.extract, print_time=True)(embedded_image)
                is_successful = \
                    not np.any(original_image - recovered_image) and extraction_iterations == iterations_count
                hidden_data_size = len(extracted_data) * 8
            except Exception:
                traceback.print_exc()
                is_successful = False
                hidden_data_size = 0
                recovered_image = None

            if is_successful:
                print(hidden_data_size, 'bits')
                print(hidden_data_size / 8000, 'kb')
                print(round(hidden_data_size / original_image.size, 3), 'bit/pixel')
                mean = np.abs(np.mean(original_image) - np.mean(embedded_image, dtype=np.float64))
                std = float(np.std(embedded_image, dtype=np.float64))
                ssim = structural_similarity(original_image, embedded_image)
                ratio = hidden_data_size / original_image.size

                image_stats.append_iteration(mean, std, ssim, ratio)
            else:
                print('extraction failed')
                if recovered_image is not None:
                    print('PSNR =', cv2.PSNR(original_image, recovered_image))
                break

            print('total time:', stopwatch)
            print('----------------------')
        run_stats.append_image_stats(image_stats)
    write_data(run_stats)
