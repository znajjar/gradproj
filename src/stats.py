import traceback
from os.path import join as join_path

import cv2

from rdh_algorithm import *
from util.measure import Measure
from util.util import *
from write_data import RunStats, ImageStats, write_data

# IMAGES_PATH = 'res/dataset-50/'
SAVE_IMAGES = True
IMAGES_PATH = 'res/under_over_exposed/'
original_images = []  # path relative to ORIGINAL_IMAGES_PATH

# if list is empty, find all images in ORIGINAL_IMAGES_PATH
if not original_images:
    for f in os.listdir(IMAGES_PATH):
        joined_path = join_path(IMAGES_PATH, f)
        if is_image(joined_path):
            original_images.append(f)

# read all images and pair them with their filenames
original_images = [(image, read_image(join_path(IMAGES_PATH, image))) for image in original_images]

RDH_ALGORITHMS = [
    original_algorithm,
    scaling_algorithm,
    vo_scaling_algorithm,
    vb_scaling_algorithm,
    nb_vo_original_algorithm,
    bp_scaling_algorithm,
    bp_vo_scaling_algorithm,
    bp_vb_scaling_algorithm,
    bp_nb_vo_original_algorithm
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
                recovered_image, extraction_iterations, extracted_data = Measure(extractor.extract, print_time=True)(
                    embedded_image)
                is_successful = \
                    not np.any(original_image - recovered_image) and \
                    extraction_iterations == iterations_count
                correct_recovered_data = extracted_data[:-1] == data[:len(extracted_data) - 1]
                if not correct_recovered_data:
                    print('recovered data is incorrect')
                    is_successful = False
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
                print_error('EXTRACTION FAILED')
                if recovered_image is not None:
                    print('PSNR =', cv2.PSNR(original_image, recovered_image))
                break

            print('total time:', stopwatch)
            print('----------------------')
        if SAVE_IMAGES:
            os.makedirs(f'out/stats/{filename}/', exist_ok=True)
            cv2.imwrite(f'out/stats/{filename}/{label}.png', embedded_image)
        run_stats.append_image_stats(image_stats)
    write_data(run_stats)
