import cv2
import matplotlib.pyplot as plt
import numpy as np

import compress
import original_embed
import scaling_embed
import original_extract
import scaling_extract
from measure import Measure
from rdh import RDH

ORIGINAL_IMAGE_PATH = 'res/lena_gray_256.png'
DATA_PATH = 'res/data.txt'
# RDH_ALGORITHMS = [RDH('original', embed.embed, extract.extract)]
RDH_ALGORITHMS = [RDH('scaling', scaling_embed.embed, scaling_extract.extract),
                  RDH('original', original_embed.embed, original_extract.extract)]
COMPRESSION_ALGORITHM = compress.Zlib()

original_image = cv2.imread(ORIGINAL_IMAGE_PATH)[:, :, 0]
data = open(DATA_PATH, 'rb').read()

for rdh in RDH_ALGORITHMS:
    print('======================')
    print(rdh)
    print('======================')
    print('----------------------')
    ratios = []
    stopwatch = Measure()
    for iterations_count in range(1, 65):
        print(f'{iterations_count} iterations:')

        processed_image, remaining_data = Measure(rdh.embed, 'embedding')(original_image, data, iterations_count,
                                                                          COMPRESSION_ALGORITHM.compress)

        if rdh.extract:
            try:
                recovered_image, iterations, extracted_data = Measure(rdh.extract, 'extraction') \
                    (processed_image, COMPRESSION_ALGORITHM.decompress)
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
            ratios.append(hidden_data_size / original_image.size)
        else:
            print('extraction failed')
            if recovered_image is not None:
                print('PSNR =', cv2.PSNR(original_image, recovered_image))
            ratios.append(0)
        print('total time:', stopwatch)
        print()
    plt.plot(ratios, label=rdh.label)

plt.xlabel('Iterations')
plt.ylabel('Pure hiding ratio (bit per pixel')
plt.legend()
plt.show()
