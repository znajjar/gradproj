import cv2
from util import *
from unidirection import *
from bidirectional import *

np.random.seed(2115)
data = bits_to_bytes(np.random.randint(0, 2, size=2000 * 2000 * 20) > 0)

img_file = "20220604_171945.jpg"
bgr_img = cv2.imread(img_file)
hsv_img_org = cv2.cvtColor(bgr_img, cv2.COLOR_RGB2HLS_FULL)

embedders = [
    # (UnidirectionEmbedder, 1000),
    # (BPUnidirectionEmbedder, 1000),
    # (ImprovedBPUnidirectionEmbedder, 1000),
    # (NbVoEmbedder, 64),
    (VariableBitsScalingEmbedder, 20),
    # (BPNbVoEmbedder, 64),
    # (BPVariableBitsScalingEmbedder, 96),
]

for embedder_class, max_iterations in embedders:
    hsv_img = hsv_img_org.copy()
    value = hsv_img[:, :, 1]

    print(f'algorithm: {embedder_class.__name__}')

    embedder = embedder_class(value, data)
    embedded_image, iterations_count, embedded_data_size = embedder.embed(max_iterations)

    print(f'{round(embedded_data_size / hsv_img.size * 3, 4)} bpp')
    print(f'Old STD: {value.std()}')
    print(f'New STD: {embedded_image.std()}')
    print(f'Abs Mean Difference: {abs(value.mean() - embedded_image.mean())}')
    print(f'SSIM: {structural_similarity(embedded_image, value)}')
    print()


    hsv_img[:, :, 1] = embedded_image

    print(np.sum(np.abs(hsv_img_org[..., 0] - hsv_img[..., 0])))
    print(np.sum(np.abs(hsv_img_org[..., 1] - hsv_img[..., 1])))

    enhanced_image = cv2.cvtColor(hsv_img, cv2.COLOR_HLS2RGB_FULL)

    cv2.imwrite(f'{img_file}_{embedder_class.__name__}.png', enhanced_image)
