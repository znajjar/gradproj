import csv
from util import *

datasets = ["kodek_dataset", "research_dataset", "under_over_exposed"]

if __name__ == '__main__':
    file_names = ["5.3.01.tiff", "peppers.tif", "Bike_grayscale.png", "Sign_grayscale.png", "Pond_grayscale.png",
                  "kodim20_org.png", "kodim08_org.png", "Hanok_grayscale.png"]
    header = ["filename", "mean"]

    with open('means.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write the data
        for filename in file_names:
            for dataset in datasets:
                image_path = f'res/{dataset}/{filename}'
                if os.path.exists(image_path):
                    image = read_image(image_path)
                    writer.writerow([filename, image.mean()])
