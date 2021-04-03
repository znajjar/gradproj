import requests
import shutil


def download_image(image_url, filename=''):
    if filename == '':
        filename = image_url.split("/")[-1]

    r = requests.get(image_url, stream=True)

    if r.status_code == 200:
        r.raw.decode_content = True
        with open('img/' + filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
        print('Image sucessfully Downloaded: ', filename)
    else:
        print('Image Couldn\'t be retreived')
