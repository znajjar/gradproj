import os

names = os.listdir()

for full_name in names:
    name, extension = os.path.splitext(full_name)
    split_name = name.split('_')
    new_name = f'{split_name[1]}_{split_name[0]}{extension}'
    os.rename(full_name, new_name)

