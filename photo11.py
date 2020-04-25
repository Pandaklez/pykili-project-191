import os
import random

def random_picture(photos_papka):
    files_list = os.listdir(photos_papka)
    choice = random.choice(files_list)
    photo = photos_papka + '\\' + choice
    return os.system(photo)

random_picture(r'C:\НИУ ВШЭ\Программирование\Project\cat')

