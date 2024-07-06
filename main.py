import os
from PIL import Image
import numpy as np
from process import process_image_function


def round_to_odd(value):
    return int(value) if int(value) % 2 != 0 else int(value) + 1

def calculate_parameters(width, height):
    # Данные, которые вычислил опытным путем интерполирую на изображение
    sizes = np.array([
        [3949, 2868],
        [8000, 11679]
    ])
    parameters = np.array([
        [5, 40, 0.001],
        [5, 22, 0.0001]
    ])
    
    # Интерполяция параметров
    interp_params = []
    for i in range(3):
        interp = np.interp(
            [width, height],
            [sizes[0, i % 2], sizes[1, i % 2]],
            [parameters[0, i], parameters[1, i]]
        )
        if i < 2:  # Первые два значения параметров должны быть нечётными
            interp_params.append(round_to_odd(interp.mean()))
        else:
            interp_params.append(interp.mean())
    
    return interp_params


# эмулирую отправку изображения на сервер из клиента в простом виде
image_name = '_20240620_164708_fix.png'
image_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'image' , image_name)
with open(image_file, 'rb') as file:
    image = Image.open(image_file)
    params = calculate_parameters(image.width, image.height)
    #использовал список параметров для создания нескольких вариантов распознавания
    params = [params, ]
    #params
    #0 - blur
    #1 - thresh
    #2 - epsilon factor
    #Вызываю функцию, которая на сервере 
    response = process_image_function(image_name, params)

    contour_list = response[0]
    wall_coordinates_list = response[1]
    room_coordinates_list = response[2]
    edges_list = response[3]
    """
    for outer_contour_coordinates in contour_list:
        print(outer_contour_coordinates)

    for wall_coordinates in wall_coordinates_list:
        for wall in wall_coordinates:
            print(wall)

    for room_coordinates in room_coordinates_list:
        for room in room_coordinates:
            print(room)

    for edge in edges_list:
        print(edge)

    """


