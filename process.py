from PIL import Image
from graph import *



def process_image_function(image_name, params):
    # Инициализация списков для хранения результатов всех обработок
    all_outer_contour_coordinates = []
    all_room_coordinates = []
    all_wall_coordinates = []
    all_edges = []

    # Цикл по всем комбинациям параметров, делал до интерполяции, 
    # когда вычислял при каких параметрах распознавание лучше, сейчас использую вычисленые параметры
    for data in params:
        # Получение координат из функции обработки изображений
        outer_contour_coordinates, room_coordinates, wall_coordinates, edges = image_graph(image_name, data[2])
        
        # Добавление результатов текущей итерации в общие списки
        all_outer_contour_coordinates.append(outer_contour_coordinates)
        all_room_coordinates.append(room_coordinates)
        all_wall_coordinates.append(wall_coordinates)
        all_edges.append(edges)
    
    return [all_outer_contour_coordinates, all_room_coordinates, all_wall_coordinates, all_edges]

def convert_np_float64_to_float(data):
    if isinstance(data, list):
        return [convert_np_float64_to_float(item) for item in data]
    elif isinstance(data, np.float64) or isinstance(data, np.int32):
        return float(data)
    else:
        return data
    

