import cv2
import numpy as np
from utils import detect

import networkx as nx
from shapely.geometry import LineString, Point, Polygon

from PIL import Image

import os

def intersects_any_wall(line, walls_lines):
    line_string = LineString(line)
    for wall_line in walls_lines:
        if line_string.intersects(wall_line):
            return True
    return False

def create_grid_within_contour(outer_contour, walls_lines, grid_size=10):
    min_x, min_y, max_x, max_y = outer_contour.bounds

    x_range = np.arange(min_x, max_x, grid_size)
    y_range = np.arange(min_y, max_y, grid_size)

    nodes = []
    for x in x_range:
        for y in y_range:
            cell_center = (x + grid_size / 2, y + grid_size / 2)
            cell_polygon = Polygon([
                (x, y), 
                (x + grid_size, y), 
                (x + grid_size, y + grid_size), 
                (x, y + grid_size)
            ])
            if outer_contour.contains(Point(cell_center)) and not intersects_any_wall([cell_center, (cell_center[0] + 1, cell_center[1] + 1)], walls_lines):
                nodes.append(cell_center)
    
    G = nx.Graph()
    for node in nodes:
        G.add_node(node)

    return G, nodes

def add_grid_edges(G, nodes, walls_lines, grid_size, outer_contour_polygon):
    edges = []
    for node in nodes:
        for direction in [(grid_size, 0), (0, grid_size)]:
            neighbor = (node[0] + direction[0], node[1] + direction[1])
            if neighbor in nodes:
                # Проверяем, находятся ли обе точки внутри внешнего контура
                if outer_contour_polygon.contains(Point(node)) and outer_contour_polygon.contains(Point(neighbor)):
                    # Проверяем, не пересекает ли ребро стену
                    if not intersects_any_wall([node, neighbor], walls_lines):
                        G.add_edge(node, neighbor)
                        edges.append([list(node), list(neighbor)])
    return edges

def remove_edges_near_walls(G, edges, walls_lines, grid_size):
    edges_to_remove = []
    for edge in edges:
        edge_line = LineString(edge)
        for wall_line in walls_lines:
            if edge_line.distance(wall_line) < grid_size/2.5:
                edges_to_remove.append(edge)
                break
    for edge in edges_to_remove:
        G.remove_edge(tuple(edge[0]), tuple(edge[1]))
        edges.remove(edge)
    return edges

def calculate_polygon_area(coords):
    n = len(coords)
    area = 0.0
    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

def add_third_coordinate(points):
    return [[x, y, 0] for x, y in points]

def add_first_point(points):
    return points + [points[0]]


def image_graph(img_name, blur_ksize, thresh_blocksize, eps_fac):
    # Получаем текущий путь
    current_folder = os.path.dirname(os.path.abspath(__file__))

    #img_path = os.path.join(current_folder, 'static', 'cv', "img", "received", img_name)
    
    img_path = os.path.join(current_folder, 'image' , img_name)

    img_new_name = "_" + str(blur_ksize) + "_" + str(thresh_blocksize) + img_name

    print(blur_ksize, thresh_blocksize, eps_fac)
    # Загружаем изображение
    gray_image = detect.load_image(img_path)
    # повышаем контрастность
    enhanced_image = detect.enhance_contrast(gray_image)
    cv2.imwrite(os.path.join(current_folder, "image", "enhanced", img_new_name), enhanced_image)
    # Предобработка изображения для улучшения распознавания стен.
    processed_image = detect.pre_process_image(enhanced_image, 
                                               blur_ksize=blur_ksize, #25
                                               thresh_blocksize=thresh_blocksize, #15
                                               min_size=1500, 
                                               current_folder = current_folder,
                                               img_new_name=img_new_name)
    cv2.imwrite(os.path.join(current_folder, "image", "processed_image", img_new_name), processed_image)
    # Получение контуров стен
    processed_image = cv2.bitwise_not(processed_image)
    cv2.imwrite(os.path.join(current_folder, "image", "processed_image", "invert_" + img_new_name), processed_image)

    wall_contours = detect.get_wall_contours(processed_image)
    print(wall_contours)
    #cv2.imwrite(os.path.join(current_folder, "image", "walls_with_contours", "contours_" + img_new_name), wall_contours)
    # Получение координат стен
    wall_coordinates = detect.get_wall_coordinates(wall_contours, gray_image.shape[0], epsilon_factor=eps_fac)

    # Высота и ширина изображения
    h_img = gray_image.shape[0]
    w_img = gray_image.shape[1]

    # отрисовка контуров на изображении
    cv2.imwrite(os.path.join(current_folder, "image", "walls_with_contours", img_new_name), detect.draw_contours(gray_image, wall_contours))
    outer_contour_coordinates, image_with_outer_contour = detect.detectOuterContours(gray_image, gray_image.copy(), epsilon_factor=0.001)
    cv2.imwrite(os.path.join(current_folder, "image", "outer_contour", img_new_name), image_with_outer_contour)
    """
    print("2.5")
    rooms, colored_rooms = detect.find_rooms(gray_image.copy())
    print(2.6)
    cv2.imwrite(os.path.join(current_folder, 'static', 'cv', "img", "colored_room", img_new_name), colored_rooms)

    # Пропускаем шаг преобразования изображения в серое
    room_coordinates = detect.get_room_coordinates(detect.detectPreciseBoxes(gray_image)[0], gray_image.shape[0])
    """
    room_coordinates = []
    walls_lines = [LineString(wall) if len(wall) < 4 else Polygon(wall) for wall in wall_coordinates]
    outer_contour_polygon = Polygon(outer_contour_coordinates)

    grid_size = 50
    G, nodes = create_grid_within_contour(outer_contour_polygon, walls_lines, grid_size=grid_size)
    
    edges = add_grid_edges(G, nodes, walls_lines, grid_size=grid_size, outer_contour_polygon=outer_contour_polygon)

    edges = remove_edges_near_walls(G, edges, walls_lines, grid_size)

    return outer_contour_coordinates, room_coordinates, wall_coordinates, edges

