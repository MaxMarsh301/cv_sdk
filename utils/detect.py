import cv2
import numpy as np

import time
from concurrent.futures import ThreadPoolExecutor

import networkx as nx

def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def enhance_contrast(image):
    """
    Повышение контрастности изображения.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def pre_process_image(image, blur_ksize=5, thresh_blocksize=11, min_size=1500):
    """
    Предобработка изображения для улучшения распознавания стен.
    """
    # Применение гауссового размытия для уменьшения шума
    blurred = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)

    # Использование адаптивного порога
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, thresh_blocksize, 2)
    
    # Морфологические операции для удаления тонких линий
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Удаление мелких объектов (таких как двери и окна)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_bg, connectivity=8)
    sizes = stats[1:, -1]

    processed_img = np.zeros(labels.shape, np.uint8)

    for i in range(0, num_labels - 1):
        if sizes[i] >= min_size:
            processed_img[labels == i + 1] = 255

    return processed_img

def get_wall_contours(image):
    """
    Получение контуров стен из изображения.
    """
    start_time = time.time()
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    end_time = time.time()
    print(f"Time for findContours: {end_time - start_time:.4f} seconds")
    return contours

def get_wall_coordinates(contours, img_height, epsilon_factor=0.001):
    """
    Получение координат стен с инверсией по оси Y.
    """
    start_time = time.time()
    walls_coordinates = []
    for contour in contours:
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        wall = [[point[0][0], img_height - point[0][1]] for point in approx]
        walls_coordinates.append(wall)
    end_time = time.time()
    print(f"Total time for get_wall_coordinates: {end_time - start_time:.4f} seconds")
    return walls_coordinates

def draw_contours(image, contours):
    """
    Отрисовка контуров на изображении.
    """
    start_time = time.time()
    result = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)
    end_time = time.time()
    print(f"Time for drawContours: {end_time - start_time:.4f} seconds")
    return result

def detectOuterContours(detect_img, output_img=None, color=[255, 255, 255], epsilon_factor=0.001):
    """
    Получение внешних контуров плана этажа.
    """
    start_time = time.time()
    _, thresh = cv2.threshold(detect_img, 230, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = epsilon_factor * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    if output_img is not None:
        output_img = cv2.drawContours(output_img, [approx], 0, color, 2)
    img_height = detect_img.shape[0]
    contour_coordinates = [[point[0][0], img_height - point[0][1]] for point in approx]
    end_time = time.time()
    print(f"Total time for detectOuterContours: {end_time - start_time:.4f} seconds")
    return contour_coordinates, output_img

def detectPreciseBoxes(detect_img, output_img=None, color=[100, 100, 0]):
    """
    Детектирование углов в изображении с высокой точностью.
    """
    start_time = time.time()
    res = []
    contours, _ = cv2.findContours(detect_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if output_img is not None:
            output_img = cv2.drawContours(output_img, [approx], 0, color)
        res.append(approx)
    end_time = time.time()
    print(f"Total time for detectPreciseBoxes: {end_time - start_time:.4f} seconds")
    return res, output_img

def remove_noise(img, noise_removal_threshold):
    """
    Удаление шума из изображения.
    """
    start_time = time.time()
    img[img < 128] = 0
    img[img > 128] = 255
    contours, _ = cv2.findContours(~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img)
    for contour in contours:
        if cv2.contourArea(contour) > noise_removal_threshold:
            cv2.fillPoly(mask, [contour], 255)
    end_time = time.time()
    print(f"Total time for remove_noise: {end_time - start_time:.4f} seconds")
    return mask

def find_corners_and_draw_lines(img, corners_threshold, room_closing_max_length):
    """
    Нахождение углов и отрисовка линий между ними.
    """
    start_time = time.time()
    kernel = np.ones((1, 1), np.uint8)
    dst = cv2.cornerHarris(img, 2, 3, 0.04)
    dst = cv2.erode(dst, kernel, iterations=10)
    corners = dst > corners_threshold * dst.max()
    for y, row in enumerate(corners):
        x_same_y = np.argwhere(row)
        for x1, x2 in zip(x_same_y[:-1], x_same_y[1:]):
            if x2[0] - x1[0] < room_closing_max_length:
                cv2.line(img, (x1[0], y), (x2[0], y), 0, 1)
    for x, col in enumerate(corners.T):
        y_same_x = np.argwhere(col)
        for y1, y2 in zip(y_same_x[:-1], y_same_x[1:]):
            if y2[0] - y1[0] < room_closing_max_length:
                cv2.line(img, (x, y1[0]), (x, y2[0]), 0, 1)
    end_time = time.time()
    print(f"Total time for find_corners_and_draw_lines: {end_time - start_time:.4f} seconds")
    return img

def mark_outside_black(img, mask):
    """
    Закрашивание фона черным цветом.
    """
    start_time = time.time()
    contours, _ = cv2.findContours(~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, mask
    biggest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(mask)
    cv2.fillPoly(mask, [biggest_contour], 255)
    img[mask == 0] = 0
    end_time = time.time()
    print(f"Total time for mark_outside_black: {end_time - start_time:.4f} seconds")
    return img, mask

def find_rooms(img, wall_coordinates, gap_in_wall_min_threshold=50):
    """
    Нахождение комнат в изображении на основе координат стен.
    """
    start_time = time.time()

    G = nx.Graph()
    
    # Добавление узлов и ребер в граф
    for wall in wall_coordinates:
        for i in range(len(wall)):
            p1 = tuple(wall[i])
            p2 = tuple(wall[(i + 1) % len(wall)])
            G.add_node(p1)
            G.add_node(p2)
            G.add_edge(p1, p2)
    
    # Поиск компонентов связности (комнат)
    rooms = [list(comp) for comp in nx.connected_components(G) if len(comp) > gap_in_wall_min_threshold]
    
    end_time = time.time()
    print(f"Total time for find_rooms: {end_time - start_time:.4f} seconds")
    return rooms, img

def get_room_coordinates(boxes, img_height):
    """
    Получение координат комнат из контуров.
    """
    start_time = time.time()
    coordinates = [[[point[0][0], img_height - point[0][1]] for point in box] for box in boxes]
    end_time = time.time()
    print(f"Total time for get_room_coordinates: {end_time - start_time:.4f} seconds")
    return coordinates