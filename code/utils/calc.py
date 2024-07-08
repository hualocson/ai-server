from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
import cv2
import os
from utils.custom_inferrence_model import VisualizationDemo
from utils.preprocess_image import preprocess_image
from detectron2.utils.visualizer import ColorMode

def detect_and_draw_line_on_image(mask):
    # Chuyển đổi mask từ PyTorch tensor sang numpy array
    mask_np = mask.cpu().numpy()
    # Khởi tạo danh sách để lưu các đường thẳng được phát hiện
    list_of_lines = []

    # Xử lý từng layer của mask
    for i in range(mask_np.shape[0]):
        # Tìm contours từ mask
        contours, _ = cv2.findContours(mask_np[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Bỏ qua nếu không có contours nào
        if not contours:
            continue

        # Khởi tạo các biến để tìm contour lớn nhất
        max_area = 0
        max_polygon = None

        # Duyệt qua từng contour
        for contour in contours:
            # Kiểm tra nếu contour có ít nhất 4 điểm
            if len(contour) >= 4:
                # Tạo đa giác từ contour và tính diện tích
                polygon = ShapelyPolygon(contour.reshape(-1, 2))
                area = polygon.area
                # Cập nhật nếu đa giác có diện tích lớn nhất
                if area > max_area:
                    max_area = area
                    simplified_polygon = polygon.simplify(6, preserve_topology=False)
                    max_polygon = simplified_polygon

        # Xử lý nếu có đa giác lớn nhất
        if max_polygon is not None:
            # Tìm hình chữ nhật nhỏ nhất chứa đa giác
            rect = max_polygon.minimum_rotated_rectangle
            # Lấy điểm của hình chữ nhật
            rect_points = np.array(rect.exterior.xy).T

            # Xử lý nếu có đủ điểm để tạo đường thẳng
            if rect_points.shape[0] >= 2:
                # Tính chiều dài các cạnh và tìm 2 cạnh ngắn nhất
                edge_lengths = np.linalg.norm(np.diff(rect_points, axis=0), axis=1)
                min_edge_length_indices = np.argsort(edge_lengths)[:2]

                # Xử lý nếu tìm thấy 2 cạnh ngắn nhất
                if len(min_edge_length_indices) >= 2:
                    # Lấy 2 cạnh ngắn và tính trung điểm
                    short_edge1 = rect_points[min_edge_length_indices[0]:min_edge_length_indices[0] + 2]
                    short_edge2 = rect_points[min_edge_length_indices[1]:min_edge_length_indices[1] + 2]

                    midpoint_short_edge1 = np.mean(short_edge1, axis=0)
                    midpoint_short_edge2 = np.mean(short_edge2, axis=0)
                    # Lưu thông tin đoạn thẳng
                    list_of_lines.append([midpoint_short_edge1.astype(int), midpoint_short_edge2.astype(int)])
                else:
                    # Xử lý khi không đủ chỉ số để tạo đường thẳng
                    pass
            else:
                # Xử lý khi rect_points không đủ để tạo đường thẳng
                pass

    # Trả về hình ảnh đã được vẽ và danh sách các đường thẳng
    return list_of_lines

def calculate_average_length(lines):
    total_length = 0
    num_lines = len(lines)

    if num_lines == 0:
        return 0

    for line in lines:
        if len(line) >= 2:
            # Tính độ dài của đoạn thẳng bằng cách sử dụng khoảng cách Euclidean
            length = np.linalg.norm(line[1] - line[0])
            total_length += length

    # Tính kích thước trung bình
    average_length = total_length / num_lines

    return average_length

def get_images_from_directory(directory, image_extensions=['.jpg', '.jpeg', '.png']):
    """
    Hàm này trả về một danh sách các đường dẫn đến các ảnh trong một thư mục cụ thể.

    :param directory: Đường dẫn đến thư mục chứa ảnh.
    :param image_extensions: Danh sách các định dạng ảnh (mặc định là jpg, jpeg, png).
    :return: Danh sách các đường dẫn đến ảnh.
    """
    images = []
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            images.append(os.path.join(directory, filename))
    return images


import time
def get_average_length_of_pictures_in_directory(directory, cfg):
    '''
    Input : directory
    Return a dictionary of lengths
    '''
    average_lengths_of_picture = []
    count = 0
    datas = get_images_from_directory(directory)
    for d in datas:
        img_origin = cv2.imread(d)
            # Load the model
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        predictor = VisualizationDemo(cfg, instance_mode=ColorMode.SEGMENTATION, parallel=False)
        img = preprocess_image(img_origin)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Perform inferrence
        predictions_amodal, visualized_output_amodal = predictor.run_on_image(img, segm_type  = 'amodal')

        list_of_lines = detect_and_draw_line_on_image(predictions_amodal['instances'].pred_masks)

        average_length = calculate_average_length(list_of_lines)
        if average_length>0:
            average_lengths_of_picture.append(average_length)
            count = count + 1
    return average_lengths_of_picture, count

def measure_average_lengths_of_images(directory, cfg, is_log_time = False):
    start_time = time.time()
    # Tính độ dài trung bình của hình ảnh
    average_lengths, count = get_average_length_of_pictures_in_directory(directory, cfg)
    end_time = time.time()
    if is_log_time==True:
        delta = end_time - start_time
        average = np.mean(average_lengths)
        print("Kích thước trung bình của các tinh thể:", round(average,3), "micromet")
        print("Tổng thời gian thực hiện tính toán:", round(delta,1), "giây trên", count , "hình ảnh")
    return average_lengths, count