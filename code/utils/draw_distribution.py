# Import necessary libraries
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering

from matplotlib.ticker import ScalarFormatter, LogLocator

# Define the function to plot and save the size distribution chart
def plot_size_distribution(crystal_sizes, file_path):
    # Xác định các khoảng giá trị tròn
    max_value = max(crystal_sizes)
    bin_edges = np.linspace(0, max_value + 1, num=16)  # Chia thành 16 điểm để có 15 khoảng
    bin_edges = np.round(bin_edges, -3)  # Làm tròn đến hàng nghìn

    # Tính tần số của mỗi nhóm
    hist, _ = np.histogram(crystal_sizes, bins=bin_edges)

    # Tạo nhãn cho mỗi nhóm
    bin_labels = [f'{int(bin_edges[i])} - {int(bin_edges[i+1])}' for i in range(len(bin_edges) - 1)]

    # Vẽ biểu đồ cột ngang
    plt.figure(figsize=(12, 8))
    bars = plt.barh(bin_labels, hist, height=0.8)

    # Thêm số lượng cho từng cột
    for bar in bars:
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', va='center')

    # Thiết lập thang logarit cho trục x
    plt.xscale('log')
    plt.gca().xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())

    # Thiết lập tiêu đề và nhãn
    plt.title('Phân bố kích thước tinh thể ')
    plt.xlabel('Số lượng')
    plt.ylabel('Diện tích')

    # Save the chart to an image file
    plt.savefig(file_path)
    plt.close()  # Close the plot to free up memory