"""整合上面的代码"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def getX(image_path):
    # 读取图像
    x = [0]
    image = cv2.imread(image_path)

    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算每列像素值的和
    column_sums = np.sum(gray_image, axis=0).astype(np.int64)

    # 所有列的平均值，虚线显示
    mean_value = np.mean(column_sums)
    # print(mean_value)

    # 寻找极大值和极小值点，自定义寻找的范围：±5
    peaks, _ = find_peaks(column_sums, distance=20)
    valleys, _ = find_peaks(-column_sums, distance=20)

    # 突出显示极小值点，并进行比较
    for valley in valleys[:-1]:
        # # 比较左边点和右边点的差值
        left_diff = column_sums[max(valley - 4, 0)] - column_sums[valley]
        right_diff = column_sums[min(valley + 4, len(column_sums) - 1)] - column_sums[valley]
        # print(left_diff, right_diff)

        if left_diff < 0 or right_diff < 0:
            # 舍弃极小值点
            continue

        # 计算变化值的平均值
        variations = column_sums[valley - 3 : valley + 3]
        variations = sorted(variations)[1:-1]  # 去除最高和最低值
        average_variation = np.mean(variations)

        if average_variation > 200000:
            # print(average_variation)
            x.append(valley)

    x.append(image.shape[1])

    return x


def getY(image_path):
    # 读取图像
    y = [0]
    image = cv2.imread(image_path)

    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算每列像素值的和
    column_sums = np.sum(gray_image, axis=1).astype(np.int64)

    # 所有列的平均值，虚线显示
    mean_value = np.mean(column_sums)
    # print(mean_value)

    # 寻找极大值和极小值点，自定义寻找的范围：±5
    peaks, _ = find_peaks(column_sums, distance=10)
    valleys, _ = find_peaks(-column_sums, distance=10)

    # 突出显示极小值点，并进行比较
    for valley in valleys[:-1]:
        # 比较左边点和右边点的差值
        left_diff = column_sums[max(valley - 4, 0)] - column_sums[valley]
        right_diff = column_sums[min(valley + 4, len(column_sums) - 1)] - column_sums[valley]
        # print(left_diff, right_diff)

        if left_diff < 0 or right_diff < 0:
            # 舍弃极小值点
            continue

        # 计算变化值的平均值
        variations = column_sums[valley - 3 : valley + 3]
        variations = sorted(variations)[1:-1]  # 去除最高和最低值
        average_variation = np.mean(variations)

        if average_variation > 148000:
            # print(average_variation)
            y.append(valley)

    y.append(image.shape[0])

    return y


def block_image(image, x, y):
    blocks = []
    for i in range(len(y) - 1):
        for j in range(len(x) - 1):
            block = image[y[i] : y[i + 1], x[j] : x[j + 1]]
            blocks.append(block)
    return blocks


def main():
    # Read the image
    image_path = "img/cropped/17.jpg"
    image = cv2.imread(image_path)

    x, y = getX(image_path), getY(image_path)

    # Convert image to RGB color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Plot the image
    plt.imshow(image_rgb)

    # Draw vertical lines for x values (columns)
    for val in x:
        plt.axvline(x=val, color="red", linestyle="--")

    # Draw horizontal lines for y values (rows)
    for val in y:
        plt.axhline(y=val, color="blue", linestyle="--")

    # Block the image based on x and y coordinates
    blocks = block_image(image, x, y)

    # Display the blocks
    for block in blocks:
        plt.figure()
        plt.imshow(block)
        plt.axis("off")
        plt.show()

    # Show the image with grid
    plt.show()

    # Show the image with grid
    plt.show()


main()
