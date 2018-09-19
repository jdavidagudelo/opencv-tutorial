import cv2
from matplotlib import pyplot as plt


def read_image(file_path, flags=cv2.IMREAD_COLOR):
    img = cv2.imread(file_path, flags=flags)
    return img


def show_image(img, image_name='Image'):
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


def show_images(images):
    fig = plt.figure(figsize=(1, len(images)))
    i = 1
    for img in images:
        fig.add_subplot(1, len(images), i)
        plt.imshow(img)
        i += 1
    plt.xticks([]), plt.yticks([])
    plt.show()


def save_image(img, file_path):
    cv2.imwrite(file_path, img)