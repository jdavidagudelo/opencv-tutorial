from basic import draw_images
from basic import load_images
import numpy

img = load_images.read_image('basic/images/image1.jpeg')
img = draw_images.draw_line(img, 0, 0, 300, 300, 0, 255, 0, 5)
img = draw_images.draw_circle(img, 400, 400, 30, 255, 0, 0, 1)
img = draw_images.draw_rectangle(img, 250, 250, 300, 300, 0, 0, 255, 3)
img = draw_images.draw_ellipse(img, 200, 200, 300, 300, 0, 0, 255, 255, 255, 0, 6)
pts = numpy.array([[10, 5], [20, 30], [70, 20], [50, 10]], numpy.int32)
pts = pts.reshape((-1, 1, 2))
img = draw_images.draw_polygon(img, pts, True, 255, 0, 0, 5)
img = draw_images.draw_text(img, 'Hello World', 300, 200, 0, 0, 255, 2, 5)
load_images.show_image(img)
