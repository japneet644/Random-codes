from PIL import Image, ImageDraw
import math
import numpy as np

def draw_grid(lattice_size,angle):
    height = 640
    width = 640
    image = Image.new(mode='L', size=(height, width), color=255)

    # Draw some lines
    draw = ImageDraw.Draw(image)
    y_start = 0
    y_end = image.height
    step_size = int(image.width / lattice_size)

    for x in range(0, image.width, step_size):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill=128)

    x_start = 0
    x_end = image.width

    for y in range(0, image.height, step_size):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill=128)
    # del draw
    a = step_size//2
    for i in range(0, image.width, step_size):
        for j in range(0, image.height, step_size):
            draw.line(((i+a, j+a) , ( i + a + a*math.cos(angle[i//step_size,j//step_size]), j + a + a*math.sin(angle[i//step_size,j//step_size]))))
            draw.line(((i+a, j+a) , ( i + a - a*math.cos(angle[i//step_size,j//step_size]), j + a - a*math.sin(angle[i//step_size,j//step_size]))))

    image.show()

if __name__ == '__main__':
    n = 32
    angle = np.random.random((n,n))
    draw_grid(n,angle)
