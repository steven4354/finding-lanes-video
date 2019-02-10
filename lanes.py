import cv2
import numpy as np
import matplotlib.pyplot as plt

# TODO: place in section comments
# TODO: check why I can't put lane_image in the combo image, but canny and anyting after works
def simplify_img(image):
    # change the image to gray for faster processing
    # reasoning: 1 value in each pixel array rather than 3 (RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply a blur to prevent detection of false edges
    # basically a 5 x 5 "kernel" is placed throughout image and the center pixel #
    # becomes the avg of the pixel values in the kernel
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # TODO: get the 0 value

    # use the Canny function to give an outline of the areas that have a gradient
    # aka it takes a derivative of two pixels and does this in all directions -- | /
    # basically a white line is made anywhere that exceeds the second gradient input
    # gradients between first gradient and second will only be white if it is connected to a pixel with gradient above ^^
    # everything else is blacked out
    canny = cv2.Canny(blur, 50, 150)
    return canny

def show_polygon_in_lane_pic(image):
    # gets the bottom of the image (max # - just show image to see)
    height = image.shape[0]
    # array of triangles
    # each triangle array [point A, point B, point C]
    # the non-height numbers were just eye-balled using the plot
    polygons = np.array([
    [(200, height), (1100, height), (550, 250)]
    ])
    # blacken everything in image
    mask = np.zeros_like(image)
    # take black image and place a polygon inside, fill everything in the poly with color 255
    cv2.fillPoly(mask, polygons, 255)
    return mask

def remove_all_outside_region_of_car_lane(image):
    # get the polygon on black image
    polygon_mask_image = show_polygon_in_lane_pic(image)
    # overlay ^^ into originall via bitwise
    # aka black + black = black (1 && 1 = 1)
    #     white + black = balck (0 && 1 = 1)
    #     white + white = white (0 && 0 = 0)
    masked_image = cv2.bitwise_and(image, polygon_mask_image)
    return masked_image

def display_lines(image, lines):
    # create a black image the size of the input image
    line_image = np.zeros_like(image)

    # take each line (point values) and draw into our image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            # (1) iamge to use
            # (2) & (3) coordinates of line to draw
            # (4) color of the line drawn & (5) thickness of line
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    return line_image

# line_parameters as in [m, yintercept]
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    # y1 is the y coordinate of the bottom of the image
    # matplotlib this to see
    y1 = image.shape[0]
    y2 = int(y1 * 3/5)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        # just pulls out the points in the array
        x1, y1, x2, y2 = line.reshape(4)
        # (3) if it is one tells to use the line type y=mx+b (instead of ~ 1 = cox(theta)x + sin(theta)y)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        # will give array of [yintercept, slope]
        # print(paramters)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    #  axis=0 https://docs.scipy.org/doc/numpy-1.14.5/reference/generated/numpy.average.html
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    print(left_line, "left line")
    print(right_line, "right line")
    return np.array([left_line, right_line])

# reads the image & returns a multidimension array
# np.copy is just a clone of it so we don't edit the original
# then gets a black and white edge (gradient) - image
image = cv2.imread('test_image.jpg')
image_copy = np.copy(image)
canny = simplify_img(image_copy)

# gets just the lanes relevant to this car
# TODO: fix this name
just_lane_image = remove_all_outside_region_of_car_lane(canny)

# create mathematical lines on those lanes
# lines holds the points for the lines
# (1) image to use,
# (2) and (3) describe how large each bin should be aka the kernel see: https://www.udemy.com/applied-deep-learningtm-the-complete-self-driving-car-course/learn/v4/t/lecture/11241794?start=0
# (4) is the lowest # of bins for a line to be considered
# (5) is just a placeholder for the output
# (6) is threshold by length of the line to show up
# (7) is the min gap length between two lines to not have them merged into 1 line
lines = cv2.HoughLinesP(just_lane_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# image of just the lines
line_image = display_lines(just_lane_image, lines)
# combo of the lines image and the original image
# (1) image (2) "weight / darkness"
combo_image = cv2.addWeighted(canny, 0.8, line_image, 1, 1)

# merge the segment lines together to make 2 giant lines
averaged_lines = average_slope_intercept(just_lane_image, lines)
print(averaged_lines, "averaged_lines")
averaged_line_image = display_lines(just_lane_image, averaged_lines)
new_combo_image = cv2.addWeighted(canny, 0.8, averaged_line_image, 1, 1)

# shows the image when ran (opens in a new window)
# @waitKey sets the opened image to stay open until x is pressed
# cv2.imshow('result', canny)
# cv2.waitKey(0)

# shows the image with a graph
plt.imshow(new_combo_image)
plt.show()
