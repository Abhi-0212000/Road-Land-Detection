import cv2
import numpy as np



def applyHSV(blurreded_frame):
    # converting BGR to HSV
    hsv_blurred_frame = cv2.cvtColor(blurreded_frame, cv2.COLOR_BGR2HSV)

    # Usually road lanes are white or yellow. Now We filter out these colors out of the hsv_blurred_frame

    white_lower = np.array([0, 0, 170])   # white can be obtained from all the colors whith Value (lightness) in range of 170, 255
    white_upper = np.array([255, 255, 255])

    yellow_lower = np.array([13, 100, 0])
    yellow_upper = np.array([50, 255, 255])

    white_mask = cv2.inRange(hsv_blurred_frame, white_lower, white_upper)
    yellow_mask = cv2.inRange(hsv_blurred_frame, yellow_lower, yellow_upper)

    colored_mask = cv2.bitwise_or(yellow_mask, white_mask)
    colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_GRAY2BGR)

    color_masked_frame = cv2.bitwise_and(blurreded_frame, colored_mask)

    return color_masked_frame

def canny(blurreded_frame):
    #blurreded_frame = cv2.GaussianBlur(gray_frame, (3, 3), cv2.BORDER_DEFAULT)
    canny_frame = cv2.Canny(blurreded_frame, 60, 150)
    return canny_frame

def roadROIMask(canny_frame):
    height, width = canny_frame.shape
    road_mask = np.zeros_like(canny_frame)
    triangular_roi = np.array([(200, height), (1100, height), (575, 250)])  # array of 3 co-ordinates of triangular ROI
    road_roi_mask = cv2.fillPoly(road_mask, [triangular_roi], 255)  # We pass the triangular_roi of co-ords as a list 
    return road_roi_mask

def detectLines(masked_canny_frame):
    hough_lines = cv2.HoughLinesP(image=masked_canny_frame, rho=2, theta=np.pi/180, threshold=100, minLineLength = 40, maxLineGap=5)
    detected_lines = []
    if hough_lines is not None:
        for hough_line in hough_lines:
            detected_lines.append(hough_line.reshape(4))      
    #detected_lines = detected_lines.flatten()
    #print(detected_lines)
    return detected_lines

def linesMask(frame, detected_lines):
    height, width, _ = frame.shape
    lines_mask = np.zeros_like(frame)
    if detected_lines is not None:
        for x1, y1, x2, y2 in detected_lines:
            #print(detected_line)
            #x1, y1, x2, y2 = detected_line.flatten()
            #print(type(x1), y1, x2, y2)
            cv2.line(lines_mask, (x1, y1), (x2, y2), (0, 255, 0), 6)
    return lines_mask
    #for i in 
    #cv2.line(lines_mask, lines_mask, )



def optimizedLinesCoords(frame, optimized_lines):
    height, width, _ = frame.shape
    global last_detected_left_slope, last_detected_left_intercept, last_detected_right_slope, last_detected_right_intercept
    #print(not np.isnan(optimized_lines[0]).all())
    if optimized_lines is not None:
        if not np.isnan(optimized_lines[0]).all():
            left_slope, left_intercept = optimized_lines[0]
            last_detected_left_slope, last_detected_left_intercept = left_slope, left_intercept

            left_line_y1 = height   # we want to draw the line from botoom of frame
            left_line_y2 = int(height * (2/3))  # We are drawing the line upto half the frame height
            left_line_x1 = int((left_line_y1 - left_intercept) / left_slope)  # using y = mx + c  --> x = (y-c)/m
            left_line_x2 = int((left_line_y2 - left_intercept) / left_slope)
        else:
            left_slope, left_intercept = last_detected_left_slope, last_detected_left_intercept
            left_line_y1 = height   # we want to draw the line from botoom of frame
            left_line_y2 = int(height * (2/3))  # We are drawing the line upto half the frame height
            left_line_x1 = int((left_line_y1 - left_intercept) / left_slope)  # using y = mx + c  --> x = (y-c)/m
            left_line_x2 = int((left_line_y2 - left_intercept) / left_slope)

        if not np.isnan(optimized_lines[1]).all():
            right_slope, right_intercept = optimized_lines[1]
            last_detected_right_slope, last_detected_right_intercept = right_slope, right_intercept

            right_line_y1 = height   # we want to draw the line from botoom of frame
            right_line_y2 = int(height * (2/3))  # We are drawing the line upto half the frame height
            right_line_x1 = int((right_line_y1 - right_intercept) / right_slope)  # using y = mx + c  --> x = (y-c)/m
            right_line_x2 = int((right_line_y2 - right_intercept) / right_slope)
        else:
            right_slope, right_intercept = last_detected_right_slope, last_detected_right_intercept

            right_line_y1 = height   # we want to draw the line from botoom of frame
            right_line_y2 = int(height * (2/3))  # We are drawing the line upto half the frame height
            right_line_x1 = int((right_line_y1 - right_intercept) / right_slope)  # using y = mx + c  --> x = (y-c)/m
            right_line_x2 = int((right_line_y2 - right_intercept) / right_slope)

        optimized_lines_coords = [(left_line_x1, left_line_y1), (left_line_x2, left_line_y2), (right_line_x1, right_line_y1), (right_line_x2, right_line_y2)]

    return optimized_lines_coords

def optimizedRegion(optimized_lines_mask, optimized_lines_coords):
    polygon_coords = [optimized_lines_coords[0], optimized_lines_coords[1], optimized_lines_coords[3], optimized_lines_coords[2]]
    polygon_coords = np.array(polygon_coords, np.int32)
    #polygon_coords = polygon_coords.reshape((-1, 1, 2))

    cv2.fillPoly(optimized_lines_mask, [polygon_coords], color=(0, 255, 127))
    return optimized_lines_mask


def optimizedLinesMask(frame, optimized_lines_coords):

    left_lower  = optimized_lines_coords[0]
    left_upper = optimized_lines_coords[1]

    right_lower = optimized_lines_coords[2]
    right_upper = optimized_lines_coords[3]

    optimized_lines_mask = np.zeros_like(frame)
    cv2.line(optimized_lines_mask, left_lower, left_upper, (0, 0, 255), 20)
    cv2.line(optimized_lines_mask, right_lower, right_upper, (0, 0, 255), 20)

    return optimized_lines_mask

def optimizedLines(detected_lines):
    left_lines = []
    right_lines = []
    if detected_lines is not None:
        for x1, y1, x2, y2 in detected_lines:
            coeff = np.polyfit([x1, x2], [y1, y2], 1)
            if coeff[0] < 0:
                left_lines.append(coeff)
            else:
                right_lines.append(coeff)
    if len(left_lines) > 0:
        avg_left_line = np.average(left_lines, axis=0) # returns a list = [avg slope, avg intercept]
    else:
        avg_left_line = []
    if len(right_lines) > 0:
        avg_right_line = np.average(right_lines, axis=0)
    else:
        avg_right_line = []

    return [avg_left_line, avg_right_line]

# Global variables to store last detected slpe, intercepts. We use these values if road lines are not visible (may be road line is not painted for some distance)

last_detected_left_slope = 0
last_detected_left_intercept = 0
last_detected_right_slope = 0
last_detected_right_intercept = 0

# Applying code on a video

cap = cv2.VideoCapture('./data/test_video.mp4')
success, frame = cap.read()

while success:
    
    #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurreded_frame = cv2.GaussianBlur(frame, (3, 3), cv2.BORDER_DEFAULT)
    color_masked_frame = applyHSV(blurreded_frame)
    gray_frame = cv2.cvtColor(color_masked_frame, cv2.COLOR_BGR2GRAY)
    canny_frame = canny(blurreded_frame)
    #canny_frame = canny(gray_frame)
    road_roi_mask = roadROIMask(canny_frame)
    masked_canny_frame = cv2.bitwise_and(canny_frame, road_roi_mask)
    detected_lines = detectLines(masked_canny_frame)

    # For visualizing all the lines detected.
    # Uncomment the below code and comment the Optimization code to visualize this.
    '''lines_mask = linesMask(frame, detected_lines)
    frame = cv2.addWeighted(frame, 0.6, lines_mask, 0.4, 0.0)  # for all the lines detected'''


    # For Optimization lines (1 left, 1 right line) visualization
    optimized_lines = optimizedLines(detected_lines)
    optimized_lines_coords = optimizedLinesCoords(frame, optimized_lines)
    optimized_lines_mask = optimizedLinesMask(frame, optimized_lines_coords)
    optimized_region_to_drive = optimizedRegion(optimized_lines_mask, optimized_lines_coords)
    frame = cv2.addWeighted(frame, 0.6, optimized_region_to_drive, 0.4, 0.0)  # for optimized left, right line'''

    #cv2.namedWindow('test_video', cv2.WINDOW_NORMAL)
    cv2.imshow('test_video', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

    success, frame = cap.read()

cap.release()
cv2.destroyAllWindows()






# Applying code on Single image
""" 

frame = cv2.imread(img)
blurreded_frame = cv2.GaussianBlur(frame, (3, 3), cv2.BORDER_DEFAULT)
color_masked_frame = applyHSV(blurreded_frame)
gray_frame = cv2.cvtColor(color_masked_frame, cv2.COLOR_BGR2GRAY)
canny_frame = canny(blurreded_frame)
#canny_frame = canny(gray_frame)
road_roi_mask = roadROIMask(canny_frame)
masked_canny_frame = cv2.bitwise_and(canny_frame, road_roi_mask)
detected_lines = detectLines(masked_canny_frame)

# For visualizing all the lines detected.
# Uncomment the below code and comment the Optimization code to visualize this.
'''lines_mask = linesMask(frame, detected_lines)
frame = cv2.addWeighted(frame, 0.6, lines_mask, 0.4, 0.0)  # for all the lines detected'''


# For Optimization lines (1 left, 1 right line) visualization
optimized_lines = optimizedLines(detected_lines)
print(optimized_lines)
optimized_lines_coords = optimizedLinesCoords(frame, optimized_lines)
print(optimized_lines_coords)
optimized_lines_mask = optimizedLinesMask(frame, optimized_lines_coords)
optimized_region_to_drive = optimizedRegion(optimized_lines_mask, optimized_lines_coords)
frame = cv2.addWeighted(frame, 0.6, optimized_region_to_drive, 0.4, 0.0)  # for optimized left, right line'''

cv2.namedWindow('test_video', cv2.WINDOW_NORMAL)
cv2.imshow('test_video', frame)
cv2.waitKey(100)

"""
