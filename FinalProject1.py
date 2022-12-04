import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def lanesDetection(img):
    # img = cv.imread("images/Test_road_2.jpg")
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    print(img.shape)
    height = img.shape[0]
    width = img.shape[1]

    # region_of_interest_vertices = [
    #     (300, height), (width/2, height/1.4), (width-400, height)

    # region_of_interest_vertices = [
    #     (680, height-10), (width/2.2, height/1.78), (width+200, height-100)
    # ]

    region_of_interest_vertices = [
        (500, height-10), (width/2.2, height/1.78), (width+100, height-100)
    ]
    
    gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    smoot = cv.GaussianBlur(gray_img,(5,5),0.2)
    smooth = cv.medianBlur(smoot,7)
    # smooth = cv.bilateralFilter(gray_img,15,75,75)
    # edge = cv.Canny(gray_img, 50, 100, apertureSize=3)
    edge = cv.Canny(smooth, 180,240)

    # lines = cv.HoughLinesP(img, 1, np.pi/180, 20 , np.array([]), 20 ,180)
    # line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # #draw_lines(line_img, lines)
    # line_img = slope_lines(line_img,lines)
    # houghed_lines = hough_lines(img = masked_img, rho = 1, theta = np.pi/180, threshold = 20, min_line_len = 20, max_line_gap = 180)
    
    
    cropped_image = region_of_interest(edge, np.array([region_of_interest_vertices], np.int32))
    

    lines = cv.HoughLinesP(cropped_image, rho=2, theta=np.pi/180,
                           threshold=40, lines=np.array([]), minLineLength=4, maxLineGap=5)
    image_with_lines = draw_lines(img, lines)
    # plt.imshow(image_with_lines)
    # plt.show()
    return image_with_lines


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    # channel_count = img.shape[2]
    match_mask_color = (255)
    cv.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(blank_image, (x1, y1), (x2, y2), (255, 255 , 0), 2)

    img = cv.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img


def rescaleFrame(frame, scale):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimensions = (width,height)

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

def videoLanes():
    cap = cv.VideoCapture('videos/road_lane.mp4')
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = lanesDetection(frame)
        frame_resized = rescaleFrame(frame,1)
        cv.imshow('Lanes Detection', frame_resized)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()



# capture = cv.VideoCapture('videos/road_lane.mp4')

# while True:
#     isTrue, frame=capture.read()

#     frame_resized = rescaleFrame(frame,scale=0.25)

#     cv.imshow('Video',frame)
#     cv.imshow('Video resized',frame_resized)

#     if cv.waitKey(20) & 0xFF==ord('d'):
#         break

# capture.release()

if __name__ == "__main__":
    videoLanes()