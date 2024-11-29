import cv2

# video = 'http://86.127.212.219/cgi-bin/faststream.jpg?stream=half&fps=15&rand=COUNTER'
# capture video/ video path
cap = cv2.VideoCapture('cars.mp4')

car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

DANGER_ZONE = [450, 0, 450, 200]
# {
#     "left-top": (450, 0),
#     "right-bottom": (900, 200)
# }


def get_rectangle_points(shape):
    return {
        "top_left": (shape[0], shape[1]),
        "top_right": (shape[0] + shape[2], shape[1]),
        "bottom_left": (shape[0], shape[1] + shape[3]),
        "bottom_right": (shape[0] + shape[2], shape[1] + shape[3]),
    }


def is_point_bounded_by_rect(point, rect):
    return rect["top_left"][0] < point[0] and rect["top_left"][1] < point[1] and rect["bottom_right"][0] > point[0] and rect["bottom_right"][1] > point[1]

def is_a_bounded_by_b(a, b):
    a_rect = get_rectangle_points(a)
    b_rect = get_rectangle_points(b)
    return all([
        is_point_bounded_by_rect(a_rect["top_left"], b_rect),
        is_point_bounded_by_rect(a_rect["top_right"], b_rect),
        is_point_bounded_by_rect(a_rect["bottom_left"], b_rect),
        is_point_bounded_by_rect(a_rect["bottom_right"], b_rect),
    ])


# read until video is completed/stream is aborted
while True:
    # capture frame by frame
    ret, frame = cap.read()
    # convert frame into gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # draw a danger zone
    cv2.rectangle(
        frame,
        (DANGER_ZONE[0], DANGER_ZONE[1]),
        (DANGER_ZONE[0] + DANGER_ZONE[2], DANGER_ZONE[1] + DANGER_ZONE[3],),
        BLACK,
        2
    )

    # detect cars in the video
    cars = car_cascade.detectMultiScale(gray, 1.1, 3, 0, (10, 10))
    print(cars[0])

    # to draw a rectangle for each car
    for car in cars:
        (x,y,w,h) = car
        cv2.rectangle(frame,(x,y),(x+w,y+h),RED if is_a_bounded_by_b(car, DANGER_ZONE) else GREEN,2)
        cv2.imshow('video', frame)
        crop_img = frame[y:y+h,x:x+w]

    # press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


#release the video-capture object
cap.release()
#close all the frames
cv2.destroyAllWindows()