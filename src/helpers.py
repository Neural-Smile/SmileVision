import cv2
import time

# shows an image until user presses 'q'
def show_image(img):
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow("Frame", 1920 , 1080)
    cv2.imshow("Frame", img)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break

# save any img with a unique timestamp so you can view it later
def save_image(img):
    timestamp = str(time.time())
    timestamp = timestamp.replace('.', '')
    cv2.imwrite("img_{}.png".format(timestamp), img)
