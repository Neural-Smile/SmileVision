import cv2
from preprocessor import Preprocessor
from model import Model
from config import *
import numpy as np
import os
import time

# Local development helper, pressing spacebar while the utility
# is running will the run the captured frame through SmilePreprocessor
# and SmileVerifier. Use it for faster testing
class SmileLocal(object):
    def __init__(self, camera=0):
        self.camera_num = camera
        self.camera = None
        self.cur_frame = None
        self.preprocessor = Preprocessor()
        self.model = Model(self.preprocessor)
        self.model.initialize()

    def process_frame(self, frame=None):
        if frame is None:
            frame = self.cur_frame

        found_faces = self.preprocessor.process(frame)
        if len(found_faces) > 1:
            print("Cannot train on more than one face in the image")
            return
        if len(found_faces) > 0:
            return self.process_face(found_faces[0])

    def process_face(self, face):
        embedding = self.model.get_face_embeddings(np.array([face]))
        identity = self.model.verify(embedding)
        print(identity)
        return identity

    def get_camera(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(self.camera_num)
        return self.camera

    def save_user_imgs(self, identity, imgs):
        folder_path = SMILE_DATA_PATH + "/" + identity
        if not os.path.exists(folder_path):
            print("Creating directory {}".format(folder_path))
            os.makedirs(folder_path)
        for img in imgs:
            timestamp = str(time.time()).replace(".", "")
            filename = folder_path + "/" + timestamp + ".png"
            cv2.imwrite(filename, img)

    def train_new_identity(self, identity, imgs):
        imgs = np.array(imgs)
        self.save_user_imgs(identity, imgs)
        return self.model.train_new_identity(identity, imgs)

    def run(self):
        while True:
            (grabbed, self.cur_frame) = self.get_camera().read()

            if not grabbed:
                print "could not grab frame from camera"
                break

            identity = self.process_frame()
            print(identity)
            cv2.imshow("Frame", self.cur_frame)
            self.handle_interrupts()

    def initialize_and_test(self):
        self.model.initialize_and_test()

    def cleanup(self):
        self.get_camera().release()
        cv2.destroyAllWindows()
        self.model.save_model()

    def handle_interrupts(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            self.process_frame()
        if key == ord("q"):
            raise SystemExit("Exiting..")


def main():
    l = SmileLocal()
    try:
        l.initialize_and_test()
        l.run()
    finally:
        l.cleanup()

if __name__ == "__main__":
    main()
