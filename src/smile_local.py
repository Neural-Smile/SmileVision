import cv2
from preprocessor import Preprocessor
from model import Model
import numpy as np

# Local development helper, pressing spacebar while the utility
# is running will the run the captured frame through SmilePreprocessor
# and SmileVerifier. Use it for faster testing
class SmileLocal(object):
    def __init__(self, camera=0):
        self.camera = cv2.VideoCapture(camera)
        self.cur_frame = None
        self.preprocessor = Preprocessor()
        self.model = Model(self.preprocessor)
        self.model.initialize()

    def process_frame(self):
        found_faces = self.preprocessor.process(self.cur_frame)
        if len(found_faces) > 1:
            print("Cannot train on more than one face in the image")
            return
        if len(found_faces) > 0:
            self.process_face(found_faces[0])

    def process_face(self, face):
        embedding = self.model.get_face_embeddings(np.array([face]))
        print(self.model.verify(embedding))

    def run(self):
        while True:
            (grabbed, self.cur_frame) = self.camera.read()

            if not grabbed:
                print "could not grab frame from camera"
                break

            cv2.imshow("Frame", self.cur_frame)
            self.handle_interrupts()

    def initialize_and_test(self):
        self.model.initialize_and_test()

    def cleanup(self):
        self.camera.release()
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
