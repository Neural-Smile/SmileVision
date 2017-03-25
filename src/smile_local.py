import cv2
from smile_preprocessor import SmilePreprocessor
from smile_verifier import SmileVerifier

# Local development helper, pressing spacebar while the utility
# is running will the run the captured frame through SmilePreprocessor
# and SmileVerifier. Use it for faster testing
class SmileLocal(object):
    def __init__(self, camera=0):
        self.camera = cv2.VideoCapture(camera)
        self.preprocessor = SmilePreprocessor()
        self.verifier = SmileVerifier()
        self.cur_frame = None

    def process_frame(self):
        processed_faces = self.preprocessor.process(self.cur_frame)
        for f in processed_faces:
            self.verifier.verify(f)

    def run(self):
        while True:
            (grabbed, self.cur_frame) = self.camera.read()

            if not grabbed:
                print "could not grab frame from camera"
                break

            cv2.imshow("Frame", self.cur_frame)

            self.handle_interrupts()

    def cleanup(self):
        self.camera.release()
        cv2.destroyAllWindows()

    def handle_interrupts(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            self.process_frame()
        if key == ord("q"):
            raise SystemExit("Exiting..")


def main():
    l = SmileLocal()
    try:
        l.run()
    finally:
        l.cleanup()

if __name__ == "__main__":
    main()
