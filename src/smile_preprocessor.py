import dlib
from align_dlib import AlignDlib

# This does everything that needs to be done to an image before it's ready
# for training/verification
class SmilePreprocessor(object):
    def __init__(self, predictor_model="shape_predictor_68_face_landmarks.dat"):
        self.predictor_model = predictor_model
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_pose_predictor = dlib.shape_predictor(predictor_model)
        self.face_aligner = AlignDlib(predictor_model)

    def process(self, img):
        detected_faces = self.face_detector(img, 1)
        aligned_faces = []

        for face_r in detected_faces:
            pose_landmarks = self.face_pose_predictor(img, face_r)
            aligned_face = self.face_aligner.align(534, img, face_r, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
            aligned_faces.append(aligned_face)

        return aligned_faces
