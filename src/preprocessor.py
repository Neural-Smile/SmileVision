import dlib
from align_dlib import AlignDlib
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
import numpy as np
import cv2
from config import *


# This does everything that needs to be done to an image before it's ready
# for training/verification
class Preprocessor(object):
    def __init__(self, predictor_model="shape_predictor_68_face_landmarks.dat"):
        # we use a pretrained model that can be downloaded from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        # download it to the repo root and unzip using `bunzip2 [filename]`
        self.predictor_model = predictor_model
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_pose_predictor = dlib.shape_predictor(predictor_model)
        self.face_aligner = AlignDlib(predictor_model)

    def flatten(self, img):
        return img.flatten()

    def unflatten(self, img, h, w):
        return np.reshape(img, (h, w))

    def process(self, img):
        if len(img.shape) > 2 and img.shape[2] != 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detected_faces = self.face_detector(img, 1)
        aligned_faces = []
        for face_r in detected_faces:
            pose_landmarks = self.face_pose_predictor(img, face_r)
            aligned_face = self.face_aligner.align(H, img, face_r, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
            aligned_face = cv2.resize(aligned_face, (H, W))
            aligned_face = aligned_face.flatten()
            aligned_faces.append(aligned_face)
        return aligned_faces

    def data_from_db(self):
        pass

    def process_lfw_images(self, images):
        processed_images = []
        not_found = []
        for i, image in enumerate(images):
            image = image.astype(np.uint8)
            faces = self.process(image)
            if len(faces) == 1:
                processed_images.append(faces[0])
            else:
                not_found.append(i)

        return np.array(processed_images), not_found

    def get_data(self):
        people = fetch_lfw_people('./data', resize=1.0, funneled=False, min_faces_per_person=config['min_faces'])
        X, not_found = self.process_lfw_images(people.images)
        y = people.target
        y = np.delete(y, not_found)
        target_names = people.target_names
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        return X_train, X_test, y_train, y_test, target_names


    #TODO: a returned instance errors on 'tried to change shape, not allowed' .. why?
    ## PREPROCESS : reduce dimensions / feature scaling
    def pca_eigenfaces(X_train, h, w):
        pca = RandomizedPCA(
            n_components=150, whiten=True).fit(X_train)
        eigenfaces = pca.components_.reshape((150, h, w))
        return pca
