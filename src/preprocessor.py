import dlib
from align_dlib import AlignDlib
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
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

    def process(self, img):
        detected_faces = self.face_detector(img, 1)
        aligned_faces = []

        for face_r in detected_faces:
            pose_landmarks = self.face_pose_predictor(img, face_r)
            aligned_face = self.face_aligner.align(534, img, face_r, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
            aligned_faces.append(aligned_face)

        #TODO: resize to config['aspect_ratio'] , dimensions must be consistent

        return aligned_faces


    def data_from_db(self):
        pass


    def get_data(self):
        people = fetch_lfw_people(
            './data', min_faces_per_person=config['min_faces'], resize=config['aspect_ratio'])
        n_samples, h, w = people.images.shape
        X = people.data 
        n_features = X.shape[1]
        y = people.target
        target_names = people.target_names
        n_classes = target_names.shape[0]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25)
        return X_train, X_test, y_train, y_test, target_names, h, w


    #TODO: a returned instance errors on 'tried to change shape, not allowed' .. why?
    ## PREPROCESS : reduce dimensions / feature scaling
    def pca_eigenfaces(X_train, h, w):
        pca = RandomizedPCA(
            n_components=150, whiten=True).fit(X_train)
        eigenfaces = pca.components_.reshape((150, h, w))
        return pca




