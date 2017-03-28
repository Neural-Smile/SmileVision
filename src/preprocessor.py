import dlib
from align_dlib import AlignDlib
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
import numpy as np
import cv2
from config import *
import os
import hashlib


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

    def unflatten(self, img, h=processed_height, w=processed_width):
        return np.reshape(img, (h, w))

    def process(self, img):
        if len(img.shape) > 2 and img.shape[2] != 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detected_faces = self.face_detector(img, 1)
        aligned_faces = []
        for face_r in detected_faces:
            pose_landmarks = self.face_pose_predictor(img, face_r)
            aligned_face = self.face_aligner.align(processed_height, img, face_r, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
            aligned_face = cv2.resize(aligned_face, (processed_height, processed_width))
            aligned_face = aligned_face.flatten()
            aligned_faces.append(aligned_face)
        return aligned_faces

    def data_from_db(self):
        pass

    def cache_file(self, hash):
        return "{}/{}.npy".format(PREPROCESSOR_CACHE_PATH, hash)

    def cache_processed_image(self, processed, hash):
        if not os.path.exists(PREPROCESSOR_CACHE_PATH):
            print("directory {} doesn't existing creating it".format(PREPROCESSOR_CACHE_PATH))
            os.makedirs(PREPROCESSOR_CACHE_PATH)
        try:
            f = open(self.cache_file(hash), "wb")
            np.save(f, processed)
        finally:
            f.close()
        return

    def processed_image_from_cache(self, hash):
        if not os.path.exists(self.cache_file(hash)):
            return None

        res = None
        try:
            f = open(self.cache_file(hash), "rb")
            res = np.load(f)
        finally:
            f.close()
        return res

    def hash_arr(self, arr):
        a = arr.view(np.uint8)
        return hashlib.sha1(a).hexdigest()

    def process_raw_images(self, images):
        processed_images = []
        not_found = []
        for i, image in enumerate(images):
            image_hash = self.hash_arr(image)
            cache = self.processed_image_from_cache(image_hash)
            if cache is not None:
                processed_images.append(cache)
                continue
            image = image.astype(np.uint8)
            faces = self.process(image)
            if len(faces) == 1:
                processed_images.append(faces[0])
                self.cache_processed_image(faces[0], image_hash)
            else:
                not_found.append(i)

        return np.array(processed_images), not_found

    def load_smile_picset(self, person):
        images = []
        path = "{}/faces/{}".format(SMILE_DATASET_PATH, person)
        if not os.path.exists(path):
            print("Images for {} not found at {}".format(person, path))
            exit(1)

        for (dirpath, dirnames, filenames) in os.walk(path):
            for f in filenames:
                p = dirpath + "/" + f
                images.append(cv2.imread(p))
            break
        return images

    def get_smile_data(self):
        known_identities = ["Hormoz_Kheradmand", "Colin_Armstrong"] # TODO: replace with a directory walker
        people = []
        images = []
        for identity in known_identities:
            identity_picset = self.load_smile_picset(identity)
            images += identity_picset
            people += [identity for _ in range(len(identity_picset))]
        people = np.array(people)
        images = np.array(images)
        images, not_found = self.process_raw_images(images)
        people = np.delete(people, not_found)
        return images, people, people

    def get_lfw_data(self):
        people = fetch_lfw_people('./data', resize=1.0, funneled=False, min_faces_per_person=config['min_faces'])
        X, not_found = self.process_raw_images(people.images)
        y = people.target
        y = np.delete(y, not_found)
        target_names = people.target_names
        return X, y, target_names

    def get_data(self):
        X, y, target_names = self.get_smile_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        return X_train, X_test, y_train, y_test, target_names
