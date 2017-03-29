import dlib
from align_dlib import AlignDlib
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
import numpy as np
import cv2
from config import *
import os
import hashlib
from os import listdir
from os.path import join, exists, isdir

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


    def get_lfw_data(self):
        people = fetch_lfw_people('./data', resize=1.0, funneled=False, min_faces_per_person=config['min_faces'])
        X, not_found = self.process_raw_images(people.images)
        y = people.target
        y = np.delete(y, not_found)
        target_names = people.target_names
        return X, y, target_names

    ## for one person
    def load_test_data(self, dir_path):
        if not isdir(dir_path):
            return None, None
        label = dir_path.split('/')[-1] #name of person
        paths = [join(dir_path, f) for f in listdir(dir_path)]
        n_pictures = len(paths)
        images = []
        for p in paths:
            images.append(cv2.imread(p))
        images = np.array(images)
        target_labels = np.array([label for i in range(n_pictures)])
        images, not_found = self.process_raw_images(images)
        target_labels = np.delete(target_labels, not_found)
        return images, target_labels

    def get_small_data(self):
        person_names, file_paths = [], []
        for person_name in sorted(listdir(SMALL_DATA_PATH)):
            folder_path = join(SMALL_DATA_PATH, person_name)
            if not isdir(folder_path):
                continue
            paths = [join(folder_path, f) for f in listdir(folder_path)]
            n_pictures = len(paths)
            person_names.extend([person_name] * n_pictures)
            file_paths.extend(paths)
        target_names = np.unique(person_names)
        target = np.searchsorted(target_names, person_names)
        n_faces = len(file_paths)
        images = []
        for p in file_paths:
            images.append(cv2.imread(p))
        images = np.array(images)

        ## network training sensitive to consecutive same labels
        indices = np.arange(n_faces)
        np.random.RandomState(42).shuffle(indices)
        images, target = images[indices], target[indices]

        images, not_found = self.process_raw_images(images)
        target = np.delete(target, not_found)
        return images, target, target_names

    def get_data(self):
        if SMALL_MODEL:
            print("Using training data from small dataset")
            X, y, target_names = self.get_small_data()
        else:
            print("Using training data from sklearn")
            X, y, target_names = self.get_lfw_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        return X_train, X_test, y_train, y_test, target_names
