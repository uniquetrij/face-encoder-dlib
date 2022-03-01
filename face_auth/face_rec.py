import os

import face_recognition
from sklearn import svm
from sklearn.metrics.pairwise import cosine_similarity

ENCODINGS = {}
CLASSIFIER = None


def build_clf():
    global CLASSIFIER
    global ENCODINGS
    ENCODINGS.clear()
    encodings = []
    names = []
    dir = 'face_auth/faces'
    # Training directory
    if dir[-1] != '/':
        dir += '/'
    train_dir = os.listdir(dir)

    # Loop through each person in the training directory
    for person in train_dir:
        pix = os.listdir(dir + person)

        # Loop through each training image for the current person
        for person_img in pix:
            # Get the face encodings for the face in each image file
            face = face_recognition.load_image_file(
                dir + person + "/" + person_img)
            face_bounding_boxes = face_recognition.face_locations(face)

            # If training image contains exactly one face
            if len(face_bounding_boxes) == 1:
                face_enc = face_recognition.face_encodings(face)[0]
                # Add face encoding for current image
                # with corresponding label (name) to the training data
                encodings.append(face_enc)
                names.append(person)
                if person not in ENCODINGS.keys():
                    ENCODINGS[person] = []
                ENCODINGS[person].append(face_enc
                                         )
            else:
                print(person + "/" + person_img + " can't be used for training")

    CLASSIFIER = svm.SVC(gamma='scale', probability=True)
    CLASSIFIER.fit(encodings, names)


def predict(query):
    prediction = None
    score = 0
    for person, encodings in ENCODINGS.items():
        p_score = cosine_similarity(query.reshape(1, -1), encodings)
        if p_score > score:
            score = p_score
            prediction = person
    return prediction, score[0][0]