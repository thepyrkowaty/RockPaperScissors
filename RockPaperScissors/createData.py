import cv2
import os
import pickle
import time
import random
import shutil

CATEGORIES = ['Rock', 'Paper', 'Scissors', 'None']
path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'Data')
gather = False
index = 0
unique_index = time.time()
indexOfTestPhoto = 0
numberOfTrainPhotos = 300
photoIndex = 0
displayPhotoIndex = 0
trainingData = []

# Get Cam
cap = cv2.VideoCapture(0)

# Prepare default photo path (Rock)
photoPath = os.path.join(path, CATEGORIES[index])

# Create directories to store data
try:
    os.mkdir(path)
    for category in CATEGORIES:
        os.mkdir(os.path.join(path, category))
except FileExistsError:
    shutil.rmtree(path)
    os.mkdir(path)
    for category in CATEGORIES:
        os.mkdir(os.path.join(path, category))


# Prepare photo for model
def preparePhoto(myImg):
    IMG_SIZE = 100
    myImg = cv2.cvtColor(myImg, cv2.COLOR_BGR2GRAY)
    myImg = cv2.resize(myImg, (IMG_SIZE, IMG_SIZE))
    myImg = myImg/255.0
    return myImg.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


def drawArea(img, name):
    cv2.rectangle(img, (300, 20), (700, 420), (0, 255, 255), thickness=2)
    # To make a black outline
    cv2.putText(img, name, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=6)
    cv2.putText(img, name, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2)


def drawMenu(img):
    cv2.putText(img, 'Press b to collect data', (10, 500), cv2.FONT_ITALIC, 0.65, (0, 0, 0), thickness=2)
    cv2.putText(img, 'Press e to change type of data', (10, 530), cv2.FONT_ITALIC, 0.65, (0, 0, 0), thickness=2)
    cv2.putText(img, 'Press s to save data', (10, 560), cv2.FONT_ITALIC, 0.65, (0, 0, 0), thickness=2)
    cv2.putText(img, 'Press q to quit', (10, 590), cv2.FONT_ITALIC, 0.65, (0, 0, 0), thickness=2)
    # To make a black outline
    cv2.putText(img, 'Press b to collect data', (10, 500), cv2.FONT_ITALIC, 0.65, (255, 255, 255), thickness=1)
    cv2.putText(img, 'Press e to change type of data', (10, 530), cv2.FONT_ITALIC, 0.65, (255, 255, 255), thickness=1)
    cv2.putText(img, 'Press s to save data', (10, 560), cv2.FONT_ITALIC, 0.65, (255, 255, 255), thickness=1)
    cv2.putText(img, 'Press q to quit', (10, 590), cv2.FONT_ITALIC, 0.65, (255, 255, 255), thickness=1)


# Main loop
while True:
    success, img = cap.read()
    img = cv2.resize(img, (800, 600))
    handImg = img[22:418, 302:698]
    drawArea(img, CATEGORIES[index])
    drawMenu(img)
    if gather:
        cv2.imwrite(os.path.join(photoPath, f'{photoIndex}.jpg'), handImg)
        photoIndex += 1
        displayPhotoIndex +=1
        trainingData.append([preparePhoto(handImg), CATEGORIES.index(CATEGORIES[index])])
        cv2.putText(img, f'Gathered {displayPhotoIndex} photos', (300, 450), cv2.FONT_ITALIC, 1, (255, 255, 255))
        numberOfTrainPhotos -= 1
        if numberOfTrainPhotos <= 0:
            gather = not gather
            numberOfTrainPhotos = 300
            displayPhotoIndex = 0

    # Start gathering photos
    k = cv2.waitKey(10)
    if k == ord('b'):
        gather = not gather
        numberOfTrainPhotos = 300
        displayPhotoIndex = 0

    # Change the category of photo
    elif k == ord('e'):
        index = (index + 1) % 4
        photoPath = os.path.join(path, CATEGORIES[index])
        print(photoPath)

    # Close the program
    elif k == ord('q'):
        break

    # Save training data as X.pickle and y.pickle
    elif k == ord('s'):
        random.shuffle(trainingData)
        X = []
        y = []

        for features, label in trainingData:
            X.append(features)
            y.append(label)

        with open(os.path.join(path, 'X.pickle'), 'wb') as f1:
            pickle.dump(X, f1)
        with open(os.path.join(path, 'y.pickle'), 'wb') as f2:
            pickle.dump(y, f2)

    cv2.imshow('vid', img)
