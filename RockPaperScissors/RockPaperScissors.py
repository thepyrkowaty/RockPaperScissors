import cv2
import os
import random
from tensorflow.keras import models
import numpy as np
CATEGORIES = ['Rock', 'Paper', 'Scissors', 'None']
path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'Data')
index = 0
cap = cv2.VideoCapture(0)
model = models.load_model(os.path.join(path, 'model'))
startDisplaying = True
computerWon, playerWon, draw = False, False, False
shapeToDisplay, computerShapeToDisplay = '', ''


def preparePhoto(myImg):
    IMG_SIZE = 100
    myImg = cv2.cvtColor(myImg, cv2.COLOR_BGR2RGB)
    myImg = cv2.resize(myImg, (IMG_SIZE, IMG_SIZE))
    myImg = myImg/255.0
    return myImg.reshape((1, IMG_SIZE, IMG_SIZE, 3))


def predictPhotoModel(myImg):
    pred = model.predict(myImg)
    return f'Your choice looks like: {CATEGORIES[int(np.argmax(pred))]}', CATEGORIES[int(np.argmax(pred))]


def drawArea(img, name):
    cv2.rectangle(img, (300, 20), (700, 420), (0, 255, 255), thickness=2)
    cv2.rectangle(img, (5, 30), (220, 180), (0, 0, 102), thickness=-1)


def drawMenu(img):
    cv2.putText(img, 'Press q to quit', (10, 590), cv2.FONT_ITALIC, 0.65, (0, 0, 0), thickness=2)
    cv2.putText(img, 'Press c to confirm your choice', (10, 560), cv2.FONT_ITALIC, 0.65, (0, 0, 0), thickness=2)
    cv2.putText(img, "Computer's choice: ", (10, 50), cv2.FONT_ITALIC, 0.65, (0, 0, 0), thickness=2)
    cv2.putText(img, "Your's choice: ", (10, 110), cv2.FONT_ITALIC, 0.65, (0, 0, 0), thickness=2)
    # To make a black outline
    cv2.putText(img, 'Press q to quit', (10, 590), cv2.FONT_ITALIC, 0.65, (255, 255, 255), thickness=1)
    cv2.putText(img, 'Press c to confirm your choice', (10, 560), cv2.FONT_ITALIC, 0.65, (255, 255, 255), thickness=1)
    cv2.putText(img, "Computer's choice: ", (10, 50), cv2.FONT_ITALIC, 0.65, (255, 255, 255), thickness=1)
    cv2.putText(img, "Your's choice: ", (10, 110), cv2.FONT_ITALIC, 0.65, (255, 255, 255), thickness=1)


def drawChoices(shape, computerShape):
    cv2.putText(img, f"{computerShape}", (10, 80), cv2.FONT_ITALIC, 0.65, (0, 0, 0), thickness=2)
    cv2.putText(img, f"{shape}", (10, 140), cv2.FONT_ITALIC, 0.65, (0, 0, 0), thickness=2)
    cv2.putText(img, f"{computerShape}", (10, 80), cv2.FONT_ITALIC, 0.65, (255, 255, 255), thickness=1)
    cv2.putText(img, f"{shape}", (10, 140), cv2.FONT_ITALIC, 0.65, (255, 255, 255), thickness=1)


while True:
    success, img = cap.read()
    img = cv2.resize(img, (800, 600))
    handImg = img[22:418, 302:698]
    drawArea(img, CATEGORIES[index])
    drawMenu(img)
    predictionPhoto = preparePhoto(handImg)
    prediction, shape = predictPhotoModel(predictionPhoto)
    cv2.putText(img, prediction, (300, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)
    cv2.putText(img, prediction, (300, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), thickness=1)

    if startDisplaying:
        shapeToDisplay = 'Waiting!'
        computerShapeToDisplay = 'Waiting!'
        startDisplaying = False

    k = cv2.waitKey(10)
    if k == ord('e'):
        index = (index + 1) % 4

    elif k == ord('q'):
        break

    if k == ord('c'):
        if shape != 'None':
            computerShape = random.choice(CATEGORIES[:-1])
            if (computerShape == 'Rock' and shape == 'Scissors') or \
                    (computerShape == 'Scissors' and shape == 'Paper') or \
                    (computerShape == 'Paper' and shape == 'Rock'):
                playerWon = False
                computerWon = True
            elif (computerShape == 'Scissors' and shape == 'Rock') or \
                    (computerShape == 'Paper' and shape == 'Scissors') or \
                    (computerShape == 'Rock' and shape == 'Paper'):
                computerWon = False
                playerWon = True
            else:
                computerWon = False
                playerWon = False
                draw = True
            shapeToDisplay = shape
            computerShapeToDisplay = computerShape

    drawChoices(shapeToDisplay, computerShapeToDisplay)
    if computerWon:
        cv2.putText(img, "Computer Won!", (10, 170), cv2.FONT_ITALIC, 0.65, (0, 0, 0), thickness=2)
        cv2.putText(img, "Computer Won!", (10, 170), cv2.FONT_ITALIC, 0.65, (255, 255, 255), thickness=1)
    elif playerWon:
        cv2.putText(img, "You Won!", (10, 170), cv2.FONT_ITALIC, 0.65, (0, 0, 0), thickness=2)
        cv2.putText(img, "You Won!", (10, 170), cv2.FONT_ITALIC, 0.65, (255, 255, 255), thickness=1)
    elif draw:
        cv2.putText(img, "Draw!", (10, 170), cv2.FONT_ITALIC, 0.65, (0, 0, 0), thickness=2)
        cv2.putText(img, "Draw!", (10, 170), cv2.FONT_ITALIC, 0.65, (255, 255, 255), thickness=1)

    cv2.imshow('vid', img)
