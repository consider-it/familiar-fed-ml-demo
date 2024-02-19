import sys, os, time
import numpy as np
import cv2 as cv
import mediapipe as mp
import traceback
import pandas as pd
import math
import traceback
import tensorflow as tf
import numpy as np
import os
import math

#region NOT NEEDED
# import pyzed.sl as sl
# from patchify import patchify, unpatchify
# from scipy.spatial import distance
# import pickle as pkl
# import socket
# import asyncio
# import json
# import threading
# from datetime import datetime
# import platform
# import datetime
# import tensorflow_datasets as tfds
# from tensorflow.python import keras
# import matplotlib.pyplot as plt
# import random
#endregion

print(os.getcwd())

class mlModel():

    DEFAULT_INPUT_H = 150
    DEFAULT_INPUT_W = 150
    DEFAULT_INPUT_D = 3
    DEFAULT_N_CLASSES = 3
    DEFAULT_CLASS_DICT = {0:'rock', 1:'paper', 2:'scissor'}

    def __init__(self,
                input_h: int = DEFAULT_INPUT_H,
                input_w: int = DEFAULT_INPUT_W,
                input_d: int = DEFAULT_INPUT_D,
                n_classes: int = DEFAULT_N_CLASSES,
                class_dict: dict = DEFAULT_CLASS_DICT
                ):
        
        self.input_h = input_h
        self.input_w = input_w
        self.input_d = input_d
        self.n_classes = n_classes
        self.class_dict = class_dict

        self.prediction = None

        self.initialize_model()
    
    def initialize_model(self):
        INPUT_IMG_SHAPE = (self.input_h, self.input_w, self.input_d)

        self.model = tf.keras.models.Sequential()

        # First convolution.
        self.model.add(tf.keras.layers.Convolution2D(
            input_shape=INPUT_IMG_SHAPE,
            filters=64,
            kernel_size=3,
            activation=tf.keras.activations.relu
        ))
        self.model.add(tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2)
        ))

        # Second convolution.
        self.model.add(tf.keras.layers.Convolution2D(
            filters=64,
            kernel_size=3,
            activation=tf.keras.activations.relu
        ))
        self.model.add(tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2)
        ))

        # Third convolution.
        self.model.add(tf.keras.layers.Convolution2D(
            filters=128,
            kernel_size=3,
            activation=tf.keras.activations.relu
        ))
        self.model.add(tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2)
        ))

        # Fourth convolution.
        self.model.add(tf.keras.layers.Convolution2D(
            filters=128,
            kernel_size=3,
            activation=tf.keras.activations.relu
        ))
        self.model.add(tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2)
        ))

        # Flatten the results to feed into dense layers.
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dropout(0.5))

        # 512 neuron dense layer.
        self.model.add(tf.keras.layers.Dense(
            units=512,
            activation=tf.keras.activations.relu
        ))

        # Output layer.
        self.model.add(tf.keras.layers.Dense(
            units=self.n_classes,
            activation=tf.keras.activations.softmax
        ))

        # adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        rmsprop_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

        self.model.compile(
            optimizer=rmsprop_optimizer,
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=['accuracy']
        )

        #self.model.load_weights(os.getcwd()+'/UC MVD-Rock Paper Scissor/rock_paper_scissors_cnn/weights.hdf5')

    def predict(self, image):
        image_np = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
        prediction_prob = self.model.predict(image_np, verbose=0)
        self.prediction = self.class_dict[np.argmax(prediction_prob)]

class handDetector():
    """
    A class to detect hands using a specified model.

    Attributes:
        mode (bool): The mode in which the detector operates.
        max_hands (int): Maximum number of hands to detect.
        detection_con (float): Confidence threshold for detection.
        track_con (float): Confidence threshold for tracking.
        model_complex (int): Complexity of the detection model.
        draw_landmarks (bool): Whether to draw landmarks on detected hands.
    """

    DEFAULT_MODE = False
    DEFAULT_MAX_HANDS = 2
    DEFAULT_DETECTION_CON = 0.5
    DEFAULT_TRACK_CON = 0.5
    DEFAULT_MODEL_COMPLEXITY = 1
    DEFAULT_DRAW_LANDMARKS = True
    DEFAULT_VIDEO_FEED = 1
    DEFAULT_PARAMS = {
            "hand_detection": True,
            "resize_factor": 1,
            }

    def __init__(self, 
                mode: bool = DEFAULT_MODE, 
                maxHands: int = DEFAULT_MAX_HANDS, 
                detectionCon: float = DEFAULT_DETECTION_CON, 
                trackCon: float = DEFAULT_TRACK_CON, 
                modelComplexity: float = DEFAULT_MODEL_COMPLEXITY, 
                draw_landmarks: bool = DEFAULT_DRAW_LANDMARKS,
                video_feed: int = DEFAULT_VIDEO_FEED,
                parameters: dict = DEFAULT_PARAMS
                ):
        
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelComplex = modelComplexity
        self.draw_landmarks = draw_landmarks
        self.parameters = parameters

        self.mpDraw = mp.solutions.drawing_utils
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        
        self.cap = cv.VideoCapture(video_feed)

    def text_with_background(self, font, font_scale, font_thickness, pos, text, img):
        text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv.rectangle(img, (pos[0] - 5, pos[1] - 5), (pos[0] + text_w + 5, pos[1] + text_h + 5), (0,0,0), -1)
        cv.putText(img, text, (pos[0], pos[1] + text_h + font_scale - 1), font, font_scale, (255, 255, 255), font_thickness)
        return img

    def resize_image(self, img, resize_factor, smaller=True):
        if smaller:
            img = cv.resize(img,(img.shape[1]//resize_factor,img.shape[0]//resize_factor))
        else:
            img = cv.resize(img,(img.shape[1]*resize_factor,img.shape[0]*resize_factor))
        return img

    def getImage(self):
        ret, self.original_image = self.cap.read()
        return ret
    
    def findHands(self, img):

        """
        This function detects hands.
        """
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.draw_landmarks:
            if self.results.multi_hand_landmarks:
                for handLms in self.results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo = 0, draw = False):
        """
        This function calculates position coordinates for each hand.
        """
        lmlist = []
        
        myHand = self.results.multi_hand_landmarks[handNo]

        for id, lm in enumerate(myHand.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmlist.append({"id":id, "cx":cx, "cy":cy})

            if self.draw_landmarks:
                cv.circle(img, (cx, cy), 3, (255, 0, 255), cv.FILLED)
        
        if lmlist!=[]:
            return True, pd.DataFrame(lmlist)
        else:
            return False, None

    def draw_roi_and_predict(self, min_x=0, max_x=0, min_y=0, max_y=0, center_x=0, center_y=0, square=True, offset=False):
        """
        This function extracts the detected region of interest and performs prediction.
        """

        h, w = self.display_image.shape[0], self.display_image.shape[1]

        if math.isnan(center_x) or math.isnan(center_y):
            return self.display_image
        
        if square:
            l = int(max([max_y-min_y, max_x-min_x])*1.2 / 2)
            min_x = int(max(0,center_x-l))
            max_x = int(min(center_x+l, w))
            min_y = int(max(0, center_y-l))
            max_y = int(min(center_y+l, h))
        elif offset:
            min_x = int(min_x/1.1)
            max_x = int(max_x*1.1)
            min_y = int(min_y/1.1)
            max_y = int(max_y*1.1)

        cv.rectangle(self.display_image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)

        if self.draw_landmarks:
            cv.circle(self.display_image, (center_x, center_y), 6, (255, 0, 255), cv.FILLED)

        try:
            self.image_to_predict = self.original_image.copy()
            cropped_hand = self.image_to_predict[ min_y: max_y, min_x: max_x, :]
            cropped_hand = cv.resize(cropped_hand,(prediction_model.input_h, prediction_model.input_w))
            cv.imshow("detected_hand", cropped_hand)

            prediction_model.predict(cropped_hand)
        except:
            pass
    
    def detect_hands(self):
        """
        This function detects hand landmarks for multiple hands if defined in hand detector class.
        """

        self.display_image = self.original_image.copy()

        try:
            self.display_image = self.findHands(self.display_image)

            if self.results.multi_hand_landmarks:
                for i in range(0, len(self.results.multi_hand_landmarks)):
                    ret, positions = self.findPosition(img=self.display_image, handNo=i)
                    if ret:
                        # positions['distances'] = positions.apply(lambda row: depth_map[row.cy,row.cx], axis=1)
                        # positions['distances'] = positions['distances'].ffill().bfill()

                        hmin_x, hmax_x = int(positions['cx'].min()), int(positions['cx'].max())
                        hmin_y, hmax_y = int(positions['cy'].min()), int(positions['cy'].max())

                        center_x = int((hmin_x+hmax_x)/2)
                        center_y = int((hmin_y+hmax_y)/2)
        
                        self.draw_roi_and_predict(hmin_x, hmax_x, hmin_y, hmax_y, center_x, center_y)
            else:
                prediction_model.prediction = "No Hands"

        except Exception as e:
            print(time.time(), traceback.format_exc())

if __name__ == '__main__':  
    font = cv.FONT_HERSHEY_PLAIN
    font_scale = 2
    font_thickness = 1

    hand_detector = handDetector(maxHands=2)
    prediction_model = mlModel()

    print(">>>>>> PRESS Q TO QUIT \n")

    while(True):
        try:
            start_time = time.time()

            ret = hand_detector.getImage()
            
            if ret:
                if hand_detector.parameters['hand_detection']:
                    hand_detector.detect_hands()

                    FPS = round(1/(time.time()-start_time))

                    hand_detector.text_with_background(
                        font=font, 
                        font_scale=font_scale, 
                        font_thickness=2, 
                        pos=(50, 50), 
                        text=f"FPS: {FPS} | Prediction: {prediction_model.prediction}", 
                        img=hand_detector.display_image)

                cv.imshow("prediction", hand_detector.resize_image(hand_detector.display_image, hand_detector.parameters['resize_factor'], smaller=True))
            else:
                FPS = round(1/(time.time()-start_time))
                hand_detector.text_with_background(
                    font=font, 
                    font_scale=font_scale, 
                    font_thickness=2, 
                    pos=(50, 50), 
                    text=f"FPS: {FPS}", 
                    img=hand_detector.display_image)

        except Exception as e:
            print(traceback.print_exc())

        key = cv.waitKey(1) 
        if key == ord('q'):
            break

    cv.destroyAllWindows()