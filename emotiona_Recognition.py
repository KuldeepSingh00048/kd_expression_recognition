import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import os
import numpy as np

Datasets_directory = "C:/Users/kuldeep Singh/Desktop/New folder/face/train/"
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
training_data = []

img_size = 224


# Creating functions to read all images of datasets and store in 1d arrays

def create_td():
    for i in emotions:
        path = os.path.join(Datasets_directory, i)
        class_num = emotions.index(i)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (img_size, img_size))
                # print(new_array.shape)
                training_data.append([new_array, class_num])
            except Exception as E:
                pass


create_td()
print(len(training_data))
temp = np.array(training_data)
print(temp.shape)

x1 = []
y1 = []

for features, label in training_data:
    x1.append(features)
    y1.append(label)

x = np.array(x1).reshape(-1, img_size, img_size, 3)  # converting the image in 4 dimension
print(x.shape)

# normalizing the data
x = x / 255.0

y = np.array(y1)

# ----------------------------------Training model part start------------------------------------------------------

model = tf.keras.applications.MobileNetV2()  # Pre Trained Models
print(model.summary())

base_input = model.layers[0].input
base_output = model.layers[-2].output
print(base_output)

final_output = layers.Dense(128)(base_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(64)(final_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(7, activation='softmax')(final_output)

# print(final_output)

new_model = keras.Model(inputs=base_input, outputs=final_output)
# print(new_model.summary()
new_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# new_model.fit(x, y, epochs=28)
new_model.save('my_model.h5')
new_model = tf.keras.models.load_model('my_model.h5')
# --------------------------------Training model part end-----------------------------------------------------------


# -----------------------------------Live Webcam Part---------------------------------------------------------------
cap = cv2.VideoCapture(0)
# img = frame.shape
# print(img)
# plt.show(block=True)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
while True:
    kd, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = faceCascade.detectMultiScale(gray, 1.1, 4)

    for x, y, w, h in face:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        faces = faceCascade.detectMultiScale(roi_gray)
        if len(faces) == 0:
            pass
        else:
            for (ex, ey, ew, eh) in faces:
                face_roi = roi_color[ey: ey + eh, ex:ex + ew]

                final = cv2.resize(face_roi, (224, 224))
                final = np.expand_dims(final, axis=0)
                final = final / 255.0
                prediction = new_model.predict(final)
                # print(prediction[0])
                percent = (100 * (0.95 - (prediction[0])))
                # print(np.argmax(prediction))
                print(str(emotions[np.argmax(prediction)]) + "--" + str(percent[np.argmax(prediction)]))
                text = str(emotions[np.argmax(prediction)]) + "--" + str(percent[np.argmax(prediction)])
                cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("face", frame)
    key = cv2.waitKey(1)
    if key == 13:  # press 'Enter' to quit
        break

cap.release()
cv2.destroyAllWindows()

# ---------------------------------------------------------------------------------------------------

# import matplotlib.pyplot as plt

# img_array = cv2.imread("C:/Users/kuldeep Singh/PycharmProjects/face_project/test/angry/PrivateTest_10131363.jpg")
# print(img_array.shape)
# pit.imshow(img_array)


# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# plt.show(block=True)

'''for catagory in classes:
    path = os.path.join(datadirectory,catagory)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        img_size = 224
        new_array = cv2.resize(img_array, (img_size, img_size))
        print(new_array.shape)
        pit.imshow(cv2.cvtColor(new_array,cv2.COLOR_BGR2RGB))
        pit.show()
        break
    break
'''
