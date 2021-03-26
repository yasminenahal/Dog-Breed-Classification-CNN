import os
import sys
import cv2
import numpy as np
from scipy import io
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing import image
model_resnet = resnet50.ResNet50()

filepath = sys.argv[1]
model_path = sys.argv[2]

TARGET_X, TARGET_Y = 224, 224

def display_prediction(pred_class):
    for imagenet_id, name, likelihood in pred_class[0]:
        print("Predicted breed : {} with {:2f} likelihood".format(name, likelihood))
        
def run_resnet_model(img_path):
    img = image.load_img(path=img_path, target_size=(224, 224))
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    X = resnet50.preprocess_input(X)
    X_pred = model_resnet.predict(X)
    display_prediction(resnet50.decode_predictions(X_pred, top=1))
    


if not os.path.exists(filepath):
    sys.exit("No such file or directory.")
    
    
else:
    
    # read labels
    train_list = io.loadmat('stanford_dogs_dataset/train_list.mat')['file_list']
    train_labels = []
    for train_sample in train_list:
        train_labels.append(train_sample[0][0].split('-',1)[1].split('/',1)[0])
    
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
    img_to_array = cv2.resize(img, (224, 224))
                
    img_to_array = np.array([img_to_array])
    
    if not os.path.exists(model_path):
        sys.exist("No such model path.")

    model = load_model(model_path, compile = True)
    
    prediction = model.predict(img_to_array)
    pred_encoded = []
    for i in prediction[0]:
        if i<prediction.max():
            pred_encoded.append(0.0)
        if i==prediction.max():
            pred_encoded.append(1.0)
    
    breeds = list(set(train_labels))
    
    le = LabelEncoder()

    le.fit(breeds)
    y = np.array(to_categorical(le.transform(breeds), 120))
    
    breeds_encod = {}
    for i, breed_name in enumerate(breeds):
        breeds_encod[breed_name] = y[i]
    
    for b_name in breeds_encod:
        if breeds_encod[b_name].tolist().index(1.0) == pred_encoded.index(1.0):
            predicted_breed = b_name
    
    print("MODEL FROM SCRATCH\n**********************")
    print("Predicted breed : {} with {:2f} likehood\n".format(predicted_breed, prediction[0].max()))
    
    print("MODEL RESNET50\n**********************")
    run_resnet_model(filepath)