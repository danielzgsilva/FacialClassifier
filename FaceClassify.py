import cv2
import numpy as np
import torch
import os
import matplotlib.image as img
import random
import torch
import torch.nn as nn
from model import *

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Currently in testing
class VideoCamera(object):
    def __init__(self):
      
        self.video = cv2.VideoCapture(0)
       
    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()
    
    def get_frame(self):

        success, image = self.video.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        i = 1
        face_crop = []
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(image, 'Face ' + str(i), (x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
            face_crop.append(image[y:y+h, x:x+w])
            i += 1

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

class FaceClassification(object):
    
    def process_image(file, image_target, face_target):
        filename = str(np.random.randint(0,1000000)) + file.filename
        image_path = image_target + filename
        img = np.array(cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)).astype(np.uint8)

        if img.shape[1] > 400:
            x = 400.0 / img.shape[0]
            new_dim = (int(img.shape[1] * x), 400)
            img = cv2.resize(img, new_dim, interpolation = cv2.INTER_LANCZOS4)

     #   if img.shape[0] < 400:
      #      y = 400.0 / img.shape[1]
       #     new_dim = (400, int(img.shape[0] * y))
        #    img = cv2.resize(img, new_dim, interpolation = cv2.INTER_LANCZOS4)

        highlighted_faces = np.copy(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detections = faceCascade.detectMultiScale(gray, 1.3, 5)
        
        num_faces = len(detections)

        i = 1
        face_names = []
        for (x, y, w, h) in detections:
            cv2.rectangle(highlighted_faces, (x, y), (x + w, y + h), (150, 110, 0), 2)
            cv2.rectangle(highlighted_faces, (x, y), (x + 15, y + 15), (150, 110, 0), -2)
            cv2.putText(highlighted_faces, str(i), (x + 4, y + 12), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)

            temp = img[y:y+h, x:x+w]
            z = 150.0 / temp.shape[0]
            face_dim = (int(temp.shape[1] * z), 150)
            face = cv2.resize(temp, face_dim, interpolation = cv2.INTER_LANCZOS4)

            face_file_name = 'face' + str(i) + '_' + filename
            face_names.append(face_file_name)
            cv2.imwrite(face_target + face_file_name, face)
            i += 1

        cv2.imwrite(image_path, highlighted_faces)

        return filename, face_names, num_faces

    def generate_home():
        file1 = 'Me.jpg'
        face_files1 = ['me_face.jpg']
        text1 = [['Young Adult', 'Male', 'White']]
        confs1 = [[76.2, 95.8, 88.6]]
     
        file2 = '4_kids.jpg'
        face_files2 = ['face1_4_kids.jpg', 'face2_4_kids.jpg', 'face3_4_kids.jpg', 'face4_4_kids.jpg']
        text2 = [['Young Adult', 'Male', 'Indian'], ['Young Adult', 'Male', 'Indian'], ['Young Adult', 'Male', 'Black'], ['Teenager', 'Male', 'Indian']]
        confs2 = [[60.9, 99.8, 64.4], [53.0, 99.8, 84.2], [84.2, 99.9, 53.0], [59.7, 99.9, 61.2]]

        file3 = 'family.jpg'
        face_files3 = ['face1_family.jpg', 'face2_family.jpg', 'face3_family.jpg', 'face4_family.jpg', 'face5_family.jpg']  
        text3 = [['Older Adult', 'Male', 'White'], ['Young Adult', 'Female', 'White'], ['Older Adult', 'Female', 'White'], ['Middle Aged', 'Male', 'White'], ['Young Adult', 'Male', 'White']]
        confs3 = [[77.6, 97.1, 90.3], [42.1, 85.9, 98.6], [77.4, 99.9, 92.9], [66.0, 99.9, 88.2], [48.7, 99.7, 99.1]]
        
        home_files = [file1, file2, file3]
        face_files = [face_files1, face_files2, face_files3]
        text = [text1, text2, text3]
        confidences = [confs1, confs2, confs3]

        return home_files, face_files, text, confidences

    def get_preds(age_preds, gender_preds, race_preds):
        
        ages = ['Baby', 'Child', 'Middle Aged', 'Older Adult', 'Senior Citizen', 'Teenager', 'Young Adult']
        genders = ['Female', 'Male']
        races = ['Asian', 'Black', 'Indian', 'Hispanic', 'White']

        age = ages[age_preds[0]]
        gender = genders[gender_preds[0]] 
        race = races[race_preds[0]]

        return age, gender, race

    def classify_image(model_path, face_path, file_names):
        random.seed(42)
        face_model = PreTrained_Senet()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        face_model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth'), map_location = device))
        face_model.to(device)

        predictions = ['']
        for file in file_names:
            file_path = face_path + file
            image = img.imread(file_path)
            image = np.array(image, np.float32)
            image = cv2.resize(image, (224,224), interpolation = cv2.INTER_CUBIC)
            image = image / 255
            image = np.clip(image, 0, 1)

            image_tensor = image.transpose((1, 2, 0))
            image_tensor = image_tensor.transpose((1, 2, 0))

            image_tensor = np.expand_dims(image_tensor, axis=0)
            image_tensor = torch.tensor(image_tensor)

            with torch.no_grad():
                image_tensor.to(device)
                face_model.eval()

                age, gender, race = face_model(image_tensor)

                # Gets the predictions of the inputs 
                _, age_preds = torch.max(age, 1)
                _, gender_preds = torch.max(gender, 1)
                _, race_preds = torch.max(race, 1)
                
                age_conf = str(np.round((torch.exp(age) * 100).numpy(), 1)[0][age_preds[0]])
                gender_conf = str(np.round((torch.exp(gender) * 100).numpy(), 1)[0][gender_preds[0]])
                race_conf = str(np.round((torch.exp(race) * 100).numpy(), 1)[0][race_preds[0]])

            age, gender, race = FaceClassification.get_preds(age_preds, gender_preds, race_preds)

            age = [age, age_conf]
            gender = [gender, gender_conf]
            race = [race, race_conf]
            
            prediction = [age, gender, race]
            predictions.append(prediction)

        return predictions