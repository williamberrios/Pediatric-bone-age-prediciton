# What do we need?
# 1. Way to upload  : endpoint
# 2. Way to save the image
# 3. Function to make prediction on the image
import os
import torch
from flask import Flask
from flask import request
from flask import render_template
import numpy as np 
import pandas as pd
import torch 
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn.functional as F
import torchvision
import albumentations as A
import pretrainedmodels
import cv2
from tqdm import tqdm
import torch.nn as nn


app = Flask(__name__)
# Path in which ypu save the images
UPLOAD_FOLDER = '/home/william/Desktop/Learning plan 2020/Programming and Others/I2A2/Bone Age Regression/Web Aplication/static/save_images/'
MODEL = None
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Class for the Bone Dataset:

class BoneAgeDataset_withGender():
    def __init__(self, image_paths,gender,targets, augmentations=None):
        self.image_paths = image_paths
        self.targets = targets
        self.augmentations = augmentations
        self.gender = gender

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        #Calculate image and targets
        image = cv2.imread(self.image_paths[item])
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        targets = self.targets[item]
        
        if self.augmentations is not None: # Use Resize inside aug
            augmented = self.augmentations(image=image)
            image = augmented["image"]
        
        # Return the image in Tensor format (First the Channels) 
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        # Calculate image and gender
        gender = self.gender[item]
        
        return {
            "image": torch.tensor(image, dtype=torch.float), 
            "targets": torch.tensor(targets, dtype=torch.long),
            "gender": torch.tensor(gender, dtype=torch.float)
        }    
    
class Resnet152_withGender(nn.Module):
    
    def __init__(self, pretrained='imagenet'):
        super(Resnet152_withGender, self).__init__()
        
        self.base_model = pretrainedmodels.__dict__["resnet152"](pretrained=pretrained)
        self.l0 = nn.Linear(2048, 512)
        self.l1 = nn.Linear(1,32)
        self.last_layer = nn.Linear(512+32, 1)
        self.drop1 = nn.Dropout(p=0.2)
    
    def forward(self, image, gender):
        batch_size, _, _, _ = image.shape
        
        x1 = self.base_model.features(image)
        x1 = F.adaptive_avg_pool2d(x1, 1).reshape(batch_size, -1)
        x1 = self.l0(x1)
        x2 = self.l1(gender.reshape(batch_size, -1))
        x = torch.cat((x1, x2), dim=1)
        x = self.drop1(x)
        out = self.last_layer(x)
        return out

def predict(data_loader, model):
    model.eval()
    final_predictions = []
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for b_idx, data in enumerate(tk0):
            for key, value in data.items():
                data[key] = value.to(DEVICE)
            predictions = model(data['image'],data['gender'])
            predictions = predictions.cpu()
            final_predictions.append(predictions)
    return final_predictions

def predict_image(image_path,model,test_gender='F'):
    # Parameters from imagenet
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    test_images  = [image_path]
    # Add a button to select if its F or M
    test_gender = [(test_gender=='F')*1]
    test_targets = [0]
    
    # Augmentation
    test_aug = A.Compose(
                    [A.Resize(512,512),
                    A.Normalize(mean,std,max_pixel_value = 255.0, always_apply = True)             
                    ])

    test_dataset = BoneAgeDataset_withGender(test_images, test_gender, test_targets, augmentations = test_aug)

    test_dataloader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size = 1,
                        shuffle = False, # Keep it False with wtfml and evaluate the metric
                        num_workers = 0
                        )

    test_pred = predict(test_dataloader, model)
    return np.vstack((test_pred)).ravel()


@app.route('/',methods = ['GET','POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        gender =request.form['gender']
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            pred = predict_image(image_location,MODEL,gender)[0]
            print(image_file.filename)
            return render_template('index.html', Prediction = pred, image_loc = image_file.filename) 
    return render_template('index.html',Prediction = 0,image_loc = None)

if __name__ == '__main__':
    import torch
    MODEL = Resnet152_withGender(pretrained = 'imagenet' )
    if str(DEVICE) == 'cpu':
        print('Using CPU')
        MODEL.load_state_dict(torch.load('../Models/Bone_age_Regression_Resnet_152_with_gender.pt',map_location = torch.device('cpu')))   
    else:
        print('Using GPU')
        MODEL.load_state_dict(torch.load('../Models/Bone_age_Regression_Resnet_152_with_gender.pt'))   
    
    MODEL.to(DEVICE)
    app.run(port = 12000, debug = True)