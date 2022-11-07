#Importing necessary modules
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn


# list of class names by index
with open("classes.txt") as file:
    class_names=[line.rstrip('\n') for line in file]

# loading the model architecture
model = models.resnet50(pretrained=False)
model.fc=nn.Linear(2048,133,bias=True)

#loading the model weights from the trained model
model.load_state_dict(torch.load("model_transfer.pt",map_location ='cpu'))

#Model now in evaluation mode (ready for predicting)
model.eval()


app = FastAPI(title="Pycon Ireland 2022 Demo")



@app.post("/find_sum_of_two_numbers",tags=['Find sum of two numbers'])
def sum_of_two_numbers(number1: int, number2: int):
    return {"sum": number1+number2}


@app.post("/Predict dog breed from image",tags=['Dog Breed Prediction'])
def predict_dog_breed(img: UploadFile = File(...)):
    image = Image.open(img.file)
    predicted_breed=predict_breed(image)
    return {'Predicted Dog Breed': predicted_breed}


def predict_breed(image):
    # takes image as input and return the predicted breed

    # Some transformations to the image before predicting
    pre_process = transforms.Compose([  transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(), #converts to tensor
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                             std =[0.229, 0.224, 0.225])])
    image = pre_process(image)

    #increases dimension by 1 of the tensor 
    image.unsqueeze_(dim=0) 
    
    # Model predicting on the image tensor
    output = model(image)
    
    # Index which has the max probability for the rprediciton 
    max_val,index = torch.max(output,dim=1) 
    
    
    predicted_class_index = index.item()

    #Using the index to find the predicted breed name from the list
    predicted_class_name = class_names[predicted_class_index]
    
    return predicted_class_name