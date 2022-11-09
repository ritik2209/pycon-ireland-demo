#Importing necessary modules
from fastapi import FastAPI
from sklearn.datasets import load_iris
from pydantic import BaseModel
import pickle

#Importing the iris dataset
iris = load_iris()


#Defining the class which takes all the parametes as input
class request_body(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float


#Defining FastAPI Instance
app = FastAPI(title="Pycon Ireland 2022 Demo")



@app.post("/find_sum_of_two_numbers",tags=['Find sum of two numbers'])
def sum_of_two_numbers(number1: int, number2: int):
    return {"sum": number1+number2}


@app.post("/find_iris_class",tags=['Find iris category'])
def predict_iris_category(data: request_body):

    # Creating list of list of the values to pass as prarameter to the model
    test_data = [[
            data.sepal_length, 
            data.sepal_width, 
            data.petal_length, 
            data.petal_width
    ]]

    #Opening the model binary file
    file = open('iris_model.sav', 'rb')

    #Loading the model into memory for prediction
    iris_model=pickle.load(file)

    #Predicting the class
    class_idx = iris_model.predict(test_data)[0]

    #Returning the predicted class
    return { 'Predicted Class' : iris.target_names[class_idx]}
