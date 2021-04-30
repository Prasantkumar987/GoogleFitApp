# Google Fit App using **Streamlit**

**This repo contains the project which predicts how much calories you burn.**

**Web App link**
![Google Fit web app](https://share.streamlit.io/prasantkumar987/googlefitapp/main)

### Preview

![fit](https://user-images.githubusercontent.com/63397654/116190887-0861a400-a749-11eb-8b9b-9517367c2291.gif)

## 1. Problem Statement: 
A few months back I started using **Google Fit app.** This App counts your daily walking distnce-steps, running distance-steps, cycling distance-steps.
Your **Heart points**, **calories** and many more things. So I downloaded my google fit dataset from **takeout.google.com** website.

I wanted to predict how much **calories** we will burn if we give a walking, running and cycling distance.
I used **Artificial Neural Network** to train my model.

## 2. Model Training:
To train my model I downloaded additional dataset from kaggle.com. After that I dropped unnecessary columns. 
Using **Simple Imputer** module from scikit-learn library I handled missing values, after that I had to use **Standard Scaler** to scale column values.

**Parameters:**

    a. Number of Units = 60
  
    b. Activation function = Rectified Linear Unit(Relu)

    c. Regularizer parameter lambda = 0.001
    
    d. Optimizer = Adamax
    
    e. Loss = mean_squared_error
    
    f. metrics = mean absoulte error, mean absoulte percentage error.
    
    g. batch size = 1
    
    h. No of epochs = 200
    
   
**Loss:**


At epochs 1- 
    
    a. mean squared error = 3183943.2643
    
    b. Mean absolute error = 1759.6287
    
    c. Mean absolute percentage error = 95.6408%
    

At epochs 200-

    a. mean squared error = 457.76
    
    b. Mean absolute error = 14. 70
    
    c. Mean absolute percentage error = 0.79%


**Why did I used regularization?**

On training dataset I got mean absoulte percentage error as 0.79% but when I evaluted my model on testing dataset I got MAPE as 52.46%. So that means my model was having high variance, to reduce the high variance we can perform following steps:

1. Add more training datset
2. Regularization
3. Drop out 
4. Data Augmentation

I had limited number of training datset so I coulden't use that. I tried Drop out but it wasn't that much effective and data aumentation is possible on image or video dataset so I tried **Regularization** and I got mean absolute percentage error as 50.67%


## 3. Model Deployment:

To create a wep app and deploy I used **streamlit** library. 


## 4. Credit:

I would like to give credit to google to develop such an amazing fitness app, to kaggle for providing necessasy dataset. 

And to my friend ![Rahul Kadam](https://www.linkedin.com/in/rahuljkadam/) who help me with this project.














