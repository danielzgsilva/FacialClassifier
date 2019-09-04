# Facial Classifier

**This is a computer vision web application, built to predict the age, race, and gender of all individuals present in an image**
-  Achieves an average 85% accuracy across all three dependent variables 
-  Back-end development done in Flask 
-  Front-end styling accomplished using Bootstrap 4 
-  CNN model trained in PyTorch

<p align="center">Testing it out on myself!
  <img src='static/home_images/Readme.JPG' alt='An example of the classifier functioning on me!' />
</p>

## Running the website locally
This facial classifier runs on Python 3.7

Start off by cloning the repo:  
`git clone https://github.com/danielzgsilva`

Navigate to the project's root and install dependencies like so:
`pip install -r requirements.txt`

Run the app with:  
`python app.py`

The project will then serve locally on port 5555:  
`http://localhost:5555/`

## Training details
The underlying Convolutional Neural Network uses a pretrained Squeeze and Excitation Network (SENet), trained on VGGFace2
-  Cao, Qiong, et al. "Vggface2: A dataset for recognising faces across pose and age." Automatic Face & Gesture Recognition (FG 2018), 2018 13th IEEE International Conference on. IEEE, 2018.

The model was fine tuned for classification of tightly cropped facial images using the UTKFace dataset
- https://susanqq.github.io/UTKFace/

## Work in progress:
- Deploying the app through Heroku
- Combatting learned biases due to imbalanced datasets
- Adding the ability to take a picture through the app, rather than requiring an image upload
- Increasing performance in poor lighting
