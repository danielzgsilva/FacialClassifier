# FacialClassification

#### Computer Vision web app which predicts the age, gender, and race of all individuals present in an uploaded image
-  Achieves an average 85% accuracy across all three dependent variables 
-  Back-end development done in Flask 
-  Front-end styling accomplished using Bootstrap 4 

The underlying Convolutional Neural Network uses a pretrained Squeeze and Excitation Network (SENet), trained on VGGFace2
-  Cao, Qiong, et al. "Vggface2: A dataset for recognising faces across pose and age." Automatic Face & Gesture Recognition (FG 2018), 2018 13th IEEE International Conference on. IEEE, 2018.

Fine tuning done on the UTKFace dataset
- https://susanqq.github.io/UTKFace/
