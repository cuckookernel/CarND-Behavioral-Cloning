# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/center_lane_driving.jpg "Center Lane Driving"
[image3]: ./examples/recover_left.jpg "Recover from left"
[image4]: ./examples/recover_right.jpg "Recovery from right"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* **`model.py`**: containing the script to create and train the model
* **`drive.py`**: for driving the car in autonomous mode. I made no changes to this file.
* **`model.h5`**: containing a trained convolution neural network 
* **`writeup_report.md`**: you are reading it

Besides this file, the video of a whole lap driven by the trained network is available here: https://youtu.be/pNxtBFdwN7k


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. 
Function `search_correction_params` shows how I tried with different combinations of parameters `side_img_correct_mode` and `side_img_correct_param` 
which basically control how the angle is corrected for left/right images taken during training. 

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model uses essentially the same architecture shown in section 15 (Even More powerful network) of the proyect chapter.
The model is constructed in function `build_model` (`model.py lines` 152-176). 
It includes 3 convolutional layers with 5x5 kernels followed by 2 convolutional layers with 3x3 kernels, followed by 3 fully connected layers. 
The main differences with respect to the model shown in the video are: 
    - I included several dropout layers between the Conv layers, with dropout parameter `dropout_cs`. 
    - There are also a couple dropout layers between fully connected layer.
    - The final layer has output of dimension 1, as we only want to predict the steering angle. I used "tanh" as the activation function of this layer to guarantee that the output was in the interval [-1,1]
    - The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

As mentioned above, the model contains dropout layers in order to reduce overfitting (function `build_model`). 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track for a whole lap.

#### 3. Model parameter tuning

The training procedure used an Adam optimizer. However, I did some tries with the default value 0.001 for the learning rate, used by keras but later lowered it to 0.0005, 
as it seem to produced a better validation loss.
The learning rate is set as part of the `cfg` parameter dict (`model.py` line 55) 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I mostly used center lane driving, ocasionally recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to take advantage of the experience gained by NVIDIA engineers about how to design 
a neural network for driving a car and just copying their architecture. 

From the beginning I knew that overfitting could be an issue. So included significant dropout on my layers, focusing on the ones with the most parameters.
As a comparison between the training and validation losses for up to 20 epochs showed, this proved rather effective in eliminating overfitting almost completely. 


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where there was visible swerving, indicating 
some instability in the steering angle produced by the model . 
 the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture 

![Architecture visualization][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center lane driving][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![Recover from left][image3]
![Recover from right][image4]
![alt text][image5]

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
