
# Udacity Project 3 - Behavioral Cloning

### The goals / steps of this project are the following:

- Use the simulator to collect data of good driving behavior
- Build, a convolution neural network in Keras that predicts steering angles from images
- Train and validate the model with a training and validation set
- Test that the model successfully drives around track one without leaving the road
- Summarize the results with a written report

### My project includes the following files:

- model.py containing the script to create and train the model
- drive.py for driving the car in autonomous mode
- model.h5 containing a trained convolution neural network
- writeup_report.md or writeup_report.pdf summarizing the results

### Model Architecture

I designed my model based on Nvidia's paper(End to End Learning for Self-Driving Cars) suggested in the SDND lectures. The model takes in input images of the size 66x200 as compared to the original images of 160x320. 

It consists of:

- a normalizing layer
- three 5x5 convolutional layers and two 3x3 convolutional layers
- a flatten layer
- three fully connected layers with tanh activation (and dropouts)
- Lastly, a fully connected output layer. 

The model also implements L2 regularization and ELU activations. Also, The model used an adam optimizer, so the learning rate was not tuned manually. I also tried my hand at trying to decrese overfitting by adding dropout layers between CNN layers. 

### Data load and collecting additional data

To train the model for the project, I initially used the dataset provided by Udacity. By reading various forums and articles on medium I realized that most of the students were able to produce more than satisfactory results using this dataset. I initially used only center images but rather accomadated left and right images by adding a correction factor to the images. The original dataset provided by udacity had aroung 8k centre images, which meant I had around 25k images in total. I then went on to neutralize the left side bias in the images by flipping images and measurements. By doing so, I now had around 50k images in my dataset. 

Udacity suggested adding modifications to the data by including “recovery” data while training. This was a little tedious but it gave the model a chance to learn recovery behavior. As the model learns from the data provided to it, a good variety is crucial to produce adequate results.

The captured training data is not directly fed into the CNN as some preprocessing was required such as cropping, resizing, blur, and a change of color space to YUV. Then I introduced some random brightness adjustments, vertical and horizon shifts to sort of keep the model unpredictable in order to avoid overfitting.

### Conclusion

This project made one thing very clear, the data is all that matters. I tried making changes to the model architecture, but this didn’t have near the impact that changing the fundamental makeup of the training data did. Changing the model architecture did help address overfitting to an extent I guess and this allowed the car to learn a more generalized driving behavior. 

But to convince the model that something like a sharp turn could be acceptable behavior required actually having sharp turns represented well in the training data itself. I intend to further work on this project by underrepresenting some of the data belonging to the straight images with negligible curvature angle assigned to it in the dataset. I believe because the test track includes long sections with very slight or no curvature, the data captured from it tended to be heavily partial toward zero turning angles. This created a problem for the CNN model,which then became biased toward driving in a straight line and was easily confused by sharp turns. I could also clean the dataset one image at a time and try to implement a more agressive cropping procedure.

All in all, due to some unforseen circumstances I don't have time to try all this atm as I am already way behind schedule as it is and need to start Advance lane finding soon!

