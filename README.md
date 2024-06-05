

# computer-vision-american-sign-language

-  Last Updated: 06/05/2024

<center><img src="images/American_Sign_Language_ASL.svg" width=500px>
<p>By Psiĥedelisto - Own work, Public Domain, <a href="https://commons.wikimedia.org/w/index.php?curid=53652991">"https://commons.wikimedia.org/w/index.php?curid=53652991</a><p>
</center>

## Description

This project aims to develop a computer vision system for American Sign Language (ASL) recognition. 


### Goals 
> - **The first/primary goal is to create a model that can classify images of letters from the ASL alphabet (26-class multi-classification)**
- Create a streamlit application that will accept an image and predict which letter it is.

> - **The second, above-and-beyond goal is to use video as the input and add object detection.**

### Features

- ASL letter and word translation
- User-friendly interface
- Support for multiple hand gestures

<center><img src="images/Sign_language_alphabet_(58).png" width=500px style="border:solid black 1px"> 

<p><a href="https://commons.wikimedia.org/wiki/File:Sign_language_alphabet_(58).png">Image Source</a> </p>
<p> Raziakhatun12, CC BY-SA 4.0 <https://creativecommons.org/licenses/by-sa/4.0>, via Wikimedia Commons</p>

</center>




## Data

### Source/Download
- Public Dataset from [Roboflow](https://public.roboflow.com/object-detection/american-sign-language-letters)


To download:
- Navigate to https://public.roboflow.com/object-detection/american-sign-language-letters
- Click `->` for Downloads.
- Select Format =  Multi-Label Classifiction
- Download zip to computer


### Data Details
- 26 letters of the alphabet
- J and Z are gesture-based letters and will likely be difficult to classify using static images.

- Number of Images: 1731
- Size: 416 x  416 
- Channels: 3

Example of Each letter:

<img src="images/ed_example_letters.png">


## EDA



### Class Balance




<img src="images/label-distribution-countplot.png">



### Methods


- Loading Images as a Tensorflow Dataset object.
    - Image size: 128 x 128
    - Batch size: 32   
    - No data augmentation due to nature of sign language.

- Constructing Convolutional Neural Networks in tensorflow.
- Applying transfer learning with pretrained models
- Explain image classifications using  Lime's ImageExplainer.




## Results

Each of the final models included in the analysis are listed in the table below. 

|   Rank | Name                      |   Precision |   Recall |   F1-Score |   Accuracy | Fit Time       | Model Save Fpath                          |
|-------:|:--------------------------|------------:|---------:|-----------:|-----------:|:---------------|:------------------------------------------|
|      1 | EfficientNetB0-1          |       0.747 |    0.733 |      0.715 |      0.732 | 0:01:24.104776 | modeling/models/EfficientNetB0-1          |
|      2 | VGG16-01                  |       0.654 |    0.612 |      0.599 |      0.614 | 0:01:19.377717 | modeling/models/VGG16-01                  |
|      3 | cnn1-fixed-lr             |       0.499 |    0.471 |      0.456 |      0.482 | 0:00:27.681576 | modeling/models/cnn1-fixed-lr             |
|      4 | towards-data-science-blog |       0.488 |    0.444 |      0.433 |      0.452 | 0:01:14.607511 | modeling/models/towards-data-science-blog |
|      5 | cnn1-scheduled-lr         |       0.002 |    0.038 |      0.003 |      0.039 | 0:00:19.401393 | modeling/models/cnn1-scheduled-lr         |

### Best Model

- EfficientNetB0-1


#### Test Results


<img src="modeling/models/EfficientNetB0-1/history.png">

```
--------------------------------------------------------------------------------
 Classification Metrics: Test Data 
--------------------------------------------------------------------------------
              precision    recall  f1-score   support

           A       0.75      0.86      0.80         7
           B       1.00      0.60      0.75         5
           C       0.88      0.88      0.88         8
           D       0.70      0.58      0.64        12
           E       0.50      0.33      0.40         6
           F       0.50      0.43      0.46         7
           G       1.00      0.60      0.75        10
           H       0.56      1.00      0.71         5
           I       0.47      0.78      0.58         9
           J       0.88      0.94      0.91        16
           K       0.83      0.71      0.77         7
           L       0.83      0.67      0.74        15
           M       0.58      0.88      0.70         8
           N       0.88      0.58      0.70        12
           O       0.86      0.60      0.71        10
           P       0.88      0.70      0.78        10
           Q       1.00      0.80      0.89        10
           R       0.50      1.00      0.67         3
           S       0.56      1.00      0.72         9
           T       0.67      0.50      0.57         8
           U       0.75      0.33      0.46         9
           V       0.57      0.50      0.53         8
           W       1.00      0.80      0.89         5
           X       0.75      1.00      0.86         9
           Y       0.75      1.00      0.86         9
           Z       0.79      1.00      0.88        11

    accuracy                           0.73       228
   macro avg       0.75      0.73      0.72       228
weighted avg       0.77      0.73      0.73       228

--------------------------------------------------------------------------------
```


<img src="modeling/models/EfficientNetB0-1/confusion_matrix.png">


### Explaining Model Predictions with LIME

- To better understand how the model was correctly or incorrectly classifying an image, we leveraged LIME's Image Explainer.
- LIME's image explainer creates its own model to explain a model's prediction for a single image at a time. By inspecting the images the model is having difficulty with, we can get insights into how to change our modeling process/architecture.

#### LIME Explanation for Correctly Classified Image

<img src="images/lime-explain-correct.png">

Here we can see that the model was able to identify the letter O using the palm and thumb. The model used the crevice betwen the pinky and ring finges to differeniate between O and F.

The letters O and F in ASL are indeed similar, as both involve the fingers forming a loop. However, they have distinct differences that can be critical for accurate classification.


- Key Similarities:
    - Both letters involve forming a loop with the fingers.
    - Both have a circular shape as part of the sign.
- Key Differences:
    - In the letter O, all fingers and the thumb touch to form the circle, while in the letter F, only the thumb and index finger touch.
    - The remaining fingers in the letter F are extended, whereas in the letter O, they are curved and relaxed.
#### LIME Explanation for an Incorrectly Classified Image (Image Quality)

<img src="images/lime-explain-incorrect-bad-img.png">


Here we can see that the model was able was confused by the image itself.
It seems to have mistaken the wood flooring for human fingers, thus preventing it from identifying the letter correctly.
Images like this should either be removed from the dataset or, ideally, augmented with an appropriate amount of data augmentaion.
If this task was converted to include object recognition, the model may have been able to classify it correctly.

This also underscores the need for quality, high-resolution training data for the best results.

#### LIME Explanation for Incorrect 
Here we can see a D that was mistaken for an I. This is an understandable mistake, as the two letters have strong similarities.

<img src="images/image-explain-incorrect.png"> 

- Key Similarities:
    - Single Extended Finger: Both D and I involve extending a single finger straight up. In D, it’s the index finger, while in I, it’s the pinky finger. This similarity can easily confuse a computer vision model, especially if the finger is not clearly visible due to angle or occlusion.
    - Curled Fingers: The rest of the fingers are curled or touching the thumb in both signs, forming a rounded shape that can look quite similar from certain angles.
     - Hand Orientation: Both letters are presented with the palm facing outward or slightly to the side, making it harder for the model to rely on palm orientation as a distinguishing feature.
- Key Differences:
     - Finger Extended: The primary difference lies in which finger is extended. The index finger for D and the pinky finger for I. The model needs to be trained to focus on which specific finger is extended.
    - Hand Shape: For D, the remaining fingers form a rounded base with the thumb, while for I, the remaining fingers form a fist. This difference in hand shape can be subtle and requires high-resolution images and careful feature extraction to capture effectively.
    - Thumb Position: In D, the thumb is actively touching the curved fingers, forming a distinct circle, while in I, the thumb is less prominent and rests over the curled fingers.

### LIME Explanations - Implications for Modeling

To increase the ease of training of a model to better distinguish between similar letters, we should consider the following:

- Use High-Resolution images to capture the fine details of finger positioning.
    - This could be as simple as increasing the height and width of the input images, or may involve acquiring higher quality training data.
- Apply Data Augmentation techniques to ensure the model sees these letters from various angles and under different lighting conditions.
    - Due to the nature of the ASL alphabet and the expected orientation of the hand, data augmentation should be applied much more carefully than with other tasks where orientation is more variable.
- Feature Extraction: Focus on extracting features related to the number of extended fingers and the specific shape and orientation of the thumb.
    - Investigate the feature maps extracted from each convolution layer to better optimize feature extraction.
    
- More robust training using more examples of letters that are very similar.

### To Dos:
- [x] Apply transfer learning
- [x] Apply LimeExplainer with best model. 
- [ ] Continue tuning the best architecture with the Keras tuner.
- [ ] Assess inclusion of data augmentation on model performance.
- [ ] Save the best model for deployment using model quantization.

- [ ] Deploy a streamlit application for live inference from uploaded images. 


### Future Work

There are many more iterations to test for this task:
- [ ] Does image augmentation help the models? If so, what augmentations can be applied safely for this task?
- [ ] Does adding additional hidden layers on top of EfficientNet improve performance?
- [ ] Do other available transfer learning models outperform efficient net?
- [ ] Does allowing the transfer learning models to train the convolutional base improve performance?


> **The next level of complexity would be to add object detection for hands, followed by sign classification.** The ultimate goal would be to use video input for ASL alphabet letter detection and classification.

## Summary

This project demonstrates proof-of-concept work classifying ASL alphabetical characters. Working with a 26-label classification model is tricky, but pales in comparison to what would be required to interpert whole ASL words.


