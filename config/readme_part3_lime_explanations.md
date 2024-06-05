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
