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
