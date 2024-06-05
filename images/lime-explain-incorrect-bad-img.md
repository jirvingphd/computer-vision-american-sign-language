#### LIME Explanation for an Incorrectly Classified Image (Image Quality)

<img src="images/lime-explain-incorrect-bad-img.png">


Here we can see that the model was able was confused by the image itself.
It seems to have mistaken the wood flooring for human fingers, thus preventing it from identifying the letter correctly.
Images like this should either be removed from the dataset or, ideally, augmented with an appropriate amount of data augmentaion.
If this task was converted to include object recognition, the model may have been able to classify it correctly.

This also underscores the need for quality, high-resolution training data for the best results.
