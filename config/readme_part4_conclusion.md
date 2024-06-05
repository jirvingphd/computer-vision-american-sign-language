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


