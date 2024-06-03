### To Dos:
- [x] Apply transfer learning
- [ ] Save best model for deployment `[Fix issue with EfficientNet and current tensorlfow!]`
- [ ] Tune the best architecture with keras tuner.
- [ ] Apply LimeExplainer with best model. 

- [ ] Deploy a streamlit application for live inference from images. 


### Future Work
>There are many more iterations to test for this task:
- [ ] Does image augmentation help the models?
- [ ] Does adding additional hiddden layers on top of EfficientNet improve performance?
- [ ] Does allowing the transfer learning models to train the convolutional base improve performance?


> **The next level of complextiy would be to add object detection for hands, followed by sign classification.**

## Summary
This project demonstrates proof-of-concept work classifying ASL alphabetical characters. Working with a 26-label classification model is tricky, but pales in comparison to what would be required to interpert whole ASL words.


