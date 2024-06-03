
## Results

|   Rank | Name                             |   Precision |   Recall |   F1-Score |   Accuracy | Fit Time       | Model Save Fpath                                 |
|-------:|:---------------------------------|------------:|---------:|-----------:|-----------:|:---------------|:-------------------------------------------------|
|      1 | EfficientNetB0-1                 |       0.772 |    0.769 |      0.754 |      0.763 | 0:02:04.131142 | modeling/models/EfficientNetB0-1                 |
|      2 | VGG16-01                         |       0.539 |    0.534 |      0.518 |      0.535 | 0:01:55.297751 | modeling/models/VGG16-01                         |
|      3 | cnn1-fixed-lr                    |       0.355 |    0.359 |      0.347 |      0.368 | 0:00:21.583943 | modeling/models/cnn1-fixed-lr                    |
|      3 | cnn1-scheduled-lr                |       0.355 |    0.359 |      0.346 |      0.368 | 0:00:22.408392 | modeling/models/cnn1-scheduled-lr                |
|      5 | towards-data-science             |       0.386 |    0.318 |      0.316 |      0.342 | 0:01:43.888912 | modeling/models/towards-data-science             |
|      6 | towards-data-science_lr-schedule |       0.312 |    0.288 |      0.271 |      0.294 | 0:04:23.701543 | modeling/models/towards-data-science_lr-schedule |

### Best Model

- EfficientNetB0-1


#### Test Results


<img src="modeling/models/EfficientNetB0-1/history.png">

```
--------------------------------------------------------------------------------
 Classification Metrics: Test Data 
--------------------------------------------------------------------------------
              precision    recall  f1-score   support

           A       1.00      1.00      1.00        13
           B       0.75      0.60      0.67         5
           C       1.00      0.70      0.82        10
           D       0.67      0.67      0.67         9
           E       0.89      0.89      0.89         9
           F       0.75      0.43      0.55         7
           G       0.77      1.00      0.87        10
           H       0.86      0.60      0.71        10
           I       0.69      0.53      0.60        17
           J       0.67      1.00      0.80         6
           K       1.00      1.00      1.00         5
           L       0.82      0.82      0.82        11
           M       0.40      0.40      0.40         5
           N       0.78      0.58      0.67        12
           O       0.70      0.88      0.78         8
           P       1.00      0.45      0.62        11
           Q       0.78      0.78      0.78         9
           R       0.50      0.83      0.62         6
           S       0.87      1.00      0.93        13
           T       1.00      0.71      0.83         7
           U       0.67      0.67      0.67         6
           V       0.64      0.82      0.72        11
           W       0.60      0.86      0.71         7
           X       0.54      0.78      0.64         9
           Y       0.75      1.00      0.86         3
           Z       1.00      1.00      1.00         9

    accuracy                           0.76       228
   macro avg       0.77      0.77      0.75       228
weighted avg       0.79      0.76      0.76       228

--------------------------------------------------------------------------------
```


<img src="modeling/models/EfficientNetB0-1/confusion_matrix.png">


#### Example Explanationsfor Model Predictions (Comging Soon!)
> Placeholder for example explanation of correctly classified image
> Placeholder for example explanation of incorrectly classified image
> Placeholder for example explanation of the 2 clases most often confused for each other.
