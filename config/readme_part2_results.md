
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

