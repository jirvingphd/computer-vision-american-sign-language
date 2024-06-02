"""Functions from Intermediate Machine Learning Wk3-4 Lessons"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def classification_metrics(y_true, y_pred, label='',
                           output_dict=False, figsize=(8,4),
                           normalize='true', cmap='Blues',
                           colorbar=False, values_format=".2f",
                           target_names = None, return_fig=True,
                           conf_matrix_text_kws= {}, print_report=True,
                           return_str_report=False):
    """
    Compute classification metrics and display confusion matrices.

    Parameters:
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels.
    - label (str): Label for the classification metrics.
    - output_dict (bool): Whether to return the classification report as a dictionary.
    - figsize (tuple): Figure size for the confusion matrix plot.
    - normalize (str): Normalization method for the confusion matrix. Default is 'true'.
    - cmap (str): Colormap for the confusion matrix plot. Default is 'Blues'.
    - colorbar (bool): Whether to display a colorbar in the confusion matrix plot.
    - values_format (str): Format for the values in the confusion matrix. Default is ".2f".
    - target_names (list): List of target names for the classification report.
    - return_fig (bool): Whether to return the figure object.
    - conf_matrix_text_kws (dict): Additional keyword arguments for the text in the confusion matrix plot.
    - print_report (bool): Whether to print the classification report.
    - return_str_report (bool): Whether to return the classification report as a string.

    Returns:
    - If output_dict is True, returns a dictionary containing the classification report.
    - If return_fig is True, returns the figure object.
    - If return_str_report is True, returns the classification report as a string.
    - If none of the above conditions are met, returns a tuple containing the dictionary and figure object.

    Note: 
        This is a modified version of the classification metrics function from Intro to Machine Learning.
        Updates:
          - Reversed raw counts confusion matrix cmap (so darker==more).
          - Added arg for normalized confusion matrix values_format.
          - Added arg for text_kw in confusion matrix display.
          - Added option to return classification report as a string.
          - Added option to return classification report as a dictionary.
          - Added option to return the figure object.
          - Made printing the report optional.
          
    Note: Only evaluate_classification_network has been updated to use this new function.
    evaluate_classification has not been updated.
    """
    from sklearn.metrics import classification_report, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    import numpy as np
    import warnings
    
    # Silence warning when 0 labels in classification report
    from sklearn.exceptions import UndefinedMetricWarning
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    ## Get classification report
    report_str = classification_report(y_true, y_pred, target_names=target_names)
    
    ## Print header and report
    header = "-"*80
    report = header + f"\n Classification Metrics: {label} \n" + header + f"\n{report_str}" +"\n" + header
    
    if print_report:
        print(report)
    
    ## CONFUSION MATRICES SUBPLOTS
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    
    # Create a confusion matrix of raw counts (left subplot)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                            normalize=None, 
                                            cmap='gist_gray_r',# Updated cmap
                                            values_format="d", 
                                            colorbar=colorbar,
                                            ax = axes[0], 
                                            display_labels=target_names,
                                            text_kw=conf_matrix_text_kws);
    axes[0].set_title("Raw Counts")
    
    # Create a confusion matrix with the data with normalize argument 
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                            normalize=normalize,
                                            cmap=cmap, 
                                            values_format=values_format, #New arg
                                            colorbar=colorbar,
                                            ax = axes[1],
                                            display_labels=target_names,
                                            text_kw=conf_matrix_text_kws);
    axes[1].set_title("Normalized Confusion Matrix")
    
    # Adjust layout and show figure
    fig.tight_layout()
    plt.show()
    
    # Return dictionary of classification_report
    return_list = []
    if output_dict==True:
        report_dict = classification_report(y_true, y_pred,target_names=target_names, output_dict=True)
        return_list.append(report_dict)
        # return report_dict
    
        
    if return_fig==True:
        return_list.append(fig)
        
    if return_str_report:
        return_list.append(report_str)

    return return_list[0] if len(return_list)==1 else tuple(return_list)
    # elif return_fig == True:
    #     return fig
    
    

def evaluate_classification_network(model, 
                                    X_train=None, y_train=None, 
                                    X_test=None, y_test=None,
                                    history=None, history_figsize=(8,3),
                                    figsize=(6,4), normalize='true',
                                    output_dict=False,
                                    cmap_train='Blues',
                                    cmap_test="Reds",
                                    values_format=".2f", 
                                    colorbar=False, target_names=None, 
                                    return_fig_history=False, return_fig_conf_matrix=False, as_frame=False, frame_include_support=True, 
                                    frame_include_macro_avg=True, return_str_report=False,
                                    conf_matrix_text_kws={}):
    """
    Evaluates a classification network model using training and test data.

    Parameters:
    - model: The classification network model to evaluate.
    - X_train: The training data features. Default is None.
    - y_train: The training data labels. Default is None.
    - X_test: The test data features. Default is None.
    - y_test: The test data labels. Default is None.
    - history: The training history of the model. Default is None.
    - history_figsize: The size of the history plot. Default is (8, 3).
    - figsize: The size of the confusion matrix plot. Default is (6, 4).
    - normalize: Whether to normalize the confusion matrix. Default is 'true'.
    - output_dict: Whether to return the results as a dictionary. Default is False.
    - cmap_train: The color map for the training data confusion matrix. Default is 'Blues'.
    - cmap_test: The color map for the test data confusion matrix. Default is 'Reds'.
    - values_format: The format of the values in the confusion matrix. Default is '.2f'.
    - colorbar: Whether to show the color bar in the confusion matrix. Default is False.
    - target_names: The names of the target classes. Default is None.
    - return_fig_history: Whether to return the history plot as a figure. Default is False.
    - return_fig_conf_matrix: Whether to return the confusion matrix plot as a figure. Default is False.
    - as_frame: Whether to return the results as a dictionary of dataframes. Default is False.
    - frame_include_support: Whether to include support in the dataframe. Default is True.
    - frame_include_macro_avg: Whether to include macro average in the dataframe. Default is True.
    - return_str_report: Whether to return the classification report as a string. Default is False.
    - conf_matrix_text_kws: Additional keyword arguments for the text in the confusion matrix. Default is {}.

    Returns:
    - If output_dict is False and neither return_fig_history nor return_fig_conf_matrix is True, returns None.
    - If output_dict is True, returns the results as a dictionary.
    - If return_fig_history or return_fig_conf_matrix is True, returns the figures as a dictionary.
    - If both output_dict and return_fig_history or return_fig_conf_matrix are True, returns a tuple of the results dictionary and the figures dictionary.
    """
    # Check if X_train or X_test is provided
    if (X_train is None) & (X_test is None):
        raise Exception('\nEither X_train & y_train or X_test & y_test must be provided.')
    
    # Initialize dict
    results_dict = {}
    fig_dict=  {}
    
    # Shared Kwargs
    shared_kwargs = dict(output_dict=True, 
                      figsize=figsize,
                      colorbar=colorbar,
                      values_format=values_format, 
                      target_names=target_names,
                      conf_matrix_text_kws=conf_matrix_text_kws,
                      return_fig = return_fig_conf_matrix,
                      return_str_report=return_str_report,
                    
                      )
    ## Adding a Print Header
    print("\n"+'='*80)
    print('- Evaluating Network...')
    print('='*80)
    
    # Plot history, if provided
    if history is not None:
        fig_history = plot_history(history, figsize=history_figsize, ncols=2, return_fig=return_fig_history)
        fig_dict['history'] = fig_history
        # Show history from abov
        plt.show()


    ## TRAINING DATA EVALUATION
    # check if X_train was provided
    if X_train is not None:
        ## Run model.evaluate         
        print("\n- Evaluating Training Data:")

        ## Check if X_train is a dataset
        if hasattr(X_train,'map'):
            
            # Run keras model.evaluate (without y since its a dataset)
            print(model.evaluate(X_train, return_dict=True),'\n')
            
            ## Get predictions for training data
            # extract y_train and y_train_pred with helper function
            y_train, y_train_pred = get_true_pred_labels(model, X_train)
        else:
            # Run keras model.evaluate (with y since its an array)
            print(model.evaluate(X_train,y_train, return_dict=True),'\n')
            
            ## Get predictions for training data
            y_train_pred = model.predict(X_train)
            
        # # Show history from above
        # if history is not None: 
        #     fig_history.show()

        ## Pass both y-vars through helper compatibility function
        y_train = convert_y_to_sklearn_classes(y_train)
        y_train_pred = convert_y_to_sklearn_classes(y_train_pred)
        
        
        # Call the helper function to obtain regression metrics for training data
        results_train = classification_metrics(y_train, y_train_pred, 
                                               cmap=cmap_train,label='Training Data',
                                               **shared_kwargs,
                                               )

        # Check how many results are returned
        if isinstance(results_train, tuple):
            
            if len(results_train)==2:
                # Unpack tuple and set train_report_str to None
                results_train, fig_conf_matrix_train = results_train
                train_report_str=None
            
            elif len(results_train)==3:
                # Unpack tuple
                results_train, fig_conf_matrix_train, train_report_str = results_train    
            
            else:
                raise ValueError("Results tuple has unexpected length.")
        
        # If only one result returned
        else:
            fig_conf_matrix_train = None
            train_report_str = None
            
        # Add results to dict               
        results_dict['train'] = results_train
        
        # Add confusion matrix to fig_dict
        if fig_conf_matrix_train is not None:
            fig_dict['train'] = {"confusion_matrix":fig_conf_matrix_train}  
        # Add report_str to results_dict
        
        if train_report_str is not None:
            fig_dict['train']['report_str'] = train_report_str
            

    ## TEST DATA EVALUATION
    # check if X_test was provided
    if X_test is not None:
        ## Run model.evaluate         
        print("\n- Evaluating Test Data:")
        
    
        ## Check if X_train is a dataset
        if hasattr(X_test,'map'):
            
            # Run keras model.evaluate (without y since its a dataset)
            print(model.evaluate(X_test, return_dict=True),'\n')
            
            ## Get predictions for training data
            # extract y_train and y_train_pred with helper function
            y_test, y_test_pred = get_true_pred_labels(model, X_test)
        else:
            # Run keras model.evaluate (with y since its an array)
            print(model.evaluate(X_test,y_test, return_dict=True),'\n')
            
            ## Get predictions for training data
            y_test_pred = model.predict(X_test)
        

        
        ## Pass both y-vars through helper compatibility function
        y_test = convert_y_to_sklearn_classes(y_test)
        y_test_pred = convert_y_to_sklearn_classes(y_test_pred)
        
        # Call the helper function to obtain regression metrics for training data
        results_test = classification_metrics(y_test, y_test_pred, cmap=cmap_test,label='Test Data', **shared_kwargs)
        # Check how many results are returned
        if isinstance(results_test, tuple):
            
            if len(results_test)==2:
                # Unpack tuple and set train_report_str to None
                results_test, fig_conf_matrix_test = results_test
                test_report_str=None
            
            elif len(results_test)==3:
                # Unpack tuple
                results_test, fig_conf_matrix_test, test_report_str = results_test    
            
            else:
                raise ValueError("Results tuple has unexpected length.")
        
        # If only one result returned
        else:
            fig_conf_matrix_test = None
            test_report_str = None
            
        # Add results to dict               
        results_dict['test'] = results_test
        
        # Add confusion matrix to fig_dict
        fig_dict['test'] = {}
        
        if fig_conf_matrix_test is not None:
            fig_dict['test']["confusion_matrix"]=fig_conf_matrix_test
        # Add report_str to results_dict
        
        if test_report_str is not None:
            fig_dict['test']['report_str'] = test_report_str
        
    # If no resuls returned, return None
    if not output_dict and not return_fig_conf_matrix and not return_fig_history:
        return 
    
    # List of return values      
    return_list = []
    
    # If output_dict is True, return the results_dict
    if (output_dict == True):    
        
        # Convert to dictionary of dataframes
        if as_frame:
            
            final_results = {}
            # results_dict = {k: get_results_df(results_dict, results_key=k, include_support=frame_include_support,
            #                                   include_macro_avg=frame_include_macro_avg) for k in results_dict.keys()
            #                 }
            
            
            for k in results_dict.keys():
                split_results = get_results_df(results_dict, results_key=k, include_support=frame_include_support,
                                               include_macro_avg=frame_include_macro_avg)
                
                # Ubpack split results
                if isinstance(split_results, tuple):
                    if len(split_results)==2:
                        final_results[k] = {'results-classes':split_results[0],
                                            'results-overall':split_results[1]}
                    else:
                        final_results[k] = {'results-classes':split_results[0]}#,
                    
        else:
            final_results = results_dict

        # Append to return list
        return_list.append(final_results)
        
    # if either figure is requested, append to return list
    if (return_fig_history==True) | (return_fig_conf_matrix==True):
        return_list.append(fig_dict)
        
    # Return final  results
    return return_list[0] if len(return_list)==1 else tuple(return_list)
    




def plot_history(history, figsize=(8,4), return_fig=False,ncols=2,
                 suptitle="Training History",suptitle_y=1.02, suptitle_fontsize=16):
    """Plots the training and validation curves for all metrics in a Tensorflow History object.

    Args:
        history (Tensorflow History): History output from training a neural network.
        figsize (tuple, optional): Total figure size. Defaults to (6,8).
        return_fig (boolean, optional): If true, return figure instead of displaying it with plt.show()

    Returns:
        None or matplotlib.figure.Figure: If return_fig is True, returns the figure object. Otherwise, displays the figure using plt.show().
    """
    import matplotlib.pyplot as plt
    import numpy as np
    # Get a unique list of metrics 
    all_metrics = np.unique([k.replace('val_','') for k in history.history.keys()])
    nrows = len(all_metrics)//ncols
    # nrows = len(unique_labels)//ncols + 1
    
    
    
    # Plot each metric
    # n_plots = len(all_metrics)
    
    fig, axes = plt.subplots(nrows=nrows,ncols=ncols, figsize=figsize)
    axes = axes.flatten()
    
    # Loop through metric names add get an index for the axes
    for i, metric in enumerate(all_metrics):
        # Get the epochs and metric values
        epochs = history.epoch
        score = history.history[metric]
        # Plot the training results
        axes[i].plot(epochs, score, label=metric, marker='.')
        # Plot val results (if they exist)
        try:
            val_score = history.history[f"val_{metric}"]
            axes[i].plot(epochs, val_score, label=f"val_{metric}",marker='.')
        except:
            pass
        finally:
            axes[i].legend()
            axes[i].set(title=metric, xlabel="Epoch",ylabel=metric)
   
    # Adjust subplots and show
    fig.tight_layout()
 
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=suptitle_fontsize, y=suptitle_y)
        
    if return_fig:
        return fig
    else:
        plt.show()


def get_true_pred_labels(model,ds):
    """Gets the labels and predicted probabilities from a Tensorflow model and Dataset object.
    Adapted from source: https://stackoverflow.com/questions/66386561/keras-classification-report-accuracy-is-different-between-model-predict-accurac
    """
    y_true = []
    y_pred_probs = []
    
    # Loop through the dataset as a numpy iterator
    for images, labels in ds.as_numpy_iterator():
        
        # Get prediction with batch_size=1
        y_probs = model.predict(images, batch_size=1, verbose=0)

        # Combine previous labels/preds with new labels/preds
        y_true.extend(labels)
        y_pred_probs.extend(y_probs)

    ## Convert the lists to arrays
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    
    return y_true, y_pred_probs
    

def convert_y_to_sklearn_classes(y, verbose=False):
    # If already one-dimension
    if np.ndim(y)==1:
        if verbose:
            print("- y is 1D, using it as-is.")
        return y
        
    # If 2 dimensions with more than 1 column:
    elif y.shape[1]>1:
        if verbose:
            print("- y has is 2D with >1 column. Using argmax for metrics.")   
        return np.argmax(y, axis=1)
    
    else:
        if verbose:
            print("y has 2D with 1 column. Using round for metrics.")
        return np.round(y).flatten().astype(int)




    
def evaluate_classification(model, X_train=None, y_train=None, X_test=None, y_test=None,
                            figsize=(6,4), normalize='true', output_dict = False,
                            cmap_train='Blues', cmap_test="Reds",colorbar=False,
                            values_format='.2f',
                            target_names=None, return_fig=False):
    """Evalutes an sklearn-compatible classification model on training and test data. 
    For each data split, return the classification report and confusion matrix display. 

    Args:
        model (sklearn estimator): Classification model to evaluate.
        X_train (Frame/Array, optional): Training data. Defaults to None.
        y_train (Series/Array, optional): Training labels. Defaults to None.
        X_test (Frame/Array, optional): Test data. Defaults to None.
        y_test (Series/Array, optional): Test labels. Defaults to None.
        figsize (tuple, optional): figsize for confusion matrix subplots. Defaults to (6,4).
        normalize (str, optional): arg for sklearn's ConfusionMatrixDisplay. Defaults to 'true' (conf mat values normalized to true class).  
        output_dict (bool, optional):  Return the results of classification_report as a dict. Defaults to False. Defaults to False.
        cmap_train (str, optional): Colormap for the ConfusionMatrixDisplay for training data. Defaults to 'Blues'.
        cmap_test (str, optional): Colormap for the ConfusionMatrixDisplay for test data.  Defaults to "Reds".
        colorbar (bool, optional): Arg for ConfusionMatrixDispaly: include colorbar or not. Defaults to False.
        values_format (str, optional): Format values on confusion matrix. Defaults to ".2f".
        target_names (array, optional): Text labels for the integer-encoded target. Passed in numeric order [label for "0", label for "1", etc.]
        return_fig (bool, optional): Whether the matplotlib figure for confusion matrix is returned. Defaults to False.
                                          Note: Must set outout_dict to False and set return_fig to True to get figure returned.
     Returns (Only 1 value is returned, but contents vary):
        dict: Dictionary that contains results for "train" and "test. 
              Contents of dictionary depending on output_dict and return_fig:
              - if output_dict==True and return_fig==False: returns dictionary of classification report results
            - if output_dict==False and return_fig==True: returns dictionary of confusion matrix displays.
    """
    # Combining arguments used for both training and test results
    shared_kwargs = dict(output_dict=output_dict,  # output_dict: Changed from hard-coded True
                    #   figsize=figsize, 
                      colorbar=colorbar, 
                      target_names=target_names,
                      values_format=values_format,
                      return_fig=return_fig)
 
    if (X_train is None) & (X_test is None):
        raise Exception('\nEither X_train & y_train or X_test & y_test must be provided.')
 
    if (X_train is not None) & (y_train is not None):
        # Get predictions for training data
        y_train_pred = model.predict(X_train)
        # Call the helper function to obtain regression metrics for training data
        results_train = classification_metrics(y_train, y_train_pred, cmap=cmap_train, figsize=figsize,label='Training Data', **shared_kwargs)
        print()
    else:
        results_train=None
  
    if (X_test is not None) & (y_test is not None):
        # Get predictions for test data
        y_test_pred = model.predict(X_test)
        # Call the helper function to obtain regression metrics for test data
        results_test = classification_metrics(y_test, y_test_pred,  cmap=cmap_test, figsize=figsize, label='Test Data' , **shared_kwargs)
    else:
        results_test = None
  
  
    if (output_dict == True) | (return_fig==True):
        # Store results in a dataframe if ouput_frame is True
        results_dict = {'train':results_train,
                        'test': results_test}
        return results_dict




def get_results_df(results_dict, results_key='test', 
                   average_rowname= 'macro avg',
                   include_support = True,
                   include_macro_avg=True):
    """
    Convert a results dictionary into a pandas DataFrame.

    Parameters:
    - results_dict (dict): A dictionary containing the results.
    - results_key (str): The key in the dictionary that contains the results. Default is 'test'.
    - average_rowname (str): The name of the row that represents the average. Default is 'macro avg'.
    - include_support (bool): Whether to include the 'support' column in the DataFrame. Default is True.
    - include_macro_avg (bool): Whether to include the 'macro avg' row in the DataFrame. Default is True.

    Returns:
    - results_df (pandas DataFrame): A DataFrame containing the results.

    """
    import pandas as pd
    results = results_dict[results_key].copy()
    
    # Remove accuracy and macro avg from results
    accuracy = results.pop('accuracy')
    macro_avg = results.pop('macro avg')
    _ = results.pop('weighted avg')
    
    # Create DataFrames
    results_df = pd.DataFrame(results).T
    
    
    if include_macro_avg:
        overall_results = pd.DataFrame(macro_avg, index=[average_rowname])#.T
        overall_results['accuracy'] = accuracy
        
        ## Concatenate the overall results to the results_df
        # results_df = pd.concat([results_df, overall_results],axis=0)
        # results_df.loc[average_rowname,'accuracy'] = accuracy
    
    # Recast support as int
    results_df['support'] = results_df['support'].astype(int)

    # Move the support column to the end
    # results_df = results_df[ results_df.drop(columns='support').columns.tolist() + ['support']]
    
    if not include_support:
        results_df = results_df.drop(columns='support')
    
    if include_macro_avg:
        return results_df, overall_results
    else:
        return results_df





## Previous Helper Function    
def convert_y_to_sklearn_classes(y, verbose=False):
    """
    Converts the target variable y to the appropriate format for sklearn classification models.

    Parameters:
    - y: The target variable array.
    - verbose: Whether to print verbose output. Default is False.

    Returns:
    - The converted target variable array.

    If y is already one-dimensional, it is returned as-is.
    If y is two-dimensional with more than one column, the argmax function is used to determine the class labels.
    If y is two-dimensional with one column, the round function is used to determine the class labels.

    Note: The returned array is flattened and casted to integer type.
    """
    # If already one-dimension
    if np.ndim(y)==1:
        if verbose:
            print("- y is 1D, using it as-is.")
        return y
        
    # If 2 dimensions with more than 1 column:
    elif y.shape[1]>1:
        if verbose:
            print("- y has is 2D with >1 column. Using argmax for metrics.")   
        return np.argmax(y, axis=1)
    
    else:
        if verbose:
            print("y has 2D with 1 column. Using round for metrics.")
        return np.round(y).flatten().astype(int)


def get_true_pred_labels_images(model, ds, include_images=True, convert_y_for_sklearn=False):
    """
    Gets the true labels, predicted probabilities, and images (optional) from a Tensorflow model and Dataset object.

    Args:
        model (tf.keras.Model): The trained Tensorflow model.
        ds (tf.data.Dataset): The dataset object containing the images and labels.
        include_images (bool, optional): Whether to include the images in the output. Defaults to True.
        convert_y_for_sklearn (bool, optional): Whether to convert the labels for sklearn compatibility. Defaults to False.

    Returns:
        tuple: A tuple containing the true labels, predicted probabilities, and images (optional).

    Adapted from source: 
    https://stackoverflow.com/questions/66386561/keras-classification-report-accuracy-is-different-between-model-predict-accurac
    """
    y_true = []
    y_pred_probs = []
    all_images = []
    
    # Loop through the dataset as a numpy iterator
    for images, labels in ds.as_numpy_iterator():
        
        # Get prediction with batch_size=1
        y_probs = model.predict(images, batch_size=1, verbose=0)

        # Combine previous labels/preds with new labels/preds
        y_true.extend(labels)
        y_pred_probs.extend(y_probs)

        if include_images == True:
            all_images.extend(images)
            
    # Convert the lists to arrays
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)

    # Convert to classes (or not)
    if convert_y_for_sklearn == True:
        # Use our previous helper function to make preds into classes
        y_true = convert_y_to_sklearn_classes(y_true)
        y_pred = convert_y_to_sklearn_classes(y_pred_probs)
    else: 
        y_pred = y_pred_probs

    # If the images should be included or not
    if include_images == False:
        return y_true, y_pred
    else:
        # Convert images to array and return everything
        all_images = np.array(all_images)
        return y_true, y_pred, all_images


