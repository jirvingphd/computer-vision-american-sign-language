#Set up logging
import logging
import time
import datetime as dt
import os
import matplotlib.pyplot as plt

def initialize_logs(log_file='logs/nn_training.log', overwrite_logs=True,
                    log_header=";start_time;name;fit_time;metrics;model_filepaths"):
    """
    Initializes the logging configuration and sets up the log file.

    Parameters:
    log_file (str): The path to the log file. Default is 'logs/nn_training.log'.
    overwrite_logs (bool): If True, the log file will be overwritten. If False, new logs will be appended to the existing file. Default is True.
    log_header (str): The header to be written in the log file. Default is ";start_time;name;fit_time;metrics;model_filepaths".

    Returns:
    None
    """
    
    if overwrite_logs:
        filemode = "w"
        force = True
    else:
        filemode = 'a'
        force = False
        
    logging.basicConfig(filename=log_file, level=logging.INFO, filemode=filemode, force=force)
    logging.info(log_header)
    


# Function to log neural network details
def log_nn_details(start_time, name, fit_time, results_overall, model_filepaths=None, sep=";", **kwargs):
    """
    Logs the details of a neural network model.

    Parameters:
    - start_time (datetime): The start time of the model training.
    - name (str): The name of the model.
    - fit_time (float): The time taken to fit the model.
    - results_overall (pandas.DataFrame or dict): The overall results of the model evaluation.
    - model_filepaths (dict or None): The filepaths of the model files, if available.
    - sep (str): The separator used to format the log information.
    - **kwargs: Additional keyword arguments.

    Returns:
    None
    """

    if model_filepaths is not None:
        fpaths_dict = model_filepaths
    else:
        fpaths_dict = "N/A"

    # Record results (except for filepaths)
    try:
        metrics = results_overall.loc['macro avg'].to_dict()
    except:
        metrics = results_overall

    info = f"{sep}{start_time.strftime('%m/%d/%Y %T')}{sep}{name}{sep}{fit_time}{sep}{metrics}{sep}{fpaths_dict}"

    ## Log Info
    logging.info(info)
    

    


def save_model_results(model_results, model_directory='modeling/models', model_save_format='tf'):
    """
    Saves the model, classification report, training history, and confusion matrix to specified directory.
    
    Parameters:
        filepath (str): Base name for the files to be saved.
        model_directory (str): Directory where files will be saved.
        model_results (dict): Dictionary containing model, history, and metrics.
        model_save_format (str): Format to save the model, default is 'tf'.
    
    Returns:
        dict: Dictionary containing paths to the saved files.
        
        
    # Example Usage:
    # model_results = {
    #     'model': tf.keras.models.Sequential([...]),  # Your trained model
    #     'history': plt.figure(),  # A matplotlib figure of your training history
    #     'result_figs': {
    #         'test': {
    #             'classification_report': "Your classification report text",
    #             'confusion_matrix': plt.figure()  # A matplotlib figure of your confusion matrix
    #         }
    #     }
    # }
    # filepaths = save_model_results(', 'model_directory', model_results)
    """
    import os
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
    # Use model.name as the base name for the files
    model = model_results['model']
    
    model_directory = os.path.join(model_directory, model.name)
    # model_fpath = os.path.join(model_directory,f"{model.name}")
    

    # Generate file paths
    if model_save_format == 'tf':
        model_save_fpath = model_directory#os.path.join(model_directory,f"{model.name}/")
    else:
        model_save_fpath = os.path.join(model_directory, f"model.{model_save_format}")
        
    # Ensure the model directory exists
    if os.path.isdir(model_save_fpath)==True:
        os.makedirs(model_save_fpath, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(model_save_fpath), exist_ok=True)

    save_classification_report_fpath = os.path.join(model_directory, "classification_report.txt")
    save_history_fpath = os.path.join(model_directory, f"history.png")
    save_confusion_matrix_fpath = os.path.join(model_directory, "confusion_matrix.png")
    
    # Save model
    try:
        model_results['model'].save(model_save_fpath, save_format=model_save_format)
        print(f"- Model saved to {model_save_fpath}")
    except Exception as e:
        print(f"[!] Error saving model: {e}")

    # Save classification report
    try:
        report_str = model_results['classification_report']
        with open(save_classification_report_fpath, "w") as f:
            f.write(report_str)
        print(f"- Classification Report saved to {save_classification_report_fpath}")
    except Exception as e:
        display(f"[!] Error saving classification report: {e}")

    # Save training history
    try:
        history_fig = model_results['history']
        history_fig.savefig(save_history_fpath, dpi=300, bbox_inches='tight', transparent=False)
        print(f"- History figure saved to {save_history_fpath}")
    except Exception as e:
        display(f"[!] Error saving history figure: {e}")
    # Save confusion matrix
    try:
        confusion_matrix_fig = model_results['result_figs']['test']['confusion_matrix']
        confusion_matrix_fig.savefig(save_confusion_matrix_fpath, dpi=300, bbox_inches='tight', transparent=False)
        print(f"- Confusion Matrix figure saved to {save_confusion_matrix_fpath}")
    except Exception as e:
        display(f"[!] Error saving confusion matrix figure: {e}")

    return {
        "model_save_fpath": model_save_fpath, 
        "save_classification_report_fpath": save_classification_report_fpath,
        "save_history_fpath": save_history_fpath, 
        "save_confusion_matrix_fpath": save_confusion_matrix_fpath
    }


import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array, array_to_img

def load_model_results(model_fpath=None, model_name=None, model_directory=None,
                       load_model=True, figs_as_matplotlib=True,
                       ):
    """
    Loads the model, classification report, training history, and confusion matrix from the specified directory.
    
    Parameters:
        model_name (str): Base name for the files to be loaded.
        model_directory (str): Directory where files are saved.
    
    Returns:
        dict: Dictionary containing the loaded files.
    """
    
    # Load model
    if model_fpath is None and model_name is  None and model_directory is  None:
                raise Exception("Please provide a model_fpath or model_name and model_directory.")

    if model_fpath is None and model_name is not None and model_directory is not None:
        model_fpath = os.path.join(model_directory, model_name)
    else:
        model_fpath = model_fpath

    # Load classification report
    try:
        classification_report_fpath = os.path.join(model_fpath, "classification_report.txt")
        with open(classification_report_fpath, "r") as f:
            classification_report = f.read()
    except Exception as e:
        classification_report = None
        print(f"Error loading classification report: {e}")
    
    # Load training history
    history_fpath = os.path.join(model_fpath, f"history.png")
    
    try:
        if figs_as_matplotlib:
            history_fig,ax  = plt.subplots()
            ax.imshow(plt.imread(history_fpath))
            ax.axis('off')

        else:
            history_fig = load_img(history_fpath)
    except Exception as e:
        history_fig = None
        print(f"Error loading history: {e}")
    # history_fig = plt.imread(history_fpath)
    
    try:
        # Load confusion matrix
        confusion_matrix_fpath = os.path.join(model_fpath, "confusion_matrix.png")
        
        if figs_as_matplotlib:
            confusion_matrix_fig, ax  = plt.subplots()
            ax.imshow(plt.imread(confusion_matrix_fpath))
            ax.axis('off')

            
        else:
            confusion_matrix_fig = load_img(confusion_matrix_fpath)
            
    except Exception as e:
        confusion_matrix_fig = None
        print(f"Error loading confusion matrix: {e}")
    
    loaded =  {
        "classification_report": classification_report,
        "history_fig": history_fig,
        "confusion_matrix_fig": confusion_matrix_fig
    }
    
    if load_model:
        try:
            model = tf.keras.models.load_model(model_fpath)
        except Exception as e:
            model = None
            print(f"Error loading model: {e}")
        loaded['model'] = model
    return loaded




def parse_log_file(log_file, sep=';', keep_only_startswith=['info:root'], clean_results=True,remove_fpaths=True,
                   save_csv=True, save_fpath=None):
    """
    Parses a log file and returns a pandas DataFrame containing the log data.

    Parameters:
    - log_file (str): The path to the log file.
    - sep (str): The separator used to split the log lines into columns. Default is ';'.
    - keep_only (str): The prefix of the lines to keep in the log file. Default is 'info:root'.

    Returns:
    - logs_df (pandas DataFrame): The parsed log data as a DataFrame.

    """
    if (save_fpath is None) and (save_csv==True):
        save_fpath = log_file.replace('.log', '.csv')
    import ast
    import pandas as pd
    
    # Read logs
    with open(log_file, 'r') as file:
        log_lines = file.readlines()
        file.seek(0)
        
    # Remove unwanted lines
    split_lines = []
    for line in log_lines:
        compare_line = line.strip().lower()
        for keep_only in keep_only_startswith:
            if compare_line.startswith(keep_only.lower()):
                split_lines.append(line.strip().strip(sep).split(sep))
                # break
            
        # if line.strip().lower().startswith(keep_only):
        #     split_lines.append(line.strip().split(sep))

    # Create DataFrame
    # try:
    logs_df = pd.DataFrame(split_lines[1:], columns=split_lines[0])
    # ?except Exception as e:
    #     display(e)
    #     # display(split_lines[0])
    #     logs_df = pd.DataFrame(split_lines)
    #     display(logs_df)
    # Fill NaN values and convert metrics column to dictionary
    logs_df = logs_df.fillna("{}")
    logs_df['metrics'] = logs_df['metrics'].map(lambda x: ast.literal_eval(x))
    metrics_df = pd.json_normalize(logs_df['metrics'])
    
    ## Concatenate metrics columns to main dataframe
    logs_df = pd.concat([logs_df.drop(columns=['metrics']), metrics_df], axis=1)

    # If model_filepaths column exists, convert to dictionary and normalize
    if 'model_filepaths' in logs_df.columns:
        logs_df['model_filepaths'] = logs_df['model_filepaths'].map(lambda x: ast.literal_eval(x))
        fpaths_df = pd.json_normalize(logs_df['model_filepaths'])
        
        
        logs_df = pd.concat([logs_df.drop(columns='model_filepaths'), fpaths_df], axis=1)
    
        # logs_df.columns=[c.replace("_"," ").title() for c in logs_df.columns]

    if save_csv and not clean_results:
        # Make titled columns
        logs_df.to_csv(save_fpath, index=False)
        print(f"\n[i] Saved parsed logs to {save_fpath}")



    if clean_results:
        drop_cols = ['support']
        
        if remove_fpaths==True:
            try:
                drop_cols.extend([f for f in logs_df.drop(columns="model_save_fpath").columns if 'fpath' in f])
            except:
                
                display(logs_df.head())
            # except Exception as e:disp
            
        
        first_col = logs_df.columns[0]
        if first_col.lower().startswith("info:root"):
            drop_cols.append(first_col)
        logs_df = logs_df.drop(columns=drop_cols, errors='ignore')# errors='ignore')
        
        logs_df = logs_df.round(3)
        
        ## rearrange columns
        move_to_end = ['fit_time','model_save_fpath']
        for col in move_to_end:
            if col not in logs_df.columns:
                move_to_end.remove(col)
                
        if logs_df.columns.isin(move_to_end).any():
            logs_df = logs_df[ logs_df.drop(columns=move_to_end, errors='ignore').columns.tolist()+move_to_end]
        # Make titled columns
        logs_df.columns=[c.replace("_"," ").title() for c in logs_df.columns]
        
        if save_csv:
            logs_df.to_csv(save_fpath, index=False)
            print(f"\n[i] Saved parsed logs to {save_fpath}")
            
    

    return logs_df