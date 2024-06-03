#Set up logging
import logging
import time
import datetime as dt

def initialize_logs(log_file = 'logs/nn_training.log', overwrite_logs=True,
                    log_header = ";start_time;name;fit_time;metrics;model_filepaths",):
    
    
    # #add deleting log file
    if overwrite_logs==True:
        filemode = "w"
        force=True
    else:
        filemode = 'a'
        force = False
        
    logging.basicConfig(filename=log_file, level=logging.INFO, filemode=filemode,force=force)#, format='%(message)s')
    logging.info(log_header)



# Function to log neural network details
def log_nn_details(start_time, name, fit_time, results_overall,model_filepaths=None,
                #    model_fpath=None, model_classification_report_fpath=None,
                #    model_history_fpath=None, model_confusion_matrix_fpath=None, 
                   sep=";"):
    
    if model_filepaths is not None:
        fpaths_dict= model_filepaths
    else:
        fpaths_dict = "N/A"
        
    
    # Record results (except for filepaths)
    metrics = results_overall.loc['macro avg'].to_dict()
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


