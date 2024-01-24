# Introduction 
This code trains a machine learning model for given a basin and list of reservoirs provided by the user. 

# How to Run
Run train_models function from the model_workflow.py file with basin and reservoirs. 

# Inputs
- Name of the basin
- List of Reservoirs

# Outputs
- Model Performance dictionary: 
  - Performance of the tuned model as a dictionary
  - Model performance on the blind test data a dataframe
  - Model performance on the validation data a dataframe
- Packaged Model: 
  - Object of ModelPackager class
  - packaged_models attribute contains the trained model and the scalers.
- Model metadata: dictionary containing the filtereing, the training, and the hyperparameter tuning configs.

# Requirements
- Python==3.10.*
- numpy==1.24.1
- pandas==1.5.3
- pyspark==3.3.1
- scikit-learn==1.2.1
- scipy==1.10.0
- xgboost==1.7.3
