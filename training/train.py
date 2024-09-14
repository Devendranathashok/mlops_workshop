import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import lightgbm as lgb

def preprocess_data(data_df):
    """Preprocess the DataFrame to handle categorical features"""
    # Convert categorical columns to 'category' dtype
    for col in data_df.select_dtypes(include=['object']).columns:
        data_df[col] = data_df[col].astype('category')

    return data_df

# Split the dataframe into test and train data
def split_data(data_df):
    """Split a dataframe into training and validation datasets"""
    features = data_df.drop(['target', 'id'], axis=1)
    labels = np.array(data_df['target'])

    features_train, features_valid, labels_train, labels_valid = \
        train_test_split(features, labels, test_size=0.2, random_state=0)

    categorical_features = list(features.select_dtypes(['category']).columns)

    train_data = lgb.Dataset(features_train, label=labels_train, categorical_feature=categorical_features)
    valid_data = lgb.Dataset(features_valid, label=labels_valid, categorical_feature=categorical_features, free_raw_data=False)

    return (train_data, valid_data)

# Train the model, return the model
def train_model(data, parameters):
    """Train a model with the given datasets and parameters"""
    train_data = data[0]
    valid_data = data[1]

    model = lgb.train(parameters,
                      train_data,
                      valid_sets=valid_data,
                      num_boost_round=500)

    return model

# Evaluate the metrics for the model
def get_model_metrics(model, data):
    """Construct a dictionary of metrics for the model"""
    predictions = model.predict(data[1].data)
    fpr, tpr, thresholds = metrics.roc_curve(data[1].label, predictions)
    model_metrics = {"auc": metrics.auc(fpr, tpr)}
    print(model_metrics)

    return model_metrics

# Main script
if __name__ == "__main__":
    # Load your data
    data_df = pd.read_csv('your_data.csv')

    # Preprocess data
    data_df = preprocess_data(data_df)
    
    # Split data
    data = split_data(data_df)
    
    # Define LightGBM parameters
    parameters = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt'
    }
    
    # Train the model
    model = train_model(data, parameters)
    
    # Get model metrics
    get_model_metrics(model, data)
