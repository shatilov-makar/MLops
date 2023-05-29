import pytest
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score


@pytest.mark.parametrize("model_path",['data/Gradient Boosting.sav','data/Naive Bayes.sav','data/Random Forest.sav'])
@pytest.mark.parametrize("dataset", ['data/test_dataset.csv'])
def test_model(model_path, dataset):
    model = pickle.load(open(model_path, 'rb'))
    tokenizer = pickle.load(open('data/tokenizer.pkl', 'rb'))
    df = pd.read_csv(dataset)

    tokens = tokenizer.transform(df['preprocessed_text'])
    y_pred = model.predict(tokens)
    accuracy = accuracy_score(df['Score'], y_pred)
    assert accuracy > 0.85


@pytest.mark.parametrize("dataset", ['data/init_dataset.csv','data/test_dataset.csv'])
def test_numbers_in_datasets(dataset):
    df = pd.read_csv(dataset)    
    assert ~any(df['preprocessed_text'].str.isnumeric())



@pytest.mark.parametrize("dataset", ['data/init_dataset.csv','data/test_dataset.csv'])
def test_nulls_in_datasets(dataset):
    df = pd.read_csv(dataset)    
    assert ~any(df['preprocessed_text'].isna())


@pytest.mark.parametrize("dataset", ['data/init_dataset.csv','data/test_dataset.csv'])
def test_capital_letters_in_datasets(dataset):
    df = pd.read_csv(dataset)    
    assert ~any(df['preprocessed_text'].str.istitle())