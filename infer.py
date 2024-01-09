import streamlit as st
import numpy as np
import pickle
import sklearn
from xgboost import XGBRegressor


def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    return model


model = load_model()
a = [1] * 15
X = np.array(a)
X = X.reshape((1, 15))
print(X.shape)
print(model.predict(X))
