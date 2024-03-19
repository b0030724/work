import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


data_URL = "https://raw.githubusercontent.com/txt2vz/pythonAI/main/diabetes.csv"
diabetes_data = pd.read_csv(data_URL)
