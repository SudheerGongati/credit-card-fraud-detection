#importing packages
from random import random
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import timeit
warnings.filterwarnings("ignore")
import streamlit as st

# setting title for the application
st.title("Credit Card Fraud Detection !!")

# reading data
df = st.cache(pd.read_csv)("creditcard.csv")
# df = df.sample(frac = 0.5, random_state = 48)

# exploring data
if st.sidebar.checkbox("Show what dataset looks like"):
    st.write(df.head())
    st.write("Shape of the Dataset ", df.shape)
    # st.write("Summary of the Dataset ", df.info())
    st.write("Description of the Dataset ", df.describe())

if st.sidebar.checkbox("Show info of missing values"):
    st.write(df.isna().sum())

fraud = df[df["Class"] == 1]
valid = df[df["Class"] == 0]
fraud_transc_percentage = (len(fraud)/len(df))*100

if st.sidebar.checkbox("Statistics of Valid and Fraud transactions"):
    st.write("Total Transactions: ", len(df))
    st.write("Valid Transactions: ", len(valid))
    st.write("Fraud Transactions: ", len(fraud))
    st.write("Percentage of Fraud transcations: %.3f%%" %fraud_transc_percentage)
    
    vf = df["Class"].value_counts(normalize = True)    
    fig = plt.figure()
    sns.barplot(x = vf.index, y = vf*100).set_title("Percentage of Valid and Fraud Transactions")    
    st.pyplot(fig)
    st.write("The number of fraud transactions are very low compared to valid transactions. We can see our dataset is highly imbalanced.")

if st.sidebar.checkbox("Show Plots"):
    fig1 = plt.figure()
    sns.histplot(df["Amount"], bins = 40).set_title("Distribution of Monetory value feature")
    st.pyplot(fig1)
    st.write("The distribution of the monetary value of all transactions is heavily right-skewed. The vast majority of transactions are relatively small and only a tiny fraction of transactions comes even close to the maximum.")
    
    fig2 = plt.figure()
    sns.histplot(valid["Time"], bins = 40).set_title("Distribution of Valid transactions over Time")
    st.pyplot(fig2)
    
    fig3 = plt.figure()
    sns.histplot(fraud["Time"], bins = 40).set_title("Distribution of Fraud transactions over Time")
    st.pyplot(fig3)
    st.write("There seems to be a decrease in number of transactions around 100000 Time mark. This might be during night. This could be the time that favours the fraudsters when valid transactions are very low.")
    
    fig4 = plt.figure()
    sns.heatmap(df.corr(), cmap = sns.color_palette("coolwarm", as_cmap = True)).set_title("Correlation Heatmap");
    st.pyplot(fig4)
    st.write("There is not much correlation between any of the attributes")

# obtaining X and Y
X = df.drop(["Class"], axis = 1)
Y = df.Class

# splitting data into training and testing sets
from sklearn.model_selection import train_test_split, GridSearchCV
size = st.sidebar.slider("Test set size", min_value = 0.2, max_value = 0.4)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = size, random_state = 42)

# shape of train and test data
if st.sidebar.checkbox("Show shape of train and test data"):
    st.write("Shape of X_train: ", X_train.shape)
    st.write("Shape of Y_train: ", Y_train.shape)
    st.write("Shape of X_test: ", X_test.shape)
    st.write("Shape of Y_test: ", Y_test.shape)

# building model
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

nvbys = GaussianNB()
logreg = LogisticRegression() 
rdmfrst = RandomForestClassifier()

# feature selection through feature importance
def feature_sort_logreg(model, X_train, Y_train):
    # feature selection
    mod = model
    # fit model
    mod.fit(X_train, Y_train)
    imp = mod.coef_[0]
    return imp

def feature_sort(model, X_train, Y_train):
    # feature selection
    mod = model
    # fit model
    mod.fit(X_train, Y_train)
    imp = mod.feature_importances_
    return imp

# classifiers for feature importance
clf = ["Logistic Regression", "Random Forest"]
mod_feature = st.sidebar.selectbox("Choose model for feature Importance", clf)
    
start_time = timeit.default_timer()
if mod_feature == "Logistic Regression":
    model = logreg
    importance = feature_sort_logreg(model, X_train, Y_train)
    
elif mod_feature == "Random Forest":
    model = rdmfrst
    importance = feature_sort(model, X_train, Y_train)

elapsed_time = timeit.default_timer() - start_time
# st.write("Execution time for feature selection: %.2f minutes" %(elapsed_time/60)) 

# plot of feature importance
if st.sidebar.checkbox("Show plot of feature importance"):
    fig = plt.figure()
    sns.barplot(x = [x for x in range(len(importance))], y = importance).set_title("Feature Importance")
    plt.xlabel("Variable Number")
    plt.ylabel("Importance")
    st.pyplot(fig)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def hyperparameterTuning(model, X_train, Y_train, X_test, Y_test):
    params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}
    nbModel_grid = GridSearchCV(estimator=model, 
                                param_grid=params_NB, 
                                verbose=1, 
                                cv=10, 
                                n_jobs=-1)

    nbModel_grid.fit(X_train, Y_train)

def compute_performance(model, X_train, Y_train, X_test, Y_test):
   model.fit(X_train, Y_train)
   Y_pred = model.predict(X_test)
   
   cr = classification_report(Y_test, Y_pred)
   
   accuracy = accuracy_score(Y_test, Y_pred)
   'Accuracy: ', accuracy
   
   "Confusion Matrix: "
   cm = confusion_matrix(Y_test, Y_pred)
   
   group_names = ['True Neg','False Pos','False Neg','True Pos']
   group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]
   group_percentages = ['{0:.2%}'.format(value) for value in
                     cm.flatten()/np.sum(cm)]
   labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
   labels = np.asarray(labels).reshape(2,2)
   
   fig = plt.figure()
   sns.heatmap(cm, annot=labels, fmt="", cmap='Reds')
   st.pyplot(fig)

    
# classification models
if st.sidebar.checkbox("Run a Credit card fraud detection model"):
    alg = ["Naive Bayes"]
    classifier = st.sidebar.selectbox("Select the Algorithm: ", alg)
       
    if classifier == "Naive Bayes":
        model = nvbys
        compute_performance(model, X_train, Y_train, X_test, Y_test)