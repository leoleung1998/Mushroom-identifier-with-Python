# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 21:31:12 2020

@author: leohl
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix,plot_roc_curve,plot_precision_recall_curve
from sklearn.metrics import precision_score,recall_score
def main():
    st.title("Binary Classification")
    st.sidebar.title("Binary Classification")
    st.markdown("Are your Mushroom edible?🍄🤮")
    st.sidebar.markdown("Are your Mushroom edible?🍄🤮")
    st.sidebar.subheader("Choose your model")
    classifier=st.sidebar.selectbox("Classifier",("SVM","Random Forest","Logistic Regression"))
    @st.cache(persist=True) ## won't run the function and return the previous output if nth change
    def load_data():
        data=pd.read_csv("mushrooms.csv")
        label=LabelEncoder()
        for col in data.columns:
            data[col]=label.fit_transform(data[col]) ##encode each col
        return data
    df=load_data()
    if st.sidebar.checkbox("Show data",False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)
    @st.cache(persist=True)
    def split(df):
        y=df["class"]
        x=df.drop(columns=["class"])
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
        return x_train,x_test,y_train,y_test
    def plot_metrics(metrics_list):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model,x_test,y_test,display_labels=class_names)
            st.pyplot()
        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model,x_test,y_test)
            st.pyplot()
        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model,x_test,y_test)
            st.pyplot()

    class_names=["edible","Poison"]
    x_train,x_test,y_train,y_test=split(df)
    if classifier=="SVM":
        st.sidebar.subheader("Model Hyperparameter")
        C=st.sidebar.number_input("C (Regularisation Parameter)",0.01,10.0,step=0.01,key="C")
        kernel=st.sidebar.radio("Kernel",("rbf","linear"),key="kernel")
        gamma=st.sidebar.radio("Gamma (Kernel Coefficient)",("scale","auto"),key="gamma")
        metrics=st.sidebar.multiselect("What metrics to plot?",("Confusion Matrix","ROC Curve","Precision-Recall Curve"))
        if st.sidebar.button("Classify",key="Classify"):
            st.subheader("SVM Result")
            model=SVC(C,kernel)
            model.fit(x_train,y_train)
            accuracy=model.score(x_test,y_test)
            y_pred=model.predict(x_test)
            st.write("Accuracy",accuracy)
            st.write("Precision",precision_score(y_test,y_pred,labels=class_names).round(2))
            st.write("Recall",recall_score(y_test,y_pred,labels=class_names).round(2))
            plot_metrics(metrics)
    if classifier=="Logistic Regression":
        st.sidebar.subheader("Model hyperparameter")
        C=st.sidebar.number_input("C (Regularisation Parameter)",0.01,10.0,step=0.01,key="C_LR")
        max_iter=st.sidebar.slider("maximum number of Iterations",100,500,key="mi")
        metrics=st.sidebar.multiselect("What metrics to plot?",("Confusion Matrix","ROC Curve","Precision-Recall Curve"))
        if st.sidebar.button("Classify",key="Classify"):
            st.subheader("Logistic Regression Result")
            model=LogisticRegression(C=C,max_iter=max_iter)
            model.fit(x_train,y_train)
            accuracy=model.score(x_test,y_test)
            y_pred=model.predict(x_test)
            st.write("Accuracy",accuracy)
            st.write("Precision",precision_score(y_test,y_pred,labels=class_names).round(2))
            st.write("Recall",recall_score(y_test,y_pred,labels=class_names).round(2))
            plot_metrics(metrics)
    if classifier=="Random Forest":
        st.sidebar.subheader("Model hyperparameter")
        n=st.sidebar.number_input("number of trees in the forest",100,5000,step=10,key="number_tree")
        max_depth=st.sidebar.number_input("maximum Depth",1,100,key="md")
        bootstrap=st.sidebar.radio("Bootstrap",("True","False"))
        metrics=st.sidebar.multiselect("What metrics to plot?",("Confusion Matrix","ROC Curve","Precision-Recall Curve"))
        if st.sidebar.button("Classify",key="Classify"):
            st.subheader("Random Forest Result")
            model=RandomForestClassifier(n_estimators=n,max_depth=max_depth,bootstrap=bootstrap)
            model.fit(x_train,y_train)
            accuracy=model.score(x_test,y_test)
            y_pred=model.predict(x_test)
            st.write("Accuracy",accuracy)
            st.write("Precision",precision_score(y_test,y_pred,labels=class_names).round(2))
            st.write("Recall",recall_score(y_test,y_pred,labels=class_names).round(2))
            plot_metrics(metrics)
        
if __name__ == "__main__":
    main()