
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression



st.title("Automatic Data Analyser")

upload_file = st.file_uploader("Upload Your CSV File For Data Analyse", type="csv")

if upload_file is not None:
        st.write("**File Upload Successfully**")
        st.subheader("The Uploaded DataSet is : ")
        df = pd.read_csv(upload_file)
        data=df
        st.dataframe(data)
        st.write("**Data Visualization**")

        st.subheader("Bar Chart")
        column = st.selectbox("Select a column for the Bar Chart", df.columns)
        st.bar_chart(df[column].value_counts())

        st.subheader("Line Chart")
        numeric_column = df.select_dtypes(include=[int, float]).columns
        column = st.selectbox("Select a column for the Line Chart",numeric_column)
        st.line_chart(df[column])

        st.subheader("Histogram")
        column = st.selectbox("Select a column for the Histogram",numeric_column)
        fig, ax = plt.subplots()
        ax.hist(df[column])
        st.pyplot(fig)

        st.subheader("Scatter Plot")
        x_axis = st.selectbox("select x axis for the scatter plot",numeric_column)
        y_axis = st.selectbox("select y axis for the scatter plot",numeric_column)
        fig, ax = plt.subplots()
        ax.scatter(df[x_axis], df[y_axis])
        st.pyplot(fig)

        st.subheader("Description of Your DataSet : ")
        st.write(data.describe())

        model = LinearRegression()
        st.subheader("Use Linear Regression ")

        selected_features = st.multiselect("Select the features", data.columns)
        selected_target = st.selectbox("Select the target column", data.columns)
        filtered_data = data[selected_features + [selected_target]]
        filtered_data.dropna(inplace=True)
        X = filtered_data[selected_features]
        y = filtered_data[selected_target]
        model = LinearRegression()
        model.fit(X, y)
        input_features = []
        for feature in selected_features:
         input_value = st.number_input(f"Enter a value for {feature}:")
        input_features.append(input_value)
        y_pred = model.predict([input_features])
        st.write("Prediction:", y_pred[0])
