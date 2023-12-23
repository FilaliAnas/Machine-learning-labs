# Lab12: Classification des fleurs iris en utilisant scikit-learn
# Realise par : Anas FILALI EMSI 2023/2024
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import streamlit as st
import pandas as pd

# Set Streamlit page title and icon
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon=":blossom:",
)

# Step 1: DataSet
iris = datasets.load_iris()
print(iris.data)
print(iris.target)
print(iris.target_names)
print(iris.feature_names)
print(iris.data.shape)
# Step 2: Model
models = {
    'RandomForest': RandomForestClassifier(),
    'DecisionTree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC()
}
model = RandomForestClassifier()
# Step 3: Train
model.fit(iris.data, iris.target)
# Step 4: Test
prediction = model.predict([[0.9,1.0 ,1.1,1.8]])
print(prediction)
print(iris.target_names[prediction])
# Web Deployment of the Model: streamlit run filename.py
st.title("Iris Flower Classification")
st.markdown("This app allows you to classify Iris flowers using different algorithms.")
st.header('Iris flowers classification')
st.image("Images/iris.jpg", caption="Iris Types Caption")
st.sidebar.markdown("### Iris Features")
st.sidebar.markdown("Adjust the sliders to set the sepal and petal measurements for classification.")

def user_input():
    sepal_length = st.sidebar.slider('sepal length',0.1,9.9,5.0)
    sepal_width = st.sidebar.slider('sepal width',0.1,9.9,5.0)
    petal_length = st.sidebar.slider('petal length',0.1,9.9,5.0)
    petal_width = st.sidebar.slider('petal width',0.1,9.9,5.0)
    data = {
        'sepal_length': sepal_length,
        'sepal_width' : sepal_width,
        'petal_length' : petal_length,
        'petal_width' : petal_width
        }
    flower_features = pd.DataFrame(data,index=[0])
    return flower_features

df = user_input()
st.write(df)
selected_model = st.sidebar.selectbox('Select your learning algorithm ',['RandomForest','DecisionTree','KNN','SVN'])
st.write('selected algorithm is :', selected_model)
model = models[selected_model]
model.fit(iris.data, iris.target)
st.subheader('Prediction')
prediction = model.predict(df)
st.write(prediction)
st.write(iris.target_names[prediction])
st.image('Images/'+iris.target_names[prediction][0]+'.jpg', caption="Iris Image Caption")

