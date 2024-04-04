import streamlit as st
import numpy as np
import pickle
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

Count=0
df=pd.read_csv(r"Crop_recommendation.csv")
df=df.drop(columns=['Unnamed: 8', 'Unnamed: 9'])
X = df[['Nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph','rainfall']][:]
y = df[['label']][:]
label=np.unique(y)
for j in range(len(label)):
    y=y.replace(label[j],j)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.set_page_config(
        page_title="Crop Desirability Checker",
        page_icon="🌿",
        layout="centered",
        initial_sidebar_state="expanded",
)

RF_model=open(r"Models\RF.pkl","rb")
RF = pickle.load(RF_model)

LR_model=open(r"Models\LR.pkl","rb")
LR = pickle.load(LR_model)

SVM_model=open(r"Models\SVM.pkl","rb")
SVM = pickle.load(SVM_model)

NB_model=open(r"Models\NB.pkl","rb")
NB = pickle.load(NB_model)

DT_model=open(r"Models\DT.pkl","rb")
DT = pickle.load(DT_model)    

#test fuction
def check_desirable_crop(n, p, k, temperature, humidity, ph, rainfall):
    values=np.array([[n, p, k, temperature, humidity, ph, rainfall]])
    RF_result=RF.predict(values)
    LR_result=LR.predict(values)
    SVM_result=SVM.predict(values)
    NB_result=NB.predict(values)
    DT_result=DT.predict(values)
    crop_prediction={
        "RF":RF_result[0],
        "LR":LR_result[0],
        "SVM":SVM_result[0],
        "NB":NB_result[0],
        "DT":DT_result[0]
    }
    counted=Counter(crop_prediction.values())
    result=counted.most_common(1)[0][0]
    return crop_prediction,result

def classification_report_to_df(report):
    data = {'Class': list(report.keys())[:-1],  # Exclude 'accuracy' from keys
            'Precision': [report[key]['precision'] for key in report.keys()[:-1]],
            'Recall': [report[key]['recall'] for key in report.keys()[:-1]],
            'F1-Score': [report[key]['f1-score'] for key in report.keys()[:-1]]}
    return pd.DataFrame(data)

def conf_mat(conf):
    num_classes = conf.shape[0]
    tp_list = []
    fn_list = []
    fp_list = []
    tn_list = []
    for i in range(num_classes):
        # True Positive (TP)
        tp = conf[i, i]
        tp_list.append(tp)

        # False Negative (FN)
        fn = np.sum(conf[i, :]) - tp
        fn_list.append(fn)

        # False Positive (FP)
        fp = np.sum(conf[:, i]) - tp
        fp_list.append(fp)

        # True Negative (TN)
        tn = np.sum(conf) - tp - fp - fn
        tn_list.append(tn)
    return tp_list, fn_list, fp_list, tp_list 

def display_confusion_matrix_pie_chart(tp_list, fn_list, fp_list, tn_list):
    global Count
    # Combine lists into a single confusion matrix
    num_classes = len(tp_list)
    confusion_matrix = np.zeros((num_classes, 4), dtype=int)
    confusion_matrix[:, 0] = tp_list
    confusion_matrix[:, 1] = fn_list
    confusion_matrix[:, 2] = fp_list
    confusion_matrix[:, 3] = tn_list

    # Get the list of class labels
    class_labels = [f'Class {namelabel[i]}' for i in range(num_classes)]

    # Dropdown to select the class
    selected_class = st.selectbox('Select a class:', class_labels, key = Count)
    Count += 1

    # Get the index of the selected class
    class_idx = class_labels.index(selected_class)

    # Get the confusion matrix values for the selected class
    tp = confusion_matrix[class_idx, 0]
    fn = confusion_matrix[class_idx, 1]
    fp = confusion_matrix[class_idx, 2]
    tn = confusion_matrix[class_idx, 3]

    # Create a pie chart
    labels = ['True Positive', 'False Negative', 'False Positive', 'True Negative']
    values = [tp, fn, fp, tn]
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title(f'Confusion Matrix for {selected_class}')

    st.pyplot(fig)

def visualize_classification_report(report):
    # Convert the report dictionary to a DataFrame
    df = pd.DataFrame(report).transpose()
    
    # Drop the 'support' row as it's not relevant for visualization
    df.drop(columns='support', inplace=True)
    
    # Plotting
    st.bar_chart(df)

def most_frequent_column_elements(matrix):
    
    most_frequent_elements = []
    

    for col in matrix.T:

        unique_elements, counts = np.unique(col, return_counts=True)
        

        max_count_index = np.argmax(counts)
        

        most_frequent_elements.append(unique_elements[max_count_index])
    

    return np.array(most_frequent_elements)

namelabel=np.array(['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee','cotton',
     'grapes', 'jute', 'kidneybeans', 'lentil', 'maize','mango', 'mothbeans', 'mungbean', 
     'muskmelon', 'orange', 'papaya',
       'pigeonpeas', 'pomegranate', 'rice', 'watermelon'])

predictions_RF = RF.predict(X_test)
predictions_DT = DT.predict(X_test)
predictions_SVM = SVM.predict(X_test)
predictions_NB = NB.predict(X_test)
predictions_LR = LR.predict(X_test)
report_DT = classification_report(y_test, predictions_DT, output_dict=True)
report_LR = classification_report(y_test, predictions_LR, output_dict=True)
report_RF = classification_report(y_test, predictions_RF, output_dict=True)
report_NB = classification_report(y_test, predictions_NB, output_dict=True)
report_SVM = classification_report(y_test, predictions_SVM, output_dict=True)
conf_DT = confusion_matrix(y_test, predictions_DT)
conf_RF = confusion_matrix(y_test, predictions_RF)
conf_SVM = confusion_matrix(y_test, predictions_SVM)
conf_LR = confusion_matrix(y_test, predictions_LR)
conf_NB = confusion_matrix(y_test, predictions_NB)

best_of_all=np.vstack([predictions_SVM,predictions_RF,predictions_DT,predictions_LR,predictions_DT,predictions_NB])
combined_output=most_frequent_column_elements(best_of_all)

conf_CO = confusion_matrix(y_test, combined_output)
report_CO = classification_report(y_test, combined_output,output_dict=True)

st.title("Crop Desirability Checker")

RF_tab,LR_tab,SVM_tab,NB_tab,DT_tab,CO_tab = st.tabs(["Random Forests","Logistic Regression","SVM","Naive Bayes","Decision Trees","Combined Output"])

with RF_tab:
    visualize_classification_report(report_RF)
    tp_list,fn_list,fp_list,tn_list=conf_mat(conf_RF)
    df2={
        "True Positive":np.sum(tp_list),
        "True Negative":np.sum(tn_list),
        "False Positive":np.sum(fp_list),
        "False Negative":np.sum(fn_list)
    }
    st.write(pd.DataFrame(df2,index=[0]).transpose())
    display_confusion_matrix_pie_chart(tp_list,fn_list,fp_list,tn_list)
with LR_tab:
    visualize_classification_report(report_LR)
    tp_list,fn_list,fp_list,tn_list=conf_mat(conf_LR)
    df2={
        "True Positive":np.sum(tp_list),
        "True Negative":np.sum(tn_list),
        "False Positive":np.sum(fp_list),
        "False Negative":np.sum(fn_list)
    }
    st.write(pd.DataFrame(df2,index=[0]).transpose())
    display_confusion_matrix_pie_chart(tp_list,fn_list,fp_list,tn_list)
with SVM_tab:
    visualize_classification_report(report_SVM)
    tp_list,fn_list,fp_list,tn_list=conf_mat(conf_SVM)
    df2={
        "True Positive":np.sum(tp_list),
        "True Negative":np.sum(tn_list),
        "False Positive":np.sum(fp_list),
        "False Negative":np.sum(fn_list)
    }
    st.write(pd.DataFrame(df2,index=[0]).transpose())
    display_confusion_matrix_pie_chart(tp_list,fn_list,fp_list,tn_list)
with NB_tab:
    visualize_classification_report(report_NB)
    tp_list,fn_list,fp_list,tn_list=conf_mat(conf_NB)
    df2={
        "True Positive":np.sum(tp_list),
        "True Negative":np.sum(tn_list),
        "False Positive":np.sum(fp_list),
        "False Negative":np.sum(fn_list)
    }
    st.write(pd.DataFrame(df2,index=[0]).transpose())
    display_confusion_matrix_pie_chart(tp_list,fn_list,fp_list,tn_list)
with DT_tab:
    visualize_classification_report(report_DT)
    tp_list,fn_list,fp_list,tn_list=conf_mat(conf_DT)
    df2={
        "True Positive":np.sum(tp_list),
        "True Negative":np.sum(tn_list),
        "False Positive":np.sum(fp_list),
        "False Negative":np.sum(fn_list)
    }
    st.write(pd.DataFrame(df2,index=[0]).transpose())
    display_confusion_matrix_pie_chart(tp_list,fn_list,fp_list,tn_list)

with CO_tab:
    visualize_classification_report(report_CO)
    tp_list,fn_list,fp_list,tn_list=conf_mat(conf_CO)
    df2={
        "True Positive":np.sum(tp_list),
        "True Negative":np.sum(tn_list),
        "False Positive":np.sum(fp_list),
        "False Negative":np.sum(fn_list)
    }
    st.write(pd.DataFrame(df2,index=[0]).transpose())
    display_confusion_matrix_pie_chart(tp_list,fn_list,fp_list,tn_list)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Input columns
st.sidebar.header("Input Parameters")
n = st.sidebar.text_input("N (Nitrogen)")
p = st.sidebar.text_input("P (Phosphorus)")
k = st.sidebar.text_input("K (Potassium)")
temperature = st.sidebar.text_input("temperature")
humidity = st.sidebar.text_input("humidity")
ph = st.sidebar.text_input("pH")
rainfall = st.sidebar.text_input("Rainfall (mm)")

# enter your output here
if st.sidebar.button("Check Desirable Crop"):
    crop_prediction,result = check_desirable_crop(float(n), float(p), float(k), float(temperature), float(humidity), float(ph), float(rainfall))
    st.success(f"The combined desirable crop is: {namelabel[result]}")

