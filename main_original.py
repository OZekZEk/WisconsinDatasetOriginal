import pandas as pd
import seaborn as sns
import numpy as np
import time

import graphviz
import csv

from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import tree
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import os

import warnings
warnings.filterwarnings("ignore")

train_sample_rate = 0.8
data_frame = pd.read_csv('breast_w.csv')
random_state = 1
does_oversample_data = False
does_undersample_data = False
does_use_pca = True
does_normalize_data = False
does_standardize_data = False

numpy_random_state = 1
np.random.seed(numpy_random_state) 

pca = PCA(n_components = 2)

def split_data_same_percentage(data_frame):
    data_frame_benign = data_frame[data_frame.iloc[:, -1]=="benign"]
    data_frame_malignant = data_frame[data_frame.iloc[:, -1]=="malignant"]

    data_frame_bening_shuffled = data_frame_benign.sample(frac=1.0, random_state=random_state)
    data_frame_malignant_shuffled = data_frame_malignant.sample(frac=1.0, random_state=random_state)

    last_sample_index_benign = int(len(data_frame_benign)*train_sample_rate)
    last_sample_index_malignant = int(len(data_frame_malignant)*train_sample_rate)

    train_data_frame_benign = data_frame_bening_shuffled[:last_sample_index_benign]
    train_data_frame_malignant = data_frame_malignant_shuffled[:last_sample_index_malignant]
    
    test_data_frame_benign = data_frame_bening_shuffled[last_sample_index_benign:]
    test_data_frame_malignant = data_frame_malignant_shuffled[last_sample_index_malignant:]
    
    train_data_frame = pd.concat([train_data_frame_benign, train_data_frame_malignant])
    test_data_frame = pd.concat([test_data_frame_benign, test_data_frame_malignant])

    return train_data_frame, test_data_frame

train_data_frame, test_data_frame = split_data_same_percentage(data_frame)

def create_csv_output(csvfile, model_name, accuracy, recall, precision, f1, train_time, eval_time):
    # data rows of csv file 
    rows = [["model_name"],
            ["accuracy"], 
            ["recall"], 
            ["precision"], 
            ["f1"], 
            ["train_time (ms)"], 
            ["eval_time (ms)"], 
            ]
    accuracy = np.array(accuracy)
    recall = np.array(recall)
    precision = np.array(precision)
    f1 = np.array(f1)

    rows[0].extend(model_name)
    rows[1].extend((accuracy*1000).astype(int).astype(float)/1000)
    rows[2].extend((recall*1000).astype(int).astype(float)/1000)
    rows[3].extend((precision*1000).astype(int).astype(float)/1000)
    rows[4].extend((f1*1000).astype(int).astype(float)/1000)
    rows[5].extend(train_time)
    rows[6].extend(eval_time)
    # name of csv file 
        
#    writing to csv file 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the data rows 
    csvwriter.writerows(rows)
    
def get_data_label_split(data_frame):
    data = data_frame.iloc[:, 1:-1]
    label = data_frame.iloc[:, -1]
    return data, label

def fill_with_mean(data_frame):
    dropped_data_frame = data_frame.copy()
    dropped_data_frame.iloc[:, 5] = data_frame[data_frame.iloc[:, 5] != "?"].copy() # manuel ayarlandı colum. generic değil. farklı veri türüvya column yeri değişirse bu düzeltme çalışmaz.
    means = dropped_data_frame.mean(skipna = True)
    for mean_idx in range(len(means)):
        data_frame.iloc[:, mean_idx] = data_frame.iloc[:, mean_idx].replace("?", means[mean_idx]) #NAN = mean
    return data_frame

def replace_nan_values(data_frame, data_label, key, column_index, value):
    for ith_key in range(len(data_frame)):
        if data_frame.iloc[ith_key, column_index]=="?" and data_label.iloc[ith_key]==key:
            data_frame.iloc[ith_key, column_index]=value
    return data_frame

def fill_with_class_mean(data_frame, data_label):
    dropped_data_frame = data_frame.copy()
    dropped_data_frame.iloc[:, 5] = data_frame[data_frame.iloc[:, 5] != "?"].copy() # manuel ayarlandı colum. generic değil. farklı veri türüvya column yeri değişirse bu düzeltme çalışmaz.
    benign_mean = dropped_data_frame[data_label=="benign"].mean(skipna = True)
    malignant_mean = dropped_data_frame[data_label=="malignant"].mean(skipna = True)
    means = dropped_data_frame.mean(skipna = True)
    for mean_idx in range(len(means)):
        data_frame = replace_nan_values(data_frame, data_label, "benign", mean_idx, benign_mean[mean_idx])
        data_frame = replace_nan_values(data_frame, data_label, "malignant", mean_idx, malignant_mean[mean_idx])
    return data_frame

def fill_dataframe(data_frame, data_label, value):
    if value=="mean":
        data_frame = fill_with_mean(data_frame)
    elif value=="class_mean":
        data_frame = fill_with_class_mean(data_frame, data_label)
    else:
        data_frame = data_frame.replace("?", value) #NAN = 0,1
    return data_frame

def fill_train_test_split(train_data, train_label, test_data, test_label, value):
    train_data = fill_dataframe(train_data, train_label, value).copy()
    test_data = fill_dataframe(test_data, test_label, value).copy()
    if does_oversample_data:
        train_data_resampled, train_label_resampled = oversample_data(train_data, train_label)
        test_data_resampled, test_label_resampled = test_data, test_label
    elif does_undersample_data:
        train_data_resampled, train_label_resampled = undersample_data(train_data, train_label)
        test_data_resampled, test_label_resampled = test_data, test_label
    else:
        train_data_resampled, train_label_resampled = train_data, train_label
        test_data_resampled, test_label_resampled = test_data, test_label
    return train_data_resampled, train_label_resampled, test_data_resampled, test_label_resampled

def undersample_data(data_frame, data_label):
    ros = RandomUnderSampler(random_state=1)
    X_resampled, y_resampled = ros.fit_resample(data_frame, data_label)
    return X_resampled, y_resampled

def oversample_data(data_frame, data_label):
    ros = RandomOverSampler(random_state=1)
    X_resampled, y_resampled = ros.fit_resample(data_frame, data_label)
    return X_resampled, y_resampled

def train_model(model, train_data, train_label):
    model.fit(train_data, train_label)
    train_end_time = time.time()
    return model, train_end_time

def predict_model(model, test_data, test_label, model_name):
    predictions = model.predict(test_data)
    correct_predictions = np.sum((predictions==test_label))

    f1_score_result = f1_score(test_label, predictions, labels=['benign', 'malignant'], pos_label='malignant')
    recall_score_result = recall_score(test_label, predictions, labels=['benign', 'malignant'], pos_label='malignant')
    precision_score_result = precision_score(test_label, predictions, labels=['benign', 'malignant'], pos_label='malignant')
    confusion_matrix_result = confusion_matrix(test_label, predictions)
    eval_end_time = time.time()
    if does_use_pca:
        label_color_dict = {"benign":[1,0,0], "malignant":[0,0,1]}
        x_test = pca.fit_transform(test_data)
        cvec = [label_color_dict[label] for label in predictions]
        plt.figure(figsize=(8,8))
        plt.scatter(x_test[:,0], x_test[:,1],
                c=cvec, edgecolor=['none'], alpha=0.5)
        os.makedirs("pca_out", exist_ok=True)
        plt.savefig(os.path.join("pca_out", model_name+".png"))
    return correct_predictions/len(test_label), recall_score_result, precision_score_result, f1_score_result, confusion_matrix_result, eval_end_time

train_data, train_label = get_data_label_split(train_data_frame)
test_data, test_label = get_data_label_split(test_data_frame)

#p=1: Manhattan, p=2: Euclidian
#weight="uniform": Weighted equally, weight="distance": Inverse of their distance

# models = [  KNeighborsClassifier(n_neighbors=2, weights="uniform", p=1), KNeighborsClassifier(n_neighbors=2, weights="distance", p=1), 
#             KNeighborsClassifier(n_neighbors=2, weights="uniform", p=2), KNeighborsClassifier(n_neighbors=2, weights="distance", p=2),
            
#             KNeighborsClassifier(n_neighbors=3, weights="uniform", p=1), KNeighborsClassifier(n_neighbors=3, weights="distance", p=1), 
#             KNeighborsClassifier(n_neighbors=3, weights="uniform", p=2), KNeighborsClassifier(n_neighbors=3, weights="distance", p=2),

#             KNeighborsClassifier(n_neighbors=5, weights="uniform", p=1), KNeighborsClassifier(n_neighbors=5, weights="distance", p=1), 
#             KNeighborsClassifier(n_neighbors=5, weights="uniform", p=2), KNeighborsClassifier(n_neighbors=5, weights="distance", p=2)
#         ]
#models = [svm.SVC(C=1, kernel="linear"),svm.SVC(C=3, kernel="linear"),svm.SVC(C=5, kernel="linear"),
#            svm.SVC(C=1, kernel="sigmoid"), svm.SVC(C=3, kernel="sigmoid"), svm.SVC(C=5, kernel="sigmoid"),
#            svm.SVC(C=1, kernel="rbf"), svm.SVC(C=3, kernel="rbf"), svm.SVC(C=5, kernel="rbf")]

# models = GaussianNB()

# models = [RandomForestClassifier()]

# models = [tree.DecisionTreeClassifier()]

models = [  
            KNeighborsClassifier(n_neighbors=2, weights="uniform", p=1), KNeighborsClassifier(n_neighbors=2, weights="distance", p=1), 
            KNeighborsClassifier(n_neighbors=2, weights="uniform", p=2), KNeighborsClassifier(n_neighbors=2, weights="distance", p=2),            
            KNeighborsClassifier(n_neighbors=3, weights="uniform", p=1), KNeighborsClassifier(n_neighbors=3, weights="distance", p=1), 
            KNeighborsClassifier(n_neighbors=3, weights="uniform", p=2), KNeighborsClassifier(n_neighbors=3, weights="distance", p=2),
            KNeighborsClassifier(n_neighbors=5, weights="uniform", p=1), KNeighborsClassifier(n_neighbors=5, weights="distance", p=1), 
            KNeighborsClassifier(n_neighbors=5, weights="uniform", p=2), KNeighborsClassifier(n_neighbors=5, weights="distance", p=2),
            KNeighborsClassifier(n_neighbors=10, weights="uniform", p=1), KNeighborsClassifier(n_neighbors=10, weights="distance", p=1), 
            KNeighborsClassifier(n_neighbors=10, weights="uniform", p=2), KNeighborsClassifier(n_neighbors=10, weights="distance", p=2),

            svm.SVC(C=1, kernel="linear"),svm.SVC(C=3, kernel="linear"),svm.SVC(C=5, kernel="linear"),
            svm.SVC(C=1, kernel="sigmoid"), svm.SVC(C=3, kernel="sigmoid"), svm.SVC(C=5, kernel="sigmoid"),
            svm.SVC(C=1, kernel="rbf"), svm.SVC(C=3, kernel="rbf"), svm.SVC(C=5, kernel="rbf"),

            GaussianNB(),

            RandomForestClassifier(),

            tree.DecisionTreeClassifier(max_features="auto"),
            tree.DecisionTreeClassifier(max_features="sqrt"),
            tree.DecisionTreeClassifier(max_features="log2"),
            tree.DecisionTreeClassifier(min_samples_split=20),
            tree.DecisionTreeClassifier(min_samples_split=10),
            tree.DecisionTreeClassifier(min_samples_split=5),
            tree.DecisionTreeClassifier(min_samples_split=2),
            tree.DecisionTreeClassifier(),
            ]

fill_values = [0, 1, "mean", "class_mean"]

def export_as_graph(model):
    dot_data = tree.export_graphviz(model, out_file=None) 
    graph = graphviz.Source(dot_data) 
    graph.render("iris_"+str(model)+"_%i" % random_state )

confusion_matrices = []
confusion_matrices_model_names = []
confusion_matrices_accuracies = []
confusion_matrices_titles = []

def arrange_parameters(models, fill_values, train_data, train_label, test_data, test_label):
    recall_values = []
    precision_values = []
    f1_values = []
    train_times = []
    eval_times = []
    for model in models:
        print("")
        for fill_value in fill_values:
            filled_train_data, filled_train_label, filled_test_data, filled_test_label = fill_train_test_split(train_data, train_label, test_data, test_label, fill_value)
            if does_normalize_data:
                filled_train_data = preprocessing.normalize(filled_train_data)
                filled_test_data = preprocessing.normalize(filled_test_data)
            elif does_standardize_data:
                scaler = preprocessing.StandardScaler().fit(filled_train_data)
                filled_train_data = scaler.transform(filled_train_data)
                filled_test_data = scaler.transform(filled_test_data)
            train_start_time = time.time()
            trained_model, train_end_time = train_model(model, filled_train_data, filled_train_label)
            eval_start_time = time.time()
            model_name = (str(model) +"_"+ str(fill_value))
            accuracy_value, recall_value, precision_value, f1_value, confusion_matrix_result, eval_end_time = predict_model(trained_model, filled_test_data, filled_test_label, model_name)
            print("Results for model %s with fill values %s are: %s accuracy | %s recall_value | %s precision_value | %s f1_value | %s train_time | %s eval_time" % (str(model), str(fill_value), str(accuracy_value), str(recall_value),
                                                                                                                                    str(precision_value), str(f1_value), 
                                                                                                                                    str(train_end_time-train_start_time), 
                                                                                                                                    str(eval_end_time-eval_start_time)))
            recall_values.append(recall_value)
            precision_values.append(precision_value)
            f1_values.append(f1_value)
            train_times.append((train_end_time-train_start_time)*1000)
            eval_times.append((eval_end_time-eval_start_time)*1000)

            confusion_matrices.append(confusion_matrix_result.ravel())
            confusion_matrices_model_names.append(str(model)+"_filled_"+str(fill_value))
            confusion_matrices_accuracies.append(accuracy_value)
            confusion_matrices_titles.append("Results for model %s with fill values %s"  % (str(model), str(fill_value)))
            if False: # Make if True when we are using decision tree classifier so we can see the decision tree as output
                export_as_graph(model)
        print("")
    filename = "output_metrics_%i.csv" % random_state
    with open(filename, 'w', newline='') as csvfile: 
        create_csv_output(csvfile, confusion_matrices_model_names, confusion_matrices_accuracies, recall_values, precision_values, f1_values, train_times, eval_times)

def sort_list(list1, list2, n_vals):
    list2 = np.array(list2)
    indices = np.argsort(list1)[::-1]
    sorted_list = list2[indices[:n_vals]]
    return sorted_list

def visualize_matrices(model_names, confusion_values, accuracy_values, titles, n_matrices):
    n_matrices = min(n_matrices, len(model_names))
    sorted_model_names = sort_list(accuracy_values, model_names, n_matrices)
    sorted_confusion_values = sort_list(accuracy_values, confusion_values, n_matrices)
    sorted_titles = sort_list(accuracy_values, titles, n_matrices)
    for ith_matrix in range(n_matrices):
        heatmap = sorted_confusion_values[ith_matrix].reshape((2, 2))
#        text = np.array([["Ground truth benign", "Ground truth malignant"], ["Predicted benign", "Predicted malignant"]])
        heatmap_figure = sns.heatmap(heatmap, annot=True, cbar=False, xticklabels=["Benign", "Malignant"],
                                     yticklabels=["Benign", "Malignant"])
        heatmap_figure.set(title=sorted_titles[ith_matrix])
        
        heatmap_figure.set_xlabel('Prediction'); 
        heatmap_figure.set_ylabel('Ground truth')
        # heatmap_figure.xaxis.tick_top()
        # heatmap_figure.xaxis.set_label_position('top') 

        figure = heatmap_figure.get_figure()
        figure.savefig(str(ith_matrix+1)+"_"+str(sorted_model_names[ith_matrix])+'_conf_matrix.png', dpi=400)
        figure.clf()

arrange_parameters(models, fill_values, train_data, train_label, test_data, test_label)
visualize_matrices(confusion_matrices_model_names, confusion_matrices, confusion_matrices_accuracies, confusion_matrices_titles, 20)
