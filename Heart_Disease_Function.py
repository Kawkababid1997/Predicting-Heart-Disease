#!/usr/bin/env python
# coding: utf-8

# ## Libraries

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve


# ## Function

# In[1]:


# Annotated Bar Graphs
def plot_bar_graph(x, y, xlabel, ylabel, title):
    """
    Annotated Bar graphs
    x = data on the x axis of the bar
    y = data on the y axis of the bar
    xlabel = Label the x-axis
    ylabel = Label the y-axis
    title = Title the graph
    """
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    sns.barplot(x, y)
    names = sns.barplot(x, y)
    for p in names.patches: 
        names.annotate(format(p.get_height(), '.0f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()/2), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9),
                   textcoords = 'offset points')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    
# Create a bar graph from a cross-tab
def cross_tab_bar_graph(column1, column2,  title, xlabel, ylabel):
    """
    Returns croostab bar graphs:
    
    Parameters
    ----------
    columns1, columns2 : array-like, Series, or list of arrays/Series
    Values to group by in the columns.
    title: Title the graph
    xlabel: Label the x-axis
    ylabel: Label the y-axis
    """
    pd.crosstab(column1, column2).plot(kind="bar",color = ["green", "blue"], figsize=(10,8))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation = 0);
    
# Create a scatter graph
def plot_scatter(column1, column2, column3, column4, title, xlabel, ylabel):
    """
    Returns scatter plot:
    
    Parameters
    ----------
    columns1, columns2, column3, columns4 : array-like, Series, or list of arrays/Series
    Values to group by in the columns.
    title: Title the graph
    xlabel: Label the x-axis
    ylabel: Label the y-axis
    """
    # Create a figure
    plt.figure(figsize=(10,6))
    plt.scatter(column1, column2, c = "red"),
    plt.scatter(column3, column4, c = "blue")
    # Add some helful information
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(["Disease", "No Disease"])
    
# Create a heatmap with corrolation matrix
def corrolation_matrix(data):
    """
    Returns scatter plot:
    
    Parameters
    ----------
    
    data: Pass the dataframe
    """
    fig, ax = plt.subplots(figsize=(15,10))
    ax = sns.heatmap(data, 
                 annot = True,
                 linewidth = 0.5,
                 fmt = ".2f", cmap="YlGnBu");
    
# Create a function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    """
    Fits and evaluates given machine learning models.
    
    Parameters
    ----------
    
    models : a dictionary of different Scikit-Learn machine learning models
    X_train : training data (no labels)
    X_test : testing data (no labels)
    y_train : training labels
    y_test : test labels
    """
    # Set random seed
    np.random.seed(42)
    # Make a dictionary to keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(X_train, y_train)
        # Evaluate the model and append its score to model_scores
        model_scores[name] = model.score(X_test, y_test)
    return model_scores

def model_comparison_score(df, ylabel, title):
    """
    A bar graph illustrating scores from a differnt model
    
    Parameters
    ----------
    
    df: Scores dataframe
    
    ylabel
    
    title
    
    """
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    dataFrame = sns.barplot(data=df)
    for p in dataFrame.patches:
        dataFrame.annotate(format(p.get_height(), '.4f'), 
                       (p.get_x() + p.get_width() / 2., p.get_height()/2), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points')
    plt.ylabel(ylabel, size=14)
    plt.title(title);

def knn_tuning(lowest_value, highest_value, X_train, y_train, X_test, y_test, xlabel, ylabel):
    """
    Tune the KNN model between two values
    Parameters
    ----------
    lowest_value, highest_value, X_train, y_train, X_test, y_test, xlabel, ylabel
    
    """
    ## Let's tune KNN
    train_scores = []
    test_scores = []
    ## Create a list of different value for n_neighbors
    neighbors = range(lowest_value, highest_value)
    ## Set up KNN instance
    knn = KNeighborsClassifier()
    ## Loop through different n_neighbors
    for i in neighbors:
        knn.set_params(n_neighbors = i)
        # Fit the algorithm
        knn.fit(X_train, y_train)
        #Update the training scores list
        train_scores.append(knn.score(X_train, y_train))
        # Update the test scores list
        test_scores.append(knn.score(X_test, y_test))       
    # Plot the Score 
    plt.plot(neighbors, train_scores, label = "Train Score")
    plt.plot(neighbors, test_scores, label = "Test Score")
    plt.xticks(np.arange(lowest_value, highest_value,1))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    print(f"Maximum KNN score on the test data : {max(test_scores)*100: .2f}%")    
    

# Visualization of Confusion Matrix
def plot_conf_mat(y_test, y_pred):
    """
    Plots a nice looking confusion matrix using Seaborn's heatmap()
    
    Parameters
    ----------
    ytest, ypred
    
    """
    sns.set(font_scale= 1.5)
    fig, ax = plt.subplots(figsize=(3,3))
    ax = sns.heatmap(confusion_matrix(y_test, y_pred),
                    annot=True,
                    cbar= False)
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")


# In[ ]:




