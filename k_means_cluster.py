#!/usr/bin/python 

""" 
    skeleton code for k-means clustering mini-project

"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """
    ### plot each cluster with a different color--add more colors for
    ### drawing more than 4 clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()

### load in the dict of dicts containing all the data on each person in the dataset
def Load(infile):
    data_dict = pickle.load( open(infile, "r") )
    data_dict.pop("TOTAL", 0) #Remove Outliers
    return data_dict
def Query(data_dict):
    SalSet=set()
    ESOSet=set()

    for k,v in data_dict.iteritems():
        if v['salary'] != 'NaN':
            SalSet.add(v['salary'])
    for k,v in data_dict.iteritems():
        if v['exercised_stock_options'] != 'NaN':
            ESOSet.add(v['exercised_stock_options'])
    print 'Max Sal',max(SalSet)    
    print 'Min Sal',min(SalSet)  
    print 'Max ESO',max(ESOSet)    
    print 'Min ESO',min(ESOSet) 

def Input_features(data_dict):
    poi  = "poi"
    features_list = [poi, feature_1, feature_2,feature_3]
    data = featureFormat(data_dict, features_list )
    poi, finance_features = targetFeatureSplit(data)
    return poi,finance_features

def Plot_Basic(finance_features):
    ### in the "clustering with 3 features" part of the mini-project,
    ### you'll want to change this line to 
    ### for f1, f2, _ in finance_features:
    for f1, f2, _ in finance_features:
        plt.scatter( f1, f2)
    plt.show()

def Plot_3_Clustoids_BeforeScaling(poi,finance_features):
    clf = KMeans(n_clusters=3)
    pred = clf.fit_predict( finance_features )
    Draw(pred, finance_features, poi, name="clusters_before_scaling.pdf", f1_name=feature_1, f2_name=feature_2)

def Plot_3_Clustoids_AfterScaling(poi,finance_features):
    clf = KMeans(n_clusters=3)
    print finance_features
    pred = clf.fit_predict( finance_features )
    #Draw(pred, finance_features, poi, name="clusters_before_scaling.pdf", f1_name=feature_1, f2_name=feature_2)

#scaler = MinMaxScaler()
#rescaled_weight = scaler.fit_transform(weights)

feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
data_dict=Load("../final_project/final_project_dataset.pkl")

Input_features(data_dict)
Query(data_dict)
poi,finance_features = Input_features(data_dict)
#Plot_Basic(finance_features)
#Plot_3_Clustoids_BeforeScaling(poi,finance_features)
Plot_3_Clustoids_AfterScaling(poi,finance_features)





