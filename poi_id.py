#!/usr/bin/python
import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from time import time


def QueryDataSet(data_dict):
	print 'Total Number of Data Points:',len(data_dict)
	print 'Number POIs:',sum(1 for v in data_dict.values() if v['poi']==True)
	print 'Number non-POIs:',sum(1 for v in data_dict.values() if v['poi']==False)
	keys = next(data_dict.itervalues()).keys()
	print 'Number of Features:', len(keys)
	FeatWNaN=dict.fromkeys(keys,0)
	#Count the number of Missing values
	for k,v in data_dict.iteritems():
		for i in v:
			if v[i] == 'NaN':
				FeatWNaN[i]+=1
	df = pd.DataFrame.from_dict(FeatWNaN, orient='index')
	df = df.rename(columns = {0: 'Missing Vals'})
	df['Existing Values'] = len(data_dict)-df['Missing Vals']
	print df.sort('Missing Vals',ascending=0)

def PlotData(target,features,Title):
    data_color = "b"
    line_color = "r"
    clf=lm.LinearRegression()
    clf.fit(features, target)
    for feature, target in zip(features, target):
        plt.scatter( feature, target, color=data_color )
    plt.plot( features, clf.predict(features),color=line_color )
    plt.xlabel('salary')
    plt.ylabel('bonus')
    plt.title(Title)
    plt.show()

def DrawClusters(pred, features, poi, Title,name="image.png", f1_name="feature 1", f2_name="feature 2",):
    """ some plotting code designed to help you visualize your clusters """
    ### plot each cluster with a different color--add more colors for
    ### drawing more than 4 clusters
    ### place red stars over points that are POIs 
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])
        if poi[ii]:
            plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.title(Title)
    plt.savefig(name)
    plt.show()

def Plot_3_Clustoids_BeforeScaling(poi,finance_features):
    clust = KMeans(n_clusters=3)
    #print finance_features
    pred = clust.fit_predict(finance_features)
    DrawClusters(pred, finance_features, poi,'Clusters Before Scaling', name="clusters_before_scaling.pdf", f1_name='salary', f2_name='exercised_stock_options')

def Plot_3_Clustoids_AfterScaling(poi,finance_features):
    scaler = MinMaxScaler()
    rescaled_features = scaler.fit_transform(finance_features)
    clust = KMeans(n_clusters=3)
    #print finance_features
    pred = clust.fit_predict(rescaled_features)
    DrawClusters(pred, rescaled_features, poi,'Clusters After Scaling', name="clusters_after_scaling.pdf", f1_name='salary', f2_name='exercised_stock_options')

### Load the dictionary containing the dataset
### Task 1: Select what features you'll use.
data_dict = pickle.load(open("final_project_dataset.pkl", "r"))

### Task 2: Remove outliers
def PlotReg(Title):
    RegFeatures = ["salary", "bonus"]
    data = featureFormat( data_dict, RegFeatures, remove_any_zeroes=True)
    target, features = targetFeatureSplit(data)
    PlotData(target,features,Title)

def RmOutliers(data_dict):
    data_dict.pop('TOTAL')   
    return data_dict    

#PlotReg('With Outlier(s)')
data_dict=RmOutliers(data_dict)
#PlotReg('Without Outlier(s)')

### Task 3: Create new feature(s)

def computeFraction( poi_messages, all_messages ):
    if poi_messages=='NaN' or all_messages == 'NaN':
        fraction = 0
    else:
        poi_messages=float(poi_messages)
        all_messages=float(all_messages)
        fraction = poi_messages/all_messages
    return fraction


def AddFeatures(data_dict):
    for name in data_dict:       
        from_poi_to_this_person = data_dict[name]['from_poi_to_this_person']
        to_messages = data_dict[name]['to_messages']
        fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages)
        data_dict[name]["fraction_from_poi"]=fraction_from_poi
        from_this_person_to_poi = data_dict[name]['from_this_person_to_poi']
        from_messages = data_dict[name]["from_messages"]
        fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
        data_dict[name]["fraction_to_poi"] = fraction_to_poi
    return data_dict

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
data_dict=AddFeatures(data_dict)
keys = next(data_dict.itervalues()).keys()
#print keys
'''
#Potential Features
'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 
'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 
'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 
'from_this_person_to_poi', 'poi', 'director_fees', 'deferred_income', 
'long_term_incentive', 'email_address', 'from_poi_to_this_person',

##Added
fraction_from_poi,fraction_to_poi
'''


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#Decent Results
def GNBAccuracy(features, labels):
    ##Using smaller training data
    #clf = SVC(kernel='linear')
    clf = GaussianNB()
    t0 = time()
    clf.fit(features,labels)
    print "training time:", round(time()-t0, 3), "s"
    t0 = time()
    pred = clf.predict(features)
    print 'predicting time',round(time()-t0,3),'s'
    accuracy = accuracy_score(labels,pred)
    print 'Naive Bayes Accuracy:',accuracy_score(labels,pred)
    print 'Naive Bayes Precision:',precision_score(labels,pred)
    print 'Naive Bayes recall:',recall_score(labels,pred)
    return clf

#Takes a LONG Time
def SVMAccuracy(features, labels):
    ##Using smaller training data
    clf = SVC(kernel='linear')
    #clf = SVC(C=10000,kernel='rbf')
    t0 = time()
    clf.fit(features,labels)
    print "training time:", round(time()-t0, 3), "s"
    t0 = time()
    pred = clf.predict(features)
    print 'predicting time',round(time()-t0,3),'s'
    accuracy = accuracy_score(labels,pred)
    print 'SVM Accuracy:',accuracy_score(labels,pred)
    print 'SVM Precision:',precision_score(labels,pred)
    print 'SVM recall:',recall_score(labels,pred)
    return clf
#Overfits
def DTAccuracy(features, labels):
    clf = tree.DecisionTreeClassifier(min_samples_split=2)
    t0 = time()
    clf.fit(features,labels)
    print "training time:", round(time()-t0, 3), "s"
    t0 = time()
    pred = clf.predict(features)
    print 'predicting time',round(time()-t0,3),'s'
    accuracy = accuracy_score(labels,pred)
    print 'DT Accuracy:',accuracy_score(labels,pred)
    print 'DT Precision:',precision_score(labels,pred)
    print 'DT recall:',recall_score(labels,pred)
    return clf

#########################################################
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

def GNBAccuracySplit(features_train, labels_train, features_test, labels_test):
    ##Using smaller training data
    #clf = SVC(kernel='linear')
    clf = GaussianNB()
    t0 = time()
    clf.fit(features_train,labels_train)
    print "training time:", round(time()-t0, 3), "s"
    t0 = time()
    pred = clf.predict(features_test)
    print 'predicting time',round(time()-t0,3),'s'
    accuracy = accuracy_score(labels_test,pred)
    print 'Naive Bayes(Split) Accuracy:',accuracy_score(labels_test,pred)
    print 'Naive Bayes(Split) Precision:',precision_score(labels_test,pred)
    print 'Naive Bayes(Split) recall:',recall_score(labels_test,pred)
    return clf   

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

def main(data_dict):
    QueryDataSet(data_dict)
    features_list=['poi','salary','exercised_stock_options','long_term_incentive','from_messages','fraction_to_poi']
    #Convert dictinonary to numpy array.
    data = featureFormat(data_dict,features_list,sort_keys = True)
    ### Extract features and labels from dataset for local testing
    labels, features = targetFeatureSplit(data)
    #Plot_3_Clustoids_BeforeScaling(labels,features)
    #Plot_3_Clustoids_AfterScaling(labels,features)
    GNBAccuracy(features, labels)
    features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    clf=GNBAccuracySplit(features_train, labels_train, features_test, labels_test)    
    dump_classifier_and_data(clf, data_dict, features_list)
    
main(data_dict)
