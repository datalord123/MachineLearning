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
from sklearn.metrics import f1_score
from time import time
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

def QueryDataSet(data_dict):
    print 'Total Number of Data Points:',len(data_dict)
    print 'Number POIs:',sum(1 for v in data_dict.values() if v['poi']==True)
    print 'Number non-POIs:',sum(1 for v in data_dict.values() if v['poi']==False)
    keys = next(data_dict.itervalues()).keys()
    print 'Number of Features:', len(keys)
    FeatWNaN=dict.fromkeys(keys,0)
    FeatWNaNPOI=dict.fromkeys(keys,0)
	#Count the number of Missing values
    for k,v in data_dict.iteritems():
        for i in v:
            if v[i] == 'NaN':
                FeatWNaN[i]+=1
            if v[i] == 'NaN' and v['poi']==True:
                FeatWNaNPOI[i]+=1
    df = pd.DataFrame.from_dict(FeatWNaN, orient='index')
    df = df.rename(columns = {0: 'Missing Vals'})
    dfPOI = pd.DataFrame.from_dict(FeatWNaNPOI, orient='index')
    dfPOI = dfPOI.rename(columns = {0: 'Missing Vals POI'})
    df=df.join(dfPOI)    
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
def PlotReg(data_dict,Title):
    RegFeatures = ["salary", "bonus"]
    data = featureFormat( data_dict, RegFeatures, remove_any_zeroes=True)
    target, features = targetFeatureSplit(data)
    PlotData(target,features,Title)

def RmOutliers(data_dict):
    data_dict.pop('TOTAL')   
    return data_dict    

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
    clf_NB = GaussianNB()
    t0 = time()
    #kf=KFold(len(features))
    clf_NB.fit(features,labels)
    print "training time:", round(time()-t0, 3), "s"
    t0 = time()
    pred = clf_NB.predict(features)
    print 'predicting time',round(time()-t0,3),'s'
    accuracy = accuracy_score(labels,pred)
    print 'Naive Bayes Accuracy:',accuracy_score(labels,pred)
    print 'Naive Bayes Precision:',precision_score(labels,pred)
    print 'Naive Bayes recall:',recall_score(labels,pred)
    return clf_NB

#Ovefits
def SVMAccuracy(features, labels):
    ##Using smaller training data
    clf_SVM = SVC(kernel='rbf')
    #clf = SVC(C=10000,kernel='rbf')
    t0 = time()
    clf_SVM.fit(features,labels)
    print "training time:", round(time()-t0, 3), "s"
    t0 = time()
    pred = clf_SVM.predict(features)
    print 'predicting time',round(time()-t0,3),'s'
    accuracy = accuracy_score(labels,pred)
    print 'SVM Accuracy:',accuracy_score(labels,pred)
    print 'SVM Precision:',precision_score(labels,pred)
    print 'SVM recall:',recall_score(labels,pred)
    return clf_SVM
#Overfits
def DTAccuracy(features, labels):
    clf_DT = tree.DecisionTreeClassifier(min_samples_split=2)
    t0 = time()
    clf_DT.fit(features,labels)
    print "training time:", round(time()-t0, 3), "s"
    t0 = time()
    pred = clf_DT.predict(features)
    print 'predicting time',round(time()-t0,3),'s'
    accuracy = accuracy_score(labels,pred)
    print 'DT Accuracy:',accuracy_score(labels,pred)
    print 'DT Precision:',precision_score(labels,pred)
    print 'DT recall:',recall_score(labels,pred)
    return clf_DT

#########################################################
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
'''
def GNBAccuracyKFold(features, labels):
    #parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}             
    clf_NB = GaussianNB()
    folds=10
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_indices, test_indices in cv:
        ##The Features and Labels Values in the loop...What are they?
        features_train = [features[ii] for ii in train_indices]
        features_test = [features[ii] for ii in test_indices]
        labels_train = [labels[ii] for ii in train_indices]
        labels_test = [labels[ii] for ii in test_indices]
        t0 = time()
        #### FOR SOME REASON SCLAING HAS NO IMPACT
        scaler = MinMaxScaler()
        features_train = scaler.fit_transform(features_train)            
        features_test = scaler.fit_transform(features_test)
        clf_NB.fit(features_train, labels_train)
        pred = clf_NB.predict(features_test) 
        for prediction, truth in zip(pred, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print 'Error'
    total_predictions = true_negatives + false_negatives + false_positives + true_positives
    accuracy = 1.0*(true_positives + true_negatives)/total_predictions
    precision = 1.0*true_positives/(true_positives+false_positives)
    recall = 1.0*true_positives/(true_positives+false_negatives)
    f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
    f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
    print clf_NB
    print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
    print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
    print ""      
    return clf_NB
'''
def SVMAccuracyGrid(features, labels):
    parameters = {'kernel':('rbf','sigmoid'),\
    'C':[1,10,1e2,1e3, 1e4, 1e5, 1e6],\
    'gamma': [0,0.0001,0.0005, 0.001, 0.005, 0.01, 0.1]} 
    svm = SVC()
    clf_SVM = GridSearchCV(svm, parameters)
    t0 = time()         
    clf_SVM.fit(features,labels)
    print "training time:", round(time()-t0, 3), "s"
    print("Best estimator found by grid search:")
    print clf_SVM.best_estimator_      
    t0 = time()
    pred = clf_SVM.predict(features)
    print 'predicting time',round(time()-t0,3),'s'
    accuracy = accuracy_score(labels,pred)
    print 'SVM Accuracy:',accuracy_score(labels,pred)
    print 'SVM Precision:',precision_score(labels,pred)
    print 'SVM recall:',recall_score(labels,pred)
    return clf_SVM
'''
SVC(C=1000.0, cache_size=200, class_weight='auto', coef0=0.0, degree=3,
  gamma=0.005, kernel='rbf', max_iter=-1, probability=False,
  random_state=None, shrinking=True, tol=0.001, verbose=False)

    #print "training time:", round(time()-t0, 3), "s"
    #t0 = time()
    #print 'predicting time',round(time()-t0,3),'s'
    #print 'Naive Bayes Accuracy:',accuracy_score(labels_test,pred)
    #print 'Naive Bayes Precision:',precision_score(labels_test,pred)
    #print 'Naive Bayes recall:',recall_score(labels_test,pred)
    #print 'Naive Bayes f1_score:',f1_score(labels_test, pred)  
'''
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

def main(data_dict):
    #QueryDataSet(data_dict)
    features_list=['poi','salary','exercised_stock_options','long_term_incentive','from_messages','fraction_to_poi']
    #PlotReg(data_dict,'With Outlier(s)')
    data_dict=RmOutliers(data_dict)
    #PlotReg(data_dict,'Without Outlier(s)')
    ##Convert dictinonary to numpy array.
    data = featureFormat(data_dict,features_list,sort_keys = True)
    ## Extract features and labels from dataset for local testing
    labels, features = targetFeatureSplit(data)
    #Plot_3_Clustoids_BeforeScaling(labels,features)
    #Plot_3_Clustoids_AfterScaling(labels,features)
    #DTAccuracy(features, labels)
    #clf=SVMAccuracy(features, labels)
    SVMAccuracyGrid(features, labels)
    #features_train, features_test, labels_train, labels_test = \
    #clf=GNBAccuracyKFold(features,labels)   
    #dump_classifier_and_data(clf, data_dict, features_list)

main(data_dict)
