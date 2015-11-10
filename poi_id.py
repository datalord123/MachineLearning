#!/usr/bin/python
import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,r2_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from time import time
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
import sklearn.neighbors as KN
import sklearn.ensemble as ensem
from sklearn.feature_selection import SelectPercentile,SelectKBest,f_classif
from sklearn.pipeline import Pipeline
from sklearn import decomposition,preprocessing
from tester import test_classifier

'''
pca=PCA(n_components=2)
pca.fit(data)
return pca
print pca.explained_variance_ratio_
first_pc=pca.components_[0]
second_pc=pca.components_[1]
transfomred_data=pca.transform(data)
'''

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

def ShowCorrel(data_dict):
    dfCor= pd.DataFrame.from_dict(data_dict,orient='index')
    dfCor.replace('NaN',np.NaN,inplace=True)
    dfCor.dropna(axis=0,how='any',inplace=True)
    print dfCor.corr()

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

def Plot_n_Clustoids_AfterScaling(poi,finance_features):
    scaler = MinMaxScaler()
    rescaled_features = scaler.fit_transform(finance_features)
    clust = KMeans(n_clusters=3)
    #print finance_features
    pred = clust.fit_predict(rescaled_features)
    DrawClusters(pred, rescaled_features, poi,'Clusters After Scaling', name="clusters_after_scaling.pdf", f1_name='salary', f2_name='exercised_stock_options')

### Load the dictionary containing the dataset
### Task 1: Select what features you'll use.
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

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#Decent Results

#########################################################
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
'''
Support vectors are a very bad fit for this dataset, 
because it is so sparse, distributed poorly, and highly imbalanced(email).

When using an rbf kernel, the support vector classifier tries to find areas around which pois 
are common, and areas around which pois are rare. Because the data is fairly spread out,
 and because there are so few pois, on this data set the SVC will end up essentially 
 building a small area around each poi, and then building enough areas to cover the 
 non-pois elsewhere. 
'''
    #print fs.get_support()
    #print 'All features:',feature_names
    #print 'Scores of these features:',fs.scores_
    #print '***Features sorted by score:', [feature_names[i] for i in np.argsort(fs.scores_)[::-1]]
    #Plot_3_Clustoids_AfterScaling(labels,features)
    #clf=GNBAccuracyShuffle(features, labels)
    #dump_classifier_and_data(clf, data_dict, feature_names)

def DTShuffleWPCA(features_train, labels_train, features_test, labels_test,feature_names,folds = 100):    
    KOpt=5
    fs=SelectKBest(f_classif, k=KOpt)
    clf = tree.DecisionTreeClassifier(min_samples_split=2)
    scaler = MinMaxScaler()
    cv = StratifiedShuffleSplit(labels_train,folds, random_state = 42)
    pca=decomposition.PCA(n_components=KOpt)   
    #How do i replace this in the pipeline!!!
    #scaled_data = preprocessing.scale(features_train)
    #I just applied PCA to the pipeline and it was a clusterfuck.
    #all of my metrics are bad now, I'm wondering if it's a scaling issue, but 
    #How could that be, is "PreProcessing Scaling" different from "MinMaxScaler"
    #pipe= Pipeline([('Scale_Features',scaler),('PCA',pca),('Select_Features',fs),('Classifier',cluster)])
    pipe= Pipeline([('Scale_Features',scaler),('PCA',pca),('Classifier',clf)])
    feature_names.remove('poi')
    Test=[1,5,10,15,20,25]
    params = dict(\
        Classifier__min_samples_split=Test)
    clf_Grid = GridSearchCV(pipe,param_grid=params,cv=cv,scoring='f1_weighted')
    clf_Grid.fit(features_train, labels_train)
    t0=time()


    #How do i deal with the testing features now? wouldn't that have  different PCA then?    
    print "training time:", round(time()-t0, 3), "s"
    prediction=clf_Grid.predict(features_test)
    print("Best estimator found by grid search:")
    print clf_Grid.best_estimator_
    print('Best Params found by grid search:')
    print clf_Grid.best_params_
    print('Best Score found by grid search:')  
    print clf_Grid.best_score_      
    print 'Accuracy:',accuracy_score(labels_test,prediction)    
    print 'Precision:',precision_score(labels_test,prediction)    
    print 'Recall:',recall_score(labels_test,prediction)    
    print 'F1 Score:',f1_score(labels_test,prediction)
    pipe.fit_transform(features_train,labels_train)
    #return clf_Grid, my_features
    return clf_Grid

#ASK about why precision goes to 0 when I include 1 in ClusTest, why does gridsearch go after 
#essentially it's own scoring metric, even though it's been assigned a weighted F1, the less clusters
#the higher it seems to go, but the expense of essentially all the remaining metrics(precision,recall,f1,etc)

#ASK about why the result sets are so random. They're all over the place.
#ASk about how to include PCA
#Ask about the error on top.

#K means is the wrong algorithm to use.
#Accuracy Scores are all over the place still
#Classifier__min_samples_split also keep selectin random values.
def DTShuffle(features_train, labels_train, features_test, labels_test,feature_names,folds = 100):    
    KOpt=5
    fs=SelectKBest(f_classif, k=KOpt)
    clf = tree.DecisionTreeClassifier(min_samples_split=2)
    scaler = MinMaxScaler()
    cv = StratifiedShuffleSplit(labels_train,folds, random_state = 42)
    #pipe= Pipeline([('Scale_Features',scaler),('Select_Features',fs),('Classifier',clf)])
    pipe= Pipeline([('Scale_Features',scaler),('Classifier',clf)])
    Test=[1,5,10,15,20,25]
    params = dict(\
        Classifier__min_samples_split=Test)
    clf_Grid = GridSearchCV(pipe,param_grid=params,cv=cv,scoring='f1_weighted')
    clf_Grid.fit(features_train, labels_train)
    t0=time()
    print "training time:", round(time()-t0, 3), "s"
    prediction=clf_Grid.predict(features_test)
    print("Best estimator found by grid search:")
    print clf_Grid.best_estimator_
    print('Best Params found by grid search:')
    print clf_Grid.best_params_
    print('Best Score found by grid search:')  
    print clf_Grid.best_score_      
    print 'Accuracy:',accuracy_score(labels_test,prediction)    
    print 'Precision:',precision_score(labels_test,prediction)    
    print 'Recall:',recall_score(labels_test,prediction)    
    print 'F1 Score:',f1_score(labels_test,prediction)
    pipe.fit_transform(features_train,labels_train)
    feature_names.remove('poi')
    #my_features=[feature_names[i]for i in pipe.named_steps['Select_Features'].get_support(indices=True)]
    #return clf_Grid, my_features
    return clf_Grid

    #Is there something else I could do, this is making my scores worse for some reason
    #pipe.fit(features_train,labels_train)    
    #print feat_new

'''
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
'''
'''
#Potential Features
'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 
'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 
'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 
'from_this_person_to_poi', 'poi', 'director_fees', 'deferred_income', 
'long_term_incentive', 'email_address', 'from_poi_to_this_person',
##Added
fraction_from_poi,fraction_to_poi

####!!!! Perform a Correlation Matrix to see which features are correlated with each other
http://pythonprogramming.net/pandas-statistics-correlation-tables-how-to/
or look at using PCA to look at variances
'''

def ExcludeFeatures(data_dict,exclude):
    #everything has a NaN have to remove the really high NaN elements.
    #Dropped columns with a large number of missing values so that I could filter out
    #missing records and Perform PCA/Correlation analysis.

    #Only seeing 31 records, see if any of the high missing values features kept in 
    #are correlated with features with lower number of missing values.    
    for k,v in data_dict.items():
        for i in v.keys():
            if i in exclude:
                del data_dict[k][i]

    feature_names = next(data_dict.itervalues()).keys()
    feature_names.insert(0, feature_names.pop(feature_names.index('poi')))
    return data_dict,feature_names

def SplitTestData(features,labels):    
    cv = StratifiedShuffleSplit(labels,n_iter=1, test_size=.2)
    for train_indices, test_indices in cv:
        features_train = [features[ii] for ii in train_indices]
        features_test = [features[ii] for ii in test_indices]
        labels_train = [labels[ii] for ii in train_indices]
        labels_test = [labels[ii] for ii in test_indices]
    return features_train,features_test,labels_train,labels_test


def main():
    data_dict = pickle.load(open("final_project_dataset.pkl", "r"))
    data_dict=AddFeatures(data_dict)
    #Removed features, with extremly high numbers of missing values,and\or
    #are highly correlated with another feature.
    exclude=['loan_advances','director_fees','restricted_stock_deferred',\
    'deferral_payments','deferred_income','email_address',\
    'exercised_stock_options','restricted_stock','other']
    #High Corr:'exercised_stock_options','restricted_stock_deferred',restricted_stock,'other'
    data_dict,feature_names=ExcludeFeatures(data_dict,exclude)
    #ShowCorrel(data_dict)
    #Next do something similar showing EDA, names of items linked together.
    #Ex:Exclude Total_Stock_value and Excercised stock options, should one be excluded?
    data = featureFormat(data_dict,feature_names,sort_keys = True)
    ## Extract features and labels from dataset for local testing
    labels, features = targetFeatureSplit(data)
    features_train,features_test,labels_train,labels_test= SplitTestData(features,labels)
    #Why is features_train in the top funciton a list, and the bottom function
    # a numpy array.
    #DTShuffle(features_train, labels_train, features_test, labels_test,feature_names) 
    DTShuffleWPCA(features_train, labels_train, features_test, labels_test,feature_names) 
    #QueryDataSet(data_dict)
    #PlotReg(data_dict,'With Outlier(s)')
    #PlotReg(data_dict,'Without Outlier(s)')
    ##Convert dictinonary to numpy array.
    #Plot_n_Clustoids_AfterScaling(labels,features)
    #SVMAccuracyGridShuffle(features, labels,feature_names)
    #clf,my_features=SVMAccuracyGridShuffle(features, labels,feature_names)
    #Above I am returning a fitted(GridSearch) Classifier. For submission,
    #should i just be returning the type of classifer? ex: SVM? 

    #New I just added this
    #print my_features
    #if 'poi' not in my_features:
    #    my_features.append('poi')
    #my_dataset={}
    #for k,v in data_dict.iteritems():
    #    my_dataset[k]={}
    #    for i in v:
    #        if i in my_features:
    #            my_dataset[k][i]=v[i]
    #feature_names.insert(0,'poi')
    #print feature_names
    #dump_classifier_and_data(clf, my_dataset, feature_names)


    #test_classifier(clf, my_dataset, feature_names)
    #It looks like it's an issue with the classifier....
    #We ren into the same issue regardless of it being the GNB classifier
    #or the SVM gridsearchCV one. So something weird is up.

main()
