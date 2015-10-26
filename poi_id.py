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
from sklearn.metrics import r2_score
from time import time
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
import sklearn.neighbors as KN
import sklearn.ensemble as ensem
from sklearn.feature_selection import SelectPercentile,SelectKBest,f_classif
from sklearn.pipeline import Pipeline
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


def GNBAccuracyShuffle(features, labels,feature_names,folds = 1000):
    clf_NB = GaussianNB()
    #Ideal Value is percentil=5, after that all metrics start to Decrease.
    #Have to use 10 though
    fs=SelectPercentile(f_classif,percentile=10)
    scaler = MinMaxScaler()
    cv = StratifiedShuffleSplit(labels,folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    NB_pipeline=Pipeline([('Scale Features',scaler),('Select Features',fs),('GuassianNB',clf_NB)]) 
    features= NB_pipeline.named_steps['Select Features'].fit_transform(features, labels)
    feat_new= [feature_names[i]for i in NB_pipeline.named_steps['Select Features'].get_support(indices=True)]
    print feat_new
    for train_indices, test_indices in cv:
        features_train = [features[ii] for ii in train_indices]
        features_test = [features[ii] for ii in test_indices]
        labels_train = [labels[ii] for ii in train_indices]
        labels_test = [labels[ii] for ii in test_indices]
        t0 = time()
        NB_pipeline.fit(features_train, labels_train)
        pred = NB_pipeline.predict(features_test)
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
    print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
    print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
    print ""
    return NB_pipeline,feat_new

def SVMAccuracyGridShuffle(features, labels,feature_names,folds = 100):    
    fs=SelectKBest(f_classif, k=2)
    clf_SCV = SVC(kernel='rbf')
    #Cs=np.logspace(-2, 10, 13)
    #gammas=np.logspace(-9, 3, 13)
    scaler = MinMaxScaler()
    cv = StratifiedShuffleSplit(labels,folds, random_state = 42)
    pipe= Pipeline([('Scale_Features',scaler),('Select_Features',fs),('SVC',clf_SCV)]) 
    #Where would PCA fit into this?
    Klist=[]
    val=0
    for i in range(1,len(feature_names)):
        Klist.append(i)
    #Is this right?
    #Throwing in all of the featurs returns a better result,but
    #Will this always be the case? I would have thought that grid search
    #would have chosen the right combination.
    Cs=[1, 10, 100, 1000]
    #So we want ot change Select features so that we can see which values have the highist 
    #Scores
    params = dict(\
        SVC__C=Cs,\
        SVC__gamma=[0,0.0001,0.0005, 0.001, 0.005, 0.01, 0.1])
    #I Switched scoring to 'recall' and actually ended up with a WORSE value for that metric
    #f1 DIDN'T Work        
    clf_Grid_SVM = GridSearchCV(pipe,param_grid=params,cv=cv,scoring='accuracy')
    t0=time()
    clf_Grid_SVM.fit(features,labels)
    pred=clf_Grid_SVM.predict(features)
    print "training time:", round(time()-t0, 3), "s"
    print("Best estimator found by grid search:")
    print clf_Grid_SVM.best_estimator_
    #Why is the optimal K ALL features? Will it be like this all the time?
    KOpt= clf_Grid_SVM.best_params_['Select_Features__k']
    fs2=SelectKBest(f_classif, k=KOpt)
    fs2.fit_transform(features,labels)   
    feat_new=[feature_names[i]for i in fs2.get_support(indices=True)]
    print('Best Params found by grid search:')
    print clf_Grid_SVM.best_params_
    print('Best Score found by grid search:')
    print clf_Grid_SVM.best_score_      
    t0 = time()
    pred = clf_Grid_SVM.predict(features)
    print 'predicting time',round(time()-t0,3),'s'
    accuracy = accuracy_score(labels,pred)
    print 'SVM Accuracy:',accuracy_score(labels,pred)
    print 'SVM Precision:',precision_score(labels,pred)
    print 'SVM recall:',recall_score(labels,pred)
    print 'SVM F1:',f1_score(labels,pred)
    return clf_Grid_SVM,feat_new    
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
'''

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

def main(data_dict):
    #QueryDataSet(data_dict)
    feature_names=['poi','salary','exercised_stock_options',\
    'long_term_incentive','fraction_from_poi','fraction_to_poi',\
    'expenses','deferred_income','director_fees','loan_advances',\
    'total_stock_value','restricted_stock_deferred','restricted_stock',\
    'bonus','total_payments','deferral_payments','to_messages','from_messages']
    #PlotReg(data_dict,'With Outlier(s)')
    data_dict=RmOutliers(data_dict)
    #PlotReg(data_dict,'Without Outlier(s)')
    ##Convert dictinonary to numpy array.
    data = featureFormat(data_dict,feature_names,sort_keys = True)
    ## Extract features and labels from dataset for local testing
    labels, features = targetFeatureSplit(data)
    #Plot_3_Clustoids_AfterScaling(labels,features)
    clf,my_features=SVMAccuracyGridShuffle(features, labels,feature_names)
    #New I just added this
    my_features.append('poi')
    #clf,my_features=GNBAccuracyShuffle(features, labels,feature_names)
    my_dataset={}
    for k,v in data_dict.iteritems():
        my_dataset[k]={}
        for i in v:
            if i in my_features:
                my_dataset[k][i]=v[i]

    #It looks like it's an issue with the classifier....
    #We ren into the same issue regardless of it being the GNB classifier
    #or the SVM gridsearchCV one. So something weird is up.
    dump_classifier_and_data(clf, my_dataset, feature_names)

main(data_dict)
