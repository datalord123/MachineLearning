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
import sklearn.neighbors as KN
from sklearn.ensemble import AdaBoostClassifier


'''
pca=PCA(n_components=2)
pca.fit(data)
return pca
print pca.explained_variance_ratio_
first_pc=pca.components_[0]
second_pc=pca.components_[1]
transfomred_data=pca.transform(data)
'''

def QueryDataSet(my_dataset):
    print 'Total Number of Data Points:',len(my_dataset)
    print 'Number POIs:',sum(1 for v in my_dataset.values() if v['poi']==True)
    print 'Number non-POIs:',sum(1 for v in my_dataset.values() if v['poi']==False)
    keys = next(my_dataset.itervalues()).keys()
    print 'Number of Features:', len(keys) 
    FeatWNaN=dict.fromkeys(keys,0)
    FeatWNaNPOI=dict.fromkeys(keys,0)
	#Count the number of Missing values
    for k,v in my_dataset.iteritems():
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
    MisVal=df.sort('Missing Vals',ascending=0)
    MisVal.to_csv('MissingValues.csv')
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

def Plot_n_Clustoids_AfterScaling(poi,finance_features):
    scaler = MinMaxScaler()
    rescaled_features = scaler.fit_transform(finance_features)
    clust = KMeans(n_clusters=2)
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

#'from_poi_to_this_person'
#'to_messages'

#'from_this_person_to_poi'
#"from_messages"


def AddFeatures(my_dataset):
    for name in my_dataset:       
        from_poi_to_this_person = my_dataset[name]['from_poi_to_this_person']
        to_messages = my_dataset[name]['to_messages']

        fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages)
        my_dataset[name]["fraction_from_poi"]=fraction_from_poi

        from_this_person_to_poi = my_dataset[name]['from_this_person_to_poi']
        from_messages = my_dataset[name]["from_messages"]
        fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages) 
        my_dataset[name]["fraction_to_poi"] = fraction_to_poi
    return my_dataset

def ShowCorrel(my_dataset):
    dfCor= pd.DataFrame.from_dict(my_dataset,orient='index')
    dfCor.replace('NaN',np.NaN,inplace=True)
    CorTab= dfCor.corr()
    print CorTab
    CorTab.to_csv('CorrTable.csv')

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

def TuneKNN(features, labels,features_list,folds = 100):    
    features_list.remove('poi')
    clf = KN.KNeighborsClassifier(n_neighbors=2)
    KInit=4
    scaler=preprocessing.MinMaxScaler()
    fs=SelectKBest(f_classif, k=KInit)
    cv = StratifiedShuffleSplit(labels,folds, random_state = 17)
    pipe= Pipeline([('Scale',scaler),('Select_Features',fs),('Classifier',clf)])
    params = dict(Classifier__weights=['uniform','distance'],
        Classifier__algorithm=['auto','ball_tree','kd_tree','brute'],
        Classifier__n_neighbors=[1,2,3,4,5])
    clf_Grid = GridSearchCV(pipe,param_grid=params,cv=cv,scoring='f1_micro')
    clf_Grid.fit(features, labels)
    print"Best estimator found by grid search:\n",clf_Grid.best_estimator_
    PipeOpt=clf_Grid.best_estimator_
    np.set_printoptions(suppress=True)
    print('Best Params found by grid search: \n')
    print clf_Grid.best_params_
    return PipeOpt

def TuneSVM(features, labels,features_list,folds = 100):    
    features_list.remove('poi')
    clf = SVC(kernel='rbf')
    KInit=4
    fs=SelectKBest(f_classif, k=KInit)
    cv = StratifiedShuffleSplit(labels,folds, random_state = 17)
    pipe= Pipeline([('Select_Features',fs),('Classifier',clf)])
    COpt=[1,10,100,1000,5000,10000]
    GOpt=[1,10,25,50,75,90,100,'auto']
    CritOpt=['rbf','sigmoid']
    params = dict(Classifier__C=COpt,
        Classifier__gamma=GOpt,
        Classifier__kernel=CritOpt,
        Classifier__class_weight=[{0:.2,1:.8},{0:.15,1:.85},{0:.1,1:.9},'balanced'])
    clf_Grid = GridSearchCV(pipe,param_grid=params,cv=cv,scoring='f1_micro')
    clf_Grid.fit(features, labels)
    print"Best estimator found by grid search:\n",clf_Grid.best_estimator_
    PipeOpt=clf_Grid.best_estimator_
    np.set_printoptions(suppress=True)
    print('Best Params found by grid search: \n')
    print clf_Grid.best_params_
    my_features=[features_list[i]for i in PipeOpt.named_steps['Select_Features'].get_support(indices=True)]
    print 'Original Features List:\n',features_list
    print 'Features sorted by score(Biggest to Smallest):\n', [features_list[i] for i in np.argsort(PipeOpt.named_steps['Select_Features'].scores_)[::-1]]
    print 'Features Scores :\n', np.sort(PipeOpt.named_steps['Select_Features'].scores_)[::-1]
    print 'My Selected Features: \n',my_features
    #print 'Feature Importances:\n',PipeOpt.named_steps['Classifier'].feature_importances_
    #print 'Dataframe Example:\n', pd.DataFrame(PipeOpt.named_steps['Select_Features'].scores_,
    #         index=features_list).sort()
    return PipeOpt

#How do I use class_weight?
def TuneDT(features, labels,features_list,folds = 100):    
    features_list.remove('poi')
    clf = tree.DecisionTreeClassifier(min_samples_split=2)
    KInit=4
    fs=SelectKBest(f_classif, k=KInit)
    cv = StratifiedShuffleSplit(labels,folds, random_state = 17)
    pipe= Pipeline([('Select_Features',fs),('Classifier',clf)])
    SplitOpt=range(1,50)
    CritOpt=['entropy','gini']
    params = dict(Classifier__min_samples_split=SplitOpt,
        Classifier__criterion=CritOpt,
        Classifier__class_weight=[{0:.2,1:.8},{0:.15,1:.85},{0:.1,1:.9},'balanced'])
    clf_Grid = GridSearchCV(pipe,param_grid=params,cv=cv,scoring='f1_micro')
    clf_Grid.fit(features, labels)
    print"Best estimator found by grid search:\n",clf_Grid.best_estimator_
    PipeOpt=clf_Grid.best_estimator_
    np.set_printoptions(suppress=True)
    print('Best Params found by grid search: \n')
    print clf_Grid.best_params_
    my_features=[features_list[i]for i in PipeOpt.named_steps['Select_Features'].get_support(indices=True)]
    print 'Original Features List:\n',features_list
    print 'Features sorted by score(Biggest to Smallest):\n', [features_list[i] for i in np.argsort(PipeOpt.named_steps['Select_Features'].scores_)[::-1]]
    print 'Features Scores :\n', np.sort(PipeOpt.named_steps['Select_Features'].scores_)[::-1]
    print 'My Selected Features: \n',my_features
    print 'Feature Importances:\n',PipeOpt.named_steps['Classifier'].feature_importances_
    print 'Dataframe Example:\n', pd.DataFrame(PipeOpt.named_steps['Select_Features'].scores_,
             index=features_list).sort()
    return PipeOpt

def NoTuneDT(features, labels,features_list,folds = 100):    
    features_list.remove('poi')
    clf = tree.DecisionTreeClassifier(min_samples_split=2)
    KInit=4
    fs=SelectKBest(f_classif, k=KInit)
    cv = StratifiedShuffleSplit(labels,folds, random_state = 17)
    PipeOpt= Pipeline([('Select_Features',fs),('Classifier',clf)])
    PipeOpt.fit(features, labels)
    np.set_printoptions(suppress=True)
    my_features=[features_list[i]for i in PipeOpt.named_steps['Select_Features'].get_support(indices=True)]
    print 'Original Features List:\n',features_list
    print 'Features sorted by score(Biggest to Smallest):\n', [features_list[i] for i in np.argsort(PipeOpt.named_steps['Select_Features'].scores_)[::-1]]
    print 'Features Scores :\n', np.sort(PipeOpt.named_steps['Select_Features'].scores_)[::-1]
    print 'My Selected Features: \n',my_features
    print 'Feature Importances:\n',PipeOpt.named_steps['Classifier'].feature_importances_
    print 'Dataframe Example:\n', pd.DataFrame(PipeOpt.named_steps['Select_Features'].scores_,
             index=features_list).sort()
    return PipeOpt
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
#Why was AdaBoost So Much less effective than normal DT
#Regarding Training Set
#https://discussions.udacity.com/t/p5-testing-results-all-over-the-place/37850/9
def main():
    data_dict = pickle.load(open("final_project_dataset.pkl", "r"))
    my_dataset=data_dict
    my_dataset=AddFeatures(my_dataset)
    #Exclude using Discretion. 
    Exc1=['email_address']
    #Replaced by creating better versions of the features
    Exc2=['to_messages','from_messages','from_this_person_to_poi',\
    'from_poi_to_this_person']
    #Exclude because Highly Correlated with stronger features
    Exc3=['deferral_payments','expenses','deferred_income',\
    'restricted_stock_deferred','director_fees',\
    'long_term_incentive','bonus','total_payments',\
    'salary','total_stock_value','restricted_stock',\
    'exercised_stock_options','other',]
    exclude=Exc1+Exc2+Exc3
    #QueryDataSet(my_dataset)
    #ShowCorrel(my_dataset)
    features_list= next(my_dataset.itervalues()).keys()
    for i in exclude:
        features_list.remove(i)
    features_list.insert(0, features_list.pop(features_list.index('poi')))
    data = featureFormat(my_dataset,features_list,sort_keys = True)
    ### Extract features and labels from dataset for local testing
    labels,features = targetFeatureSplit(data)
    features_train,features_test,labels_train,labels_test= train_test_split(features,labels,\
        test_size=.1,random_state=42,stratify=labels)
    #clf=TuneSVM(features, labels,features_list)
    #clf=TuneKNN(features, labels,features_list)
    #clf=NoTuneDT(features, labels,features_list)
    #clf=TuneDT(features,labels,features_list)
    features_list.insert(0, 'poi')
    dump_classifier_and_data(clf, my_dataset, features_list)
    test_classifier(clf, my_dataset, features_list)


main()
