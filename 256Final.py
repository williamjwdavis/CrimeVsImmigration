# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 12:34:30 2019

@author: Will
"""

import pandas as pd
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from pyclustering.cluster.optics import optics, ordering_analyser
from scipy.stats import skew
from scipy import stats

OUTERDROPCOLS=["year","murderManslaughterRate","rapeRate","robberyRate","aggravatedAssaultRate","burglaryRate",
               "larcenyTheftRate","motorVehicleTheftRate","laborForceRate","admittedImmigrantRate"]
INNERDROPCOLS=["year","violentCrimeRate","propertyCrimeRate","admittedImmigrantRate"]
STATES=["New Mexico","Florida","California","New York",
                   "Texas","Louisiana","Delaware","Tennessee"]


#ImportData
data=pd.read_excel("C:/Users/Will/Desktop/Stat256Final/Stat256AggregatedData.xlsx",sheetname=None)
#Drop all of the Columns we dont care about
for ele in data:
    data[ele]=data[ele].drop(columns=["state","violentCrimeTotal","murderAndNonnegligentManslaughter",
               "rape","robbery","aggravatedAssault","propertyCrimeTotal","burglary",
               "larcenyTheft","motorVehicleTheft","laborForce","laborForce",
               "employment","unemployment","totalPoverty","numberPoverty","admittedImmigrants",
               "employmentRate"
               #THESE ARE TEMP ONES THAT MAY BE REMOVED
               ])
               #"admittedImmigrantRateStd","admittedImmigrantRateNorm"])
newData={}
"""
Cube Root Stuff
"""
##Transform the data so that it is closer to normal
for ele in data:
    newData[ele]=pd.DataFrame()
    if ele not in ["year",'population']:
        for col in data[ele]:
            temp=pd.Series(data[ele][col]**(1/3))
            
            newData[ele][col]=temp
#Output the transformed data to excel
newData["AggregatedData"].to_excel("C:/Users/Will/Desktop/Stat256Final/AggregatedDataTransformed.xlsx")

"""
Norm Prob Plot
"""
fig = plt.figure()
ax=fig.add_subplot(111)
stats.probplot(data["AggregatedData"]["admittedImmigrantRate"],sparams=(2.5,), plot=ax)
ax.set_title("Normal Probability Plot for AdmittedImmigrantRate BEFORE Transformation")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
stats.probplot(newData["AggregatedData"]["admittedImmigrantRate"], sparams=(2.5,), plot=ax)
ax.set_title("Normal Probability Plot for AdmittedImmigrantRate AFTER Transformation")
plt.show()
"""
THIS IS THE UNDERLYING
CODE FOR STEPWISE REGRESSION
"""
def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

"""
DESCRIPTIVE STATISTICS 
ALL STATES AT ONCE
"""    
#Summary Statistics for all states as one list
data["AggregatedData"].describe(include="all")
for cols in data["AggregatedData"].columns:
    print(data["AggregatedData"][cols].describe())

tempStats=pd.Series()
count=0

for cols in data["AggregatedData"].columns:
    if count==0:
        tempStats=data["AggregatedData"][cols].describe()
    else:
        #tempStats=tempStats.append(data["AggregatedData"][cols].describe())
        tempStats=pd.concat([tempStats,pd.Series(cols),data["AggregatedData"][cols].describe()])
    count+=1

file=open("C:/Users/Will/Desktop/Stat256Final/results/aggregatedDataDescriptives.csv",'w')
#for cols in data["AggregatedData"].columns:
 #   file.write(data["AggregatedData"][cols].describe().to_csv())
file.write(tempStats.to_csv())
file.close()
"""
DESCRIPTIVE STATISTICS TRANSFORMED
ALL STATES AT ONCE
"""    
#Summary Statistics for all states as one list
newData["AggregatedData"].describe(include="all")
for cols in newData["AggregatedData"].columns:
    print(newData["AggregatedData"][cols].describe())

tempStats=pd.Series()
count=0

for cols in newData["AggregatedData"].columns:
    if count==0:
        tempStats=newData["AggregatedData"][cols].describe()
    else:
        #tempStats=tempStats.append(data["AggregatedData"][cols].describe())
        tempStats=pd.concat([tempStats,pd.Series(cols),newData["AggregatedData"][cols].describe()])
    count+=1

file=open("C:/Users/Will/Desktop/Stat256Final/results/Transformed/aggregatedDataDescriptivesTransformed.csv",'w')
#for cols in data["AggregatedData"].columns:
 #   file.write(data["AggregatedData"][cols].describe().to_csv())
file.write(tempStats.to_csv())
file.close()
"""
DESCRIPTIVE STATICS WITH TRANSFORMED DATA
ALL STATES AT ONCE
"""
newData["AggregatedData"].describe(include="all")
for cols in newData["AggregatedData"].columns:
    print(newData["AggregatedData"][cols].describe())

tempStats=pd.Series()
count=0

for cols in newData["AggregatedData"].columns:
    if count==0:
        tempStats=newData["AggregatedData"][cols].describe()
    else:
        #tempStats=tempStats.append(data["AggregatedData"][cols].describe())
        tempStats=pd.concat([tempStats,pd.Series(cols),newData["AggregatedData"][cols].describe()])
    count+=1

file=open("C:/Users/Will/Desktop/Stat256Final/results/Transformed/aggregatedDataDescriptivesTransformed.csv",'w')
#for cols in data["AggregatedData"].columns:
 #   file.write(data["AggregatedData"][cols].describe().to_csv())
file.write(tempStats.to_csv())
file.close()

"""
DESCRIPTIVE STATISTICS
ALL STATES SEPARATLEY
"""
descriptives={}
file=open('C:/Users/Will/Desktop/Stat256Final/results/separateStatesDescriptives.csv','w')
for ele in data:
    descriptives[ele]=data[ele].describe(include='all')
for ele in descriptives:
    file.write(pd.Series(ele).to_csv())
    file.write(descriptives[ele].to_csv())
file.close()

"""
DESCRIPTIVE STATISTICS TRANSFORMED
ALL STATES SEPARATLEY
"""
descriptives={}
file=open('C:/Users/Will/Desktop/Stat256Final/results/Transformed/separateStatesDescriptivesTransformed.csv','w')
for ele in newData:
    descriptives[ele]=newData[ele].describe(include='all')
for ele in descriptives:
    file.write(pd.Series(ele).to_csv())
    file.write(descriptives[ele].to_csv())
file.close()

"""
HISTOGRAMS 
AggregatedData
"""
for element in data["AggregatedData"].columns:
    plt.hist(data["AggregatedData"][element])
    plt.title("Histogram for AggregatedData:%s"% element)
    plt.savefig('C:/Users/Will/Desktop/Stat256Final/Histograms/AggregatedData/%s.png' % element)
    plt.close()
"""
HISTIGRAMS TRANFORMED
AggregatedData
"""    
for element in newData["AggregatedData"].columns:
    plt.hist(newData["AggregatedData"][element])
    plt.title("Histogram for AggregatedData:%s"%element)
    plt.savefig('C:/Users/Will/Desktop/Stat256Final/Histograms/AggregatedDataTransformed/%s.png' % element)
    plt.close()
"""
HISTOGRAMS
Specific Variables for all states
"""
#Immigration Rate
for element in data:
    plt.hist(data[element]["admittedImmigrantRate"])
    plt.title("Histogram for %s's ImmigrationRate"%element)
    plt.savefig('C:/Users/Will/Desktop/Stat256Final/Histograms/ImmigrationRate/%s.png'%element)
    plt.close()
#Violent Crime Rate
for element in data:
    plt.hist(data[element]["violentCrimeRate"])
    plt.title("Histogram for %s's Violent Crime Rate"%element)
    plt.savefig('C:/Users/Will/Desktop/Stat256Final/Histograms/ViolentCrimeRate/%s.png'%element)
    plt.close()
#Property Crime Rate
for element in data:
    plt.hist(data[element]["propertyCrimeRate"])
    plt.title("Histogram for %s's Property Crime Rate"%element)
    plt.savefig('C:/Users/Will/Desktop/Stat256Final/Histograms/PropertyCrimeRate/%s.png' % element)
    plt.close()
"""
HISTOGRAMS TRANSFORMED
Specific Variables for all states
"""
#Immigration Rate
for element in newData:
    plt.hist(newData[element]["admittedImmigrantRate"])
    plt.title("Histogram for %s's ImmigrationRate"%element)
    plt.savefig('C:/Users/Will/Desktop/Stat256Final/Histograms/ImmigrationRateTransformed/%s.png'%element)
    plt.close()
#Violent Crime Rate
for element in newData:
    plt.hist(newData[element]["violentCrimeRate"])
    plt.title("Histogram for %s's Violent Crime Rate"%element)
    plt.savefig('C:/Users/Will/Desktop/Stat256Final/Histograms/ViolentCrimeRateTransformed/%s.png'%element)
    plt.close()
#Property Crime Rate
for element in newData:
    plt.hist(newData[element]["propertyCrimeRate"])
    plt.title("Histogram for %s's Property Crime Rate"%element)
    plt.savefig('C:/Users/Will/Desktop/Stat256Final/Histograms/PropertyCrimeRateTransformed/%s.png' % element)
    plt.close()
"""
LINEAR REGRESSIONS
EXAMPLE FOR ALABAMA
"""
beenSeen={}
linearRegressions={}
states={}

for c1 in data["Alabama"].columns:
    beenSeen[c1]=True
    for c2 in data["Alabama"].columns:
        if c2 in beenSeen.keys():
            continue
        else:
            X=data["Alabama"][c1]
            y=data["Alabama"][c2]
            X=sm.add_constant(X)
            
            #Perform the Linear regression and store the results
            linearRegressions[c1+' vs. '+c2]= sm.OLS(y,X).fit()
            
            #Use Regplot to output the results
            #We dont use X here because X is 2-dimensional since we added a constant
            sns.regplot(data["Alabama"][c1],y)
            plt.show()
"""
CORRELATION MATRIX
EXAMPLE FOR ALABAMA

USES HEATMAPPING
INSTEAD OF JUST PLAIN CORRMATRIX
"""
#Heatmapping
heatCorr=data["Alabama"].corr()
labels=["Year","Population","ViolentCrimeRate","Murder&ManSalughterRate","RapeRate",
        "RobberyRate","AggravtedAssaultRate","PropertyCrimeRate","BurglaryRate",
        "Larceny&TheftRate","MotorVehicleTheftRate","LaborForce%","UnemploymentRate",
        "EmploymentRate","PovertyRate","AddmittedImmigrantRate"]
sns.heatmap(heatCorr,xticklabels=heatCorr.columns,yticklabels=heatCorr.columns)

#Outputs the heatmaps to a folder
#Also puts the plain matrix in a file
for element in data:
    if element=='AggregatedData':
        continue
    elif element=='Alabama':
        corrMatrix=data[element].corr()
        
        title=pd.DataFrame([element]*15)
        title=title.transpose()
        title=pd.DataFrame(title.values, index=['State'])
        title.columns=corrMatrix.columns
        
        finalMatrix=pd.concat([title, corrMatrix])
    else:
        corrMatrix=data[element].corr()
        
        title=pd.DataFrame([element]*15,index=corrMatrix.columns)
        title=title.transpose()
        title=pd.DataFrame(title.values, index=['State'])
        title.columns=corrMatrix.columns
        
        temp=pd.concat([title, corrMatrix])
        finalMatrix=finalMatrix.append(temp)
        
finalMatrix.to_csv('C:/Users/Will/Desktop/Stat256Final/results/corrMatrix.csv')

#AggregatedData Correlation Matrix
corrMatrix=data['AggregatedData'].corr()
corrMatrix.to_csv('C:/Users/Will/Desktop/Stat256Final/results/aggregatedDataCorrMatrix.csv')

for element in data:
    headCorr=data[element].corr()
    labels=["Year","Population","ViolentCrimeRate","Murder&ManSalughterRate","RapeRate",
        "RobberyRate","AggravtedAssaultRate","PropertyCrimeRate","BurglaryRate",
        "Larceny&TheftRate","MotorVehicleTheftRate","LaborForce%","UnemploymentRate",
        "EmploymentRate","PovertyRate","AddmittedImmigrantRate"]
    sns.heatmap(heatCorr,xticklabels=heatCorr.columns,yticklabels=heatCorr.columns)
    plt.savefig('C:/Users/Will/Desktop/Stat256Final/CorrelationPlots/(%s)CorrelationPlot.png' % element)
    plt.close()
    
"""
CORRELATION MATRIX TRANSFORMED
"""
#Outputs the heatmaps to a folder
#Also puts the plain matrix in a file
for element in newData:
    if element=='AggregatedData':
        continue
    elif element=='Alabama':
        corrMatrix=newData[element].corr()
        
        title=pd.DataFrame([element]*15)
        title=title.transpose()
        title=pd.DataFrame(title.values, index=['State'])
        title.columns=corrMatrix.columns
        
        finalMatrix=pd.concat([title, corrMatrix])
    else:
        corrMatrix=newData[element].corr()
        
        title=pd.DataFrame([element]*15,index=corrMatrix.columns)
        title=title.transpose()
        title=pd.DataFrame(title.values, index=['State'])
        title.columns=corrMatrix.columns
        
        temp=pd.concat([title, corrMatrix])
        finalMatrix=finalMatrix.append(temp)
        
finalMatrix.to_csv('C:/Users/Will/Desktop/Stat256Final/results/Transformed/corrMatrixTransformed.csv')

#AggregatedData Correlation Matrix
corrMatrix=newData['AggregatedData'].corr()
corrMatrix.to_csv('C:/Users/Will/Desktop/Stat256Final/results/Transformed/aggregatedDataCorrMatrixTransformed.csv')

for element in newData:
    heatCorr=newData[element].corr()
    labels=["Year","Population","ViolentCrimeRate","Murder&ManSalughterRate","RapeRate",
        "RobberyRate","AggravtedAssaultRate","PropertyCrimeRate","BurglaryRate",
        "Larceny&TheftRate","MotorVehicleTheftRate","LaborForce%","UnemploymentRate",
        "EmploymentRate","PovertyRate","AddmittedImmigrantRate"]
    sns.heatmap(heatCorr,xticklabels=heatCorr.columns,yticklabels=heatCorr.columns)
    plt.savefig('C:/Users/Will/Desktop/Stat256Final/CorrelationPlotsTransformed/(%s)CorrelationPlot.png' % element)
    plt.close()
"""
MULTIPLE REGRESSION
ALABAMA EXAMPLE
**All Variables**
"""
X=data["Alabama"]
X=X.drop(columns='admittedImmigrantRate')

X=sm.add_constant(X)
y=data["Alabama"]['admittedImmigrantRate']

alabamaFit=sm.OLS(y,X).fit()

print(alabamaFit.summary())
"""
MULTIPLE REGRESSION
ALABAMA EXAMPLE
**Stepwise**
"""
X=data["Alabama"]
X=X.drop(columns='admittedImmigrantRate')
X=sm.add_constant(X)
y=data["Alabama"]["admittedImmigrantRate"]

attributes=stepwise_selection(X,y,threshold_in=.15, threshold_out=.2)

alabamaStepwise=sm.OLS(y,X[attributes]).fit()
print(alabamaStepwise.summary())
"""
MULTIPLE REGRESSIONS
Only "Outer Variables"
"""
X=data["AggregatedData"]
X=X.drop(columns=OUTERDROPCOLS)
X=sm.add_constant(X)

y=data["AggregatedData"]["admittedImmigrantRate"]

pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)

outerModel=sm.OLS(y,X).fit()
print(outerModel.summary())
"""
MULTIPLE REGRESSIONS
Only "Inner Variables"
"""
X=data["AggregatedData"]
X=X.drop(columns=["admittedImmigrantRate","admittedImmigrantRateStd","admittedImmigrantNorm"])
X=X.drop(columns=["violentCrimeRate","propertyCrimeRate","employmentRate"])
X=sm.add_constant(X)

y=data["AggregatedData"]["admittedImmigrantRate"]

pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)

innerModel=sm.OLS(y,X).fit()
print(innerModel.summary())
"""
MULTIPLE REGRESSION
FOR ALL STATES
FOR AGGREGATED DATASET
"""
X=data["AggregatedData"]
X=X.drop(columns=["admittedImmigrantRate","admittedImmigrantRateStd","admittedImmigrantNorm"])
X=sm.add_constant(X)

y=data["AggregatedData"]["admittedImmigrantRate"]

aggregatedModel=sm.OLS(y,X).fit()

file = open('C:/Users/Will/Desktop/Stat256Final/results/multipleRegressionOutputAggregatedData.csv','w')
file.write(str(aggregatedModel.summary()))
file.close()
"""
MULTIPLE REGRESSION
FOR ALL STATES
**All Variables**
"""
multipleRegressions={}

for element in data:
    X=data[element]
    X=X.drop(columns=["admittedImmigrantRate","admittedImmigrantRateStd","admittedImmigrantNorm"])
    X=sm.add_constant(X)
    
    y=data[element]['admittedImmigrantRate']
    
    fit=sm.OLS(y,X).fit()
    #Creates a dictionary of regression fits
    multipleRegressions[element]=fit
    
file = open('C:/Users/Will/Desktop/Stat256Final/results/multipleRegressionOutputAllStates.csv','w')
for element in multipleRegressions:
    file.write(multipleRegressions[element].summary().as_csv())
file.close()
"""
MULTIPLE REGRESSION
FOR ALL STATES
**Stepwise**
"""
stepwiseRegressions={}

for element in data:
    X=data[element]
    X=X.drop(columns=["admittedImmigrantRate","admittedImmigrantRateStd","admittedImmigrantNorm"])
    X=sm.add_constant(X)
    
    y=data[element]['admittedImmigrantRate']
    
    attributes=stepwise_selection(X,y,threshold_in=.15,threshold_out=.2)
    
    stateStepwise=sm.OLS(y,X[attributes]).fit()
file = open('C:/Users/Will/Desktop/Stat256Final/results/stepwiseRegressionOutputAllStates.csv','w')
for element in stepwiseRegressions:
    file.write(stepwiseRegressions[element].summary().as_csv())
file.close()
"""
MULTIPLE REGRESSION TRANSFORMED
AGGREGATED DATA
STEPWISE
"""
X=newData["AggregatedData"]
X=X.drop(columns=["admittedImmigrantRate","year","murderManslaughterRate","rapeRate","robberyRate",
                  "aggravatedAssaultRate", "burglaryRate", "larcenyTheftRate", "motorVehicleTheftRate",
                  ])
X=sm.add_constant(X)

y=newData["AggregatedData"]["admittedImmigrantRate"]
attributes=stepwise_selection(X,y,threshold_in=.15,threshold_out=.2)

regression=sm.OLS(y,X[attributes]).fit()
    
print(regression.summary())

"""
SIMPLE REGRESSION TRANSFORMED
AGGREGATED DATA
"""
#AdmittedImmigrantRate vs violentCrimeRate
X=newData["AggregatedData"]["violentCrimeRate"]
y=newData["AggregatedData"]["admittedImmigrantRate"]

sns.regplot(x=X,y=y)
X=sm.add_constant(X)

regression=sm.OLS(y,X).fit()

regression.summary()

#AdmittedImmigrantRate vs propertyCrimeRate
X=newData["AggregatedData"]["propertyCrimeRate"]
y=newData["AggregatedData"]["admittedImmigrantRate"]

sns.regplot(x=X,y=y)
X=sm.add_constant(X)

regression=sm.OLS(y,X).fit()

regression.summary()

#AdmittedImmigrantRate vs population
X=newData["AggregatedData"]["population"]
y=newData["AggregatedData"]["admittedImmigrantRate"]

sns.regplot(x=X,y=y)
X=sm.add_constant(X)

regression=sm.OLS(y,X).fit()

regression.summary()

#AdmittedImmigrantRate vs poverty
X=newData["AggregatedData"]["povertyRate"]
y=newData["AggregatedData"]["admittedImmigrantRate"]

sns.regplot(x=X,y=y)
X=sm.add_constant(X)

regression=sm.OLS(y,X).fit()

regression.summary()

#AdmittedImmigrantRate vs unemployment
X=newData["AggregatedData"]["unemploymentRate"]
y=newData["AggregatedData"]["admittedImmigrantRate"]

sns.regplot(x=X,y=y)
X=sm.add_constant(X)

regression=sm.OLS(y,X).fit()

regression.summary()

"""
MUTLPLE REGRESSION TRANSFORMED
AGGREGATED DATA
OUTER MODEL
All Variables
"""
X=newData["AggregatedData"]
X=X.drop(columns=OUTERDROPCOLS)
X=sm.add_constant(X)
y=newData["AggregatedData"]["admittedImmigrantRate"]
regression=sm.OLS(y,X).fit()

#This step does VIFS
vif=pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)

file = open('C:/Users/Will/Desktop/Stat256Final/results/Transformed/multipleRegressionOuterTransformed.csv','w')
file.write(str("AggregatedData"+"\n"))
file.write(str(vif))
file.write(str(regression.summary()))
file.close()
"""
MULTIPLE REGRESSION TRANFORMED
AGGREGATED DATA
INNER MODDEL
All Variables
"""
X=newData["AggregatedData"]
X=X.drop(columns=INNERDROPCOLS)
X=sm.add_constant(X)
y=newData["AggregatedData"]["admittedImmigrantRate"]
regression=sm.OLS(y,X).fit()

#This step does VIFS
vif=pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)

file = open('C:/Users/Will/Desktop/Stat256Final/results/Transformed/multipleRegressionInnerTransformed.csv','w')
file.write(str("AggregatedData"+"\n"))
file.write(str(vif))
file.write(str(regression.summary()))
file.close()

"""
MULTIPLE REGRESSION TRANSFORMED
AGGREGATED DATA
OUTER MODEL
Stepwise
"""
X=newData["AggregatedData"]
X=X.drop(columns=OUTERDROPCOLS)
X=sm.add_constant(X)
y=newData["AggregatedData"]["admittedImmigrantRate"]

attributes=stepwise_selection(X,y,threshold_in=.1,threshold_out=.2)
regression=sm.OLS(y,X[attributes]).fit()

#This step does VIFS
vif=pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)

file = open('C:/Users/Will/Desktop/Stat256Final/results/Transformed/stepwiseRegressionOuterTransformed.csv','w')
file.write(str("AggregatedData"+"\n"))
file.write(str(vif))
file.write(str(regression.summary()))
file.close()

"""
MUTLIPLE REGRESSION TRANSFORMED
AGGREGATED DATA
INNER MODEL
Stepwise
NULL
"""
X=newData["AggregatedData"]
X=X.drop(columns=INNERDROPCOLS)
X=sm.add_constant(X)
y=newData["AggregatedData"]["admittedImmigrantRate"]
attributes=stepwise_selection(X,y,threshold_in=.1,threshold_out=.2)
regression=sm.OLS(y,X[attributes]).fit()

#This step does VIFS
vif=pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)

file = open('C:/Users/Will/Desktop/Stat256Final/results/Transformed/stepwiseRegressionInnerTransformed.csv','w')
file.write(str("AggregatedData"+"\n"))
file.write(str(vif))
file.write(str(regression.summary()))
file.close()

"""
FINAL MODEL 2
"""
X=newData["AggregatedData"]
X=X.drop(columns=INNERDROPCOLS)
X=X.drop(columns=["burglaryRate","robberyRate","murderManslaughterRate"])
X=sm.add_constant(X)
y=newData["AggregatedData"]["admittedImmigrantRate"]

attributes=stepwise_selection(X,y,threshold_in=.1,threshold_out=.2)
regression=sm.OLS(y,X[attributes]).fit()

#This step does VIFS
vif=pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)

file = open('C:/Users/Will/Desktop/Stat256Final/results/Transformed/finalModel2.csv','w')
file.write(str("AggregatedData"+"\n"))
file.write(str(vif))
file.write(str(regression.summary()))
file.close()

"""
MULTIPLE REGRESSION TRANSFORMED
ALL STATES
OUTER MODEL
All Variables
"""
regressions={}
vif={}
file=open('C:/Users/Will/Desktop/Stat256Final/results/Transformed/allStatesMultipleRegressionOuterTransformed.csv','w')

for ele in newData:
    if ele in STATES:
        X=newData[ele]
        X=X.drop(columns=OUTERDROPCOLS)
        X=sm.add_constant(X)
        
        y=newData[ele]["admittedImmigrantRate"]
        
        regressions[ele]=sm.OLS(y,X).fit()
        
        #This step does VIFS
        vif[ele]=pd.Series([variance_inflation_factor(X.values, i) 
            for i in range(X.shape[1])], 
            index=X.columns)
        
        file.write(str(ele+"\n"))
        file.write(str(vif))
        file.write("\n")
        file.write(str(regressions[ele].summary()))
        file.write("\n\n\n\n")
file.close()
             
"""
MUTLPLE REGRESSION TRANSFORMED
ALL STATES
INNER MODEL
All Variables
"""
regressions={}
vif={}
file=open('C:/Users/Will/Desktop/Stat256Final/results/Transformed/allStatesMultipleRegressionInnerTransformed.csv','w')

for ele in newData:
    if ele in STATES:
        X=newData[ele]
        X=X.drop(columns=INNERDROPCOLS)
        X=sm.add_constant(X)
        
        y=newData[ele]["admittedImmigrantRate"]
        
        regressions[ele]=sm.OLS(y,X).fit()
        
        #This step does VIFS
        vif[ele]=pd.Series([variance_inflation_factor(X.values, i) 
            for i in range(X.shape[1])], 
            index=X.columns)
        
        file.write(str(ele+"\n"))
        file.write(str(vif[ele]))
        file.write("\n")
        file.write(str(regressions[ele].summary()))
        file.write("\n\n\n\n")
file.close()

"""
MULTIPLE REGRESSION TRANSFORMED
ALL STATES
OUTER MODEL
Stepwise
"""
regressions={}
vif={}
file=open('C:/Users/Will/Desktop/Stat256Final/results/Transformed/allStatesStepwiseRegressionOuterTransformed.csv','w')

for ele in newData:
    if ele in STATES:
        X=newData[ele]
        X=X.drop(columns=OUTERDROPCOLS)
        X=sm.add_constant(X)
        
        y=newData[ele]["admittedImmigrantRate"]
        
        attributes=stepwise_selection(X,y,threshold_in=.1,threshold_out=.2)
        
        regressions[ele]=sm.OLS(y,X[attributes]).fit()
        
        #This step does VIFS
        vif[ele]=pd.Series([variance_inflation_factor(X.values, i) 
            for i in range(X.shape[1])], 
            index=X.columns)
        
        file.write(str(ele+"\n"))
        file.write(str(vif[ele]))
        file.write("\n")
        file.write(str(regressions[ele].summary()))
        file.write("\n\n\n\n")
file.close()

"""
MULTIPLE REGRESSION TRANSFORMED
ALL STATES
INNER MODEL
Stepwise
"""
regressions={}
vif={}
file=open("C:/Users/Will/Desktop/Stat256Final/results/Transformed/allStatesStepwiseRegressionInnerTransformed.csv",'w')

for ele in newData:
    if ele in STATES:
        X=newData[ele]
        X=X.drop(columns=INNERDROPCOLS)
        X=sm.add_constant(X)
        
        y=newData[ele]["admittedImmigrantRate"]
        
        attributes=stepwise_selection(X,y,threshold_in=.1,threshold_out=.2)
        
        regressions[ele]=sm.OLS(y,X[attributes]).fit()
        
        #This step does VIFS
        vif[ele]=pd.Series([variance_inflation_factor(X.values, i) 
            for i in range(X.shape[1])], 
            index=X.columns)
        
        file.write(str(ele+"\n"))
        file.write(str(vif[ele]))
        file.write("\n")
        file.write(str(regressions[ele].summary()))
        file.write("\n\n\n\n")
file.close()

"""
NEWEST MULTIPLE REGRESSION TRANSFORMED
NULL
"""

stepwiseRegressions={}

for element in newData:
    if element in STATES:
        X=newData[element]
        X=X.drop(columns=["year","murderManslaughterRate","rapeRate","robberyRate","aggravatedAssaultRate",
                      "burglaryRate","larcenyTheftRate","motorVehicleTheftRate","admittedImmigrantRate"])
        X=sm.add_constant(X)
    
        y=newData[element]["admittedImmigrantRate"]
    
        attributes=stepwise_selection(X,y,threshold_in=.1,threshold_out=.2)
    
        stepwiseRegressions[element]=sm.OLS(y,X[attributes]).fit()
        print(stepwiseRegressions[element].summary())
file = open("C:/Users/Will/Desktop/Stat256Final/results/Transformed/stepwiseRegressionOuterModelTransformed.csv",'w')
for element in stepwiseRegressions:
    file.write(element)
    file.write(stepwiseRegressions[element].summary().as_csv())
file.close()

"""
CLUSTERING ANALYSIS
AggregatedStates
2 variables at a time
"""
"""
KMEANS
"""
#Immigration vs ViolentCrime
immigrationViolentCrime=pd.DataFrame(data["AggregatedData"]["admittedImmigrantRate"])
immigrationViolentCrime=immigrationViolentCrime.join(data["AggregatedData"]["violentCrimeRate"])
kMeansViolentCrime=KMeans(n_clusters=3).fit(immigrationViolentCrime)

clusterLabels=pd.DataFrame(kMeansViolentCrime.labels_,columns=["labels"])

immigrationViolentCrime=immigrationViolentCrime.join(clusterLabels)

sns.pairplot(x_vars=["violentCrimeRate"], y_vars=["admittedImmigrantRate"], 
             data=immigrationViolentCrime, hue="labels", size=10)

#Immigration vs PropertyCrime
immigrationPropertyCrime=pd.DataFrame(data["AggregatedData"]["admittedImmigrantRate"])
immigrationPropertyCrime=immigrationPropertyCrime.join(data["AggregatedData"]["propertyCrimeRate"])
kMeansPropertyCrime=KMeans(n_clusters=3).fit(immigrationPropertyCrime)

clusterLabels=pd.DataFrame(kMeansViolentCrime.labels_,columns=["labels"])

immigrationPropertyCrime=immigrationPropertyCrime.join(clusterLabels)

sns.pairplot(x_vars=["propertyCrimeRate"], y_vars=["admittedImmigrantRate"], 
             data=immigrationPropertyCrime, hue="labels", size=10)


"""
CLUSTERING ANALYSIS TRANSFORMED
AggregatedStates
2 variables at a time
"""
"""
KMEANS TRANSFORMED
"""
#Immigration vs ViolentCrime
immigrationViolentCrime=pd.DataFrame(newData["AggregatedData"]["admittedImmigrantRate"])
immigrationViolentCrime=immigrationViolentCrime.join(newData["AggregatedData"]["violentCrimeRate"])
kMeansViolentCrime=KMeans(n_clusters=3).fit(immigrationViolentCrime)

clusterLabels=pd.DataFrame(kMeansViolentCrime.labels_,columns=["labels"])

immigrationViolentCrime=immigrationViolentCrime.join(clusterLabels)

sns.pairplot(x_vars=["violentCrimeRate"], y_vars=["admittedImmigrantRate"], 
             data=immigrationViolentCrime, hue="labels", size=10)

#Immigration vs PropertyCrime
immigrationPropertyCrime=pd.DataFrame(newData["AggregatedData"]["admittedImmigrantRate"])
immigrationPropertyCrime=immigrationPropertyCrime.join(newData["AggregatedData"]["propertyCrimeRate"])
kMeansPropertyCrime=KMeans(n_clusters=3).fit(immigrationPropertyCrime)

clusterLabels=pd.DataFrame(kMeansViolentCrime.labels_,columns=["labels"])

immigrationPropertyCrime=immigrationPropertyCrime.join(clusterLabels)

sns.pairplot(x_vars=["propertyCrimeRate"], y_vars=["admittedImmigrantRate"], 
             data=immigrationPropertyCrime, hue="labels", size=10)


"""
DBSCAN
"""
#Immigration vs ViolentCrime
immigrationViolentCrime=pd.DataFrame(data["AggregatedData"]["admittedImmigrantRate"])
immigrationViolentCrime=immigrationViolentCrime.join(data["AggregatedData"]["violentCrimeRate"])
dbscanViolentCrime=DBSCAN(eps=.00825, min_samples=2).fit(immigrationViolentCrime)
#opticsViolentCrime=optics(immigrationViolentCrime, eps=1, minpts=50)
#opticsViolentCrime.process()
#opticsOrdering=opticsViolentCrime.get_ordering()

dbscanLabels=pd.DataFrame(dbscanViolentCrime.labels_,columns=["dbscanLabels"])
#opticsLabels=pd.DataFrame(opticsViolentCrime.labels_,columns=["opticsLabels"])

immigrationViolentCrime=immigrationViolentCrime.join(dbscanLabels)
#immigrationViolentCrime=immigrationViolentCrime.join(opticsLabels)

#DBSCAN PLOT
sns.pairplot(x_vars=["violentCrimeRate"],y_vars=["admittedImmigrantRate"],
             data=immigrationViolentCrime, hue='dbscanLabels',size=10)  
#OPTICS PLOT
#sns.pairplot(x_vars=["violentCrimeRate"],y_vars=["admittedImmigrantRate"],
#             data=immigrationViolentCrime, hue='opticsLabels', size=10)


#Immigration vs PropertyCrime
immigrationPropertyCrime=pd.DataFrame(data["AggregatedData"]["admittedImmigrantRate"])
immigrationPropertyCrime=immigrationPropertyCrime.join(data["AggregatedData"]["propertyCrimeRate"])
dbscanPropertyCrime=DBSCAN(eps=.00825, min_samples=2).fit(immigrationPropertyCrime)

clusterLabels=pd.DataFrame(dbscanViolentCrime.labels_,columns=["labels"])

immigrationPropertyCrime=immigrationPropertyCrime.join(clusterLabels)

sns.pairplot(x_vars=["propertyCrimeRate"], y_vars=["admittedImmigrantRate"], 
             data=immigrationPropertyCrime, hue="labels", size=10)


"""
DBSCAN TRANSFORMED
"""
#Immigration vs ViolentCrime
immigrationViolentCrime=pd.DataFrame(newData["AggregatedData"]["admittedImmigrantRate"])
immigrationViolentCrime=immigrationViolentCrime.join(newData["AggregatedData"]["violentCrimeRate"])
dbscanViolentCrime=DBSCAN(eps=.0045, min_samples=10).fit(immigrationViolentCrime)
#opticsViolentCrime=optics(immigrationViolentCrime, eps=1, minpts=50)
#opticsViolentCrime.process()
#opticsOrdering=opticsViolentCrime.get_ordering()

dbscanLabels=pd.DataFrame(dbscanViolentCrime.labels_,columns=["dbscanLabels"])
#opticsLabels=pd.DataFrame(opticsViolentCrime.labels_,columns=["opticsLabels"])

immigrationViolentCrime=immigrationViolentCrime.join(dbscanLabels)
#immigrationViolentCrime=immigrationViolentCrime.join(opticsLabels)

#DBSCAN PLOT
sns.pairplot(x_vars=["violentCrimeRate"],y_vars=["admittedImmigrantRate"],
             data=immigrationViolentCrime, hue='dbscanLabels',size=10)  
#OPTICS PLOT
#sns.pairplot(x_vars=["violentCrimeRate"],y_vars=["admittedImmigrantRate"],
#             data=immigrationViolentCrime, hue='opticsLabels', size=10)

immigrationViolentCrime.to_csv("C:/Users/Will/Desktop/Stat256Final/results/Transformed/DBSCANViolent.csv")

#Immigration vs PropertyCrime
immigrationPropertyCrime=pd.DataFrame(newData["AggregatedData"]["admittedImmigrantRate"])
immigrationPropertyCrime=immigrationPropertyCrime.join(newData["AggregatedData"]["propertyCrimeRate"])
dbscanPropertyCrime=DBSCAN(eps=.0045, min_samples=10).fit(immigrationPropertyCrime)

clusterLabels=pd.DataFrame(dbscanViolentCrime.labels_,columns=["labels"])

immigrationPropertyCrime=immigrationPropertyCrime.join(clusterLabels)

sns.pairplot(x_vars=["propertyCrimeRate"], y_vars=["admittedImmigrantRate"], 
             data=immigrationPropertyCrime, hue="labels", size=10)

immigrationPropertyCrime.to_csv("C:/Users/Will/Desktop/Stat256Final/results/Transformed/DBSCANProperty.csv")
"""
CLUSTERING ANALYSIS
AggregatedStates
Outer Variables
"""
"""
KMEANS
"""
#Drop all of the outer columns and cluster based on all the inner variables
clusteringData=data["AggregatedData"].drop(columns=["admittedImmigrantRateStd","admittedImmigrantNorm",
                   "year","population", "murderManslaughterRate", "rapeRate", "robberyRate",
                   "aggravatedAssaultRate", "burglaryRate", "larcenyTheftRate", "motorVehicleTheftRate",
                   "laborForceRate", "employmentRate"])
kmeansOuterVariables=KMeans(n_clusters=3).fit(clusteringData)

clusterLabels=pd.DataFrame(kmeansOuterVariables.labels_,columns=["labels"])

clusteringData=clusteringData.join(clusterLabels)

#Output results of Kmeans clustering with all variables
file = open('C:/Users/Will/Desktop/Stat256Final/results/kmeansClusteringAllVariables.csv','w')
file.write(clusteringData.to_csv())
file.close()

sns.pairplot(x_vars=["violentCrimeRate"], y_vars=["admittedImmigrantRate"],
             data=clusteringData, hue="labels", size=10)

sns.pairplot(x_vars=["propertyCrimeRate"], y_vars=["admittedImmigrantRate"],
             data=clusteringData, hue="labels", size=10)
"""
DBSCAN
"""
clusteringData=data["AggregatedData"].drop(columns=["admittedImmigrantRateStd","admittedImmigrantNorm",
                   "year","population", "murderManslaughterRate", "rapeRate", "robberyRate",
                   "aggravatedAssaultRate", "burglaryRate", "larcenyTheftRate", "motorVehicleTheftRate",
                   "laborForceRate", "employmentRate"])
dbscanOuterVariables=DBSCAN(eps=.51, min_samples=5).fit(clusteringData)

clusterLabels=pd.DataFrame(dbscanOuterVariables.labels_,columns=["labels"])

clusteringData=clusteringData.join(clusterLabels)

sns.pairplot(x_vars=["violentCrimeRate"],y_vars=["admittedImmigrantRate"],
             data=clusteringData, hue='labels', size=10)

sns.pairplot(x_vars=["propertyCrimeRate"], y_vars=["admittedImmigrantRate"],
             data=clusteringData, hue='labels',size=10)

"""
KMEANS OUTER TRANSFORMED
"""
clusteringData=newData["AggregatedData"].drop(columns=OUTERDROPCOLS)

kmeansOuterVariables=KMeans(n_clusters=3).fit(clusteringData)
clusterLabels=pd.DataFrame(kmeansOuterVariables.labels_,columns=["outerLabels"])
clusteringData=clusteringData.join(clusterLabels)

"""
KMEANS INNER TRANSFORMED
"""
InnerData=newData["AggregatedData"].drop(columns=INNERDROPCOLS)

kmeansInnerVariables=KMeans(n_clusters=3).fit(InnerData)
clusterLabels=pd.DataFrame(kmeansInnerVariables.labels_,columns=["innerLabels"])
clusteringData=clusteringData.join(clusterLabels)

clusteringData.to_csv("C:/Users/Will/Desktop/Stat256Final/results/Transformed/kMeansBothModels.csv")
