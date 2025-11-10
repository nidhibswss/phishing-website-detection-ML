from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler #for scaling 
from sklearn.model_selection import train_test_split #traintestsplit
from sklearn.tree import DecisionTreeClassifier #  Decision Tree Classifier
from sklearn.linear_model import LogisticRegression #logisitic regression
from sklearn.neighbors import KNeighborsClassifier #knn 
from sklearn.svm import SVC #svm 
from sklearn.naive_bayes import GaussianNB #naive bayes

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import plot_tree
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

#for feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


#--- WEBSITE PHISING DETECTION USING MACHINE LEARNING --------


# Load ARFF file
data, meta = arff.loadarff('PhishingData.arff')


# Convert to DataFrame (df)
df = pd.DataFrame(data)

# DATA INSPECTION - explorative data analysis on dataset 

print(df.shape)              # Check dimensions = 1353 rows (instances), 10 columns 
print(df.columns)            # printing the column names to see the name of all the features
print(df.dtypes)             # Check data types = data types shows object for now - needs conversion
print(df.head())               
 
print()
print()
print()
print('end')

# -------- DATA CLEANING AND PREPROCESSING ----------

# Convert all columns to integers
df = df.astype(int)

print(df.dtypes)            #verifying if all the columns are integer 
print(df.head())
print(df.describe())


# removing emptty rows and duplicates,and checking for null
df.dropna(inplace = True)
df.drop_duplicates(inplace = True)
print(df.isnull().sum())


#normalize the target varibale
df['Result'] = df['Result'].replace(-1, 0)


#-----FEATURE EXTRACTION-------
    
X = df.drop(columns=['Result']) # removing result column from  the remaining column and storing in X 
Y = df['Result'] # target variable 



#-------SCALING FEATURES---------
   
scaler = StandardScaler()
standardized_x = scaler.fit_transform(X)
#print(standardized_x)


#------CORRELATION MATRIX - heatmap visualization ---------

plt.figure(figsize=(7, 5))
correlation_matrix = df.corr().round(2)  
sns.heatmap(data=correlation_matrix, annot=True, linewidths=0.5, cmap='coolwarm')


#---------TRAIN TEST SPLIT----------- 

X_train, X_test, Y_train, Y_test = train_test_split(standardized_x, Y, test_size=0.3,random_state=42)



#----ALGORITHMS : ----------

#----###### #model 1 : decision tree classifier #####-----------
    

model1 = DecisionTreeClassifier(criterion='entropy',max_depth=3)
model1.fit(X_train, Y_train)
decisiontree_pred = model1.predict(X_test)



# decision tree 
plt.figure(figsize=(12, 5))
plot_tree(model1,feature_names=X.columns)
plt.title("Decision Tree Structure (Depth=3)")
plt.show()

print()
print()
# Print evaluation report : 
print("Decision Tree Classifier Performance")
print(f"Accuracy : {accuracy_score(Y_test,  decisiontree_pred):.4f}")
print(f"Precision: {precision_score(Y_test,  decisiontree_pred):.4f}")
print(f"Recall   : {recall_score(Y_test, decisiontree_pred):.4f}")
print(f"F1 Score : {f1_score(Y_test, decisiontree_pred):.4f}")


#----###### model 2: logistic regression ####---------------
       
model2 = LogisticRegression(max_iter=1000)
model2.fit(X_train, Y_train)
logisticregression_pred = model2.predict(X_test)

print()
print()
print()
#evaluation report : 
print("logistic regression Performance")
print(f"Accuracy : {accuracy_score(Y_test, logisticregression_pred):.4f}")
print(f"Precision: {precision_score(Y_test, logisticregression_pred):.4f}")
print(f"Recall   : {recall_score(Y_test, logisticregression_pred):.4f}")
print(f"F1 Score : {f1_score(Y_test,logisticregression_pred):.4f}")


#------###### model 3 : k-nearest neighbor (KNN) ######----------------------------

model3 = KNeighborsClassifier(n_neighbors=5)
model3.fit(X_train, Y_train)
knn_pred = model3.predict(X_test)


print()
print()
#evaluation report 
print("K-nearest neighbor (KNN) Performance")
print(f"Accuracy : {accuracy_score(Y_test, knn_pred):.4f}")
print(f"Precision: {precision_score(Y_test, knn_pred):.4f}")
print(f"Recall   : {recall_score(Y_test, knn_pred):.4f}")
print(f"F1 Score : {f1_score(Y_test,knn_pred):.4f}")



#----###### Model 4: Support Vector Machine (Linear Kernel) ####-------------
    
model4 = SVC(kernel='linear')
model4.fit(X_train, Y_train)
svm_pred = model4.predict(X_test)


print()
print()
# Evaluation report : 
print("Support Vector Machine (Linear Kernel) Performance")
print(f"Accuracy : {accuracy_score(Y_test, svm_pred):.4f}")
print(f"Precision: {precision_score(Y_test, svm_pred):.4f}")
print(f"Recall   : {recall_score(Y_test, svm_pred):.4f}")
print(f"F1 Score : {f1_score(Y_test,svm_pred):.4f}")



##----###### Model 5: Gaussian Naive Bayes ###---------

model5 = GaussianNB()
model5.fit(X_train, Y_train)
gnb_pred = model5.predict(X_test)


print()
print()
# Evaluation report : 
print("Gaussian Naive Bayes Classifier Performance")
print(f"Accuracy : {accuracy_score(Y_test, gnb_pred):.4f}")
print(f"Precision: {precision_score(Y_test, gnb_pred):.4f}")
print(f"Recall   : {recall_score(Y_test, gnb_pred):.4f}")
print(f"F1 Score : {f1_score(Y_test,gnb_pred):.4f}")

#######################################################################################################################################

#--- WITH FEATURE SELECTION using SelectKBest-----------

feature_selection = SelectKBest(score_func=f_classif, k=3) #select only 3 features
X_new = feature_selection.fit_transform(X, Y)
selected_mask = feature_selection.get_support(indices=False)
print()
print(selected_mask) #shows list of boolean values true false 
selected_columns = X.columns[selected_mask]
print()
print()
print("Top 3 features selected by SelectKBest algorithm :")
print()
print(selected_columns)

# ----
#new dataframe with ONLY the selected columns 
X_selected = X[selected_columns] 


#scaling new x 
scaler = StandardScaler()
standardized_x_2 = scaler.fit_transform(X_selected)
#print(standardized_x_2)

# New Train-test split
X_train_2, X_test_2, Y_train_2, Y_test_2 = train_test_split(standardized_x_2 , Y, test_size=0.2, random_state=42)



# repeating all the 5 algorithms with the feature seleciton
# Re-train all models using only selected features
print()
print("----- model 6-10 with feature selection ------")

#----###### #model 6 : decision tree classifier AFTER feature selection #####-----------

model6 = DecisionTreeClassifier(criterion='entropy',max_depth=3)
model6.fit(X_train_2, Y_train_2)
decisiontree_pred_2 = model6.predict(X_test_2)

# Print evaluation report after feature selection  : 
print()
print("Decision tree classifier performance after feature selection ")
print(f"Accuracy :{accuracy_score(Y_test_2, decisiontree_pred_2):.4f}")
print(f"Precision: {precision_score(Y_test_2, decisiontree_pred_2):.4f}")
print(f"Recall   : {recall_score(Y_test_2, decisiontree_pred_2):.4f}")
print(f"F1 Score : {f1_score(Y_test_2, decisiontree_pred_2):.4f}")


#----###### model 7: logistic regression AFTER feature selection ####---------------
       
model7 = LogisticRegression(max_iter=1000)
model7.fit(X_train_2, Y_train_2)
logisticregression_pred_2 = model7.predict(X_test_2)

print()
print()
print()
#evaluation report after feature selection : 
print("logistic regression Performance after feature selection")
print(f"Accuracy : {accuracy_score(Y_test_2, logisticregression_pred_2):.4f}")
print(f"Precision: {precision_score(Y_test_2, logisticregression_pred_2):.4f}")
print(f"Recall   : {recall_score(Y_test_2, logisticregression_pred_2):.4f}")
print(f"F1 Score : {f1_score(Y_test_2,logisticregression_pred_2):.4f}")

#------###### model 8 : k-nearest neighbor (KNN) AFTER feature selection ######----------------------------

model8 = KNeighborsClassifier(n_neighbors=5)
model8.fit(X_train_2, Y_train_2)
knn_pred_2 = model8.predict(X_test_2)


print()
print()
#evaluation report after feature selection : 
print("K-nearest neighbor (KNN) Performance after feature selection")
print(f"Accuracy : {accuracy_score(Y_test_2, knn_pred_2):.4f}")
print(f"Precision: {precision_score(Y_test_2, knn_pred_2):.4f}")
print(f"Recall   : {recall_score(Y_test_2, knn_pred_2):.4f}")
print(f"F1 Score : {f1_score(Y_test_2,knn_pred_2):.4f}")



#----###### Model 9: Support Vector Machine (Linear Kernel) AFTER feature selection ####-------------
    
model9 = SVC(kernel='linear')
model9.fit(X_train_2, Y_train_2)
svm_pred_2 = model9.predict(X_test_2)


print()
print()
#evaluation report after feature selection : 
print("Support Vector Machine (Linear Kernel) Performance after feature selection")
print(f"Accuracy : {accuracy_score(Y_test_2, svm_pred_2):.4f}")
print(f"Precision: {precision_score(Y_test_2, svm_pred_2):.4f}")
print(f"Recall   : {recall_score(Y_test_2, svm_pred_2):.4f}")
print(f"F1 Score : {f1_score(Y_test_2,svm_pred_2):.4f}")



##----###### Model 10: Gaussian Naive Bayes AFTER feature selection ###---------

model10 = GaussianNB()
model10.fit(X_train_2, Y_train_2)
gnb_pred_2 = model10.predict(X_test_2)


print()
print()
#evaluation report after feature selection : 
print("Gaussian Naive Bayes Classifier Performance after feature selection")
print(f"Accuracy : {accuracy_score(Y_test_2, gnb_pred_2):.4f}")
print(f"Precision: {precision_score(Y_test_2, gnb_pred_2):.4f}")
print(f"Recall   : {recall_score(Y_test_2, gnb_pred_2):.4f}")
print(f"F1 Score : {f1_score(Y_test_2,gnb_pred_2):.4f}")














