from sklearn.datasets import load_iris 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
#............ Utility Func. ...................#

# This Function replaces numerical values with the Corresponding ClassName of Target
def ClassConvert(x):
	dic = {0:'Iris-setosa', 1:'Iris-versicolor', 2:'Iris-virginica'}

	v = [dic.get(n, n) for n in x]
	return v
def ImportDataset_Iris():
    iris = load_iris() 
    return iris
 
def SplitDataset(iris_dataset): 

	# Separating the target variable 
	X = iris_dataset.data
	Y = iris_dataset.target

	# Splitting the dataset into train and test with test data being 30%
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4, random_state = 1) 
	
	return X, Y, X_train, X_test, y_train, y_test 
	
# Function to make predictions 
def prediction(X_test, clf_object): 

	# Predicton on test with giniIndex 
	y_pred = clf_object.predict(X_test) 
	print("Predicted values:") 
	print(ClassConvert(y_pred)) 
	return y_pred 
	
def Accuracy_Report(y_test, y_pred): 
	
	print("Confusion Matrix: \n",confusion_matrix(y_test, y_pred)) 
	print ("Accuracy : ",accuracy_score(y_test,y_pred)*100) 
	print("Report : \n",classification_report(y_test, y_pred)) 

#...................................................#


#............ Decision Tree Code ...................#

# Perform training with giniIndex. 
def Gini_Train(X_train, y_train): 
 
	clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5) 

	# Performing training 
	clf_gini.fit(X_train, y_train) 
	return clf_gini 
	
# Perform training with entropy. 
def Entropy_Train(X_train, y_train):
	clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5) 

	# Performing training 
	clf_entropy.fit(X_train, y_train)
	return clf_entropy 


#.............................................#


def DecisionTree(): 
	 
	data = ImportDataset_Iris() 
	X, Y, X_train, X_test, y_train, y_test = SplitDataset(data) 
	clf_gini = Gini_Train(X_train, y_train) 
	clf_entropy = Entropy_Train(X_train, y_train) 
	 
	print("Results Using Gini Index:") 
	
	# Prediction using gini 
	y_pred_gini = prediction(X_test, clf_gini) 
	Accuracy_Report(y_test, y_pred_gini) 
	
	print("Results Using Entropy:") 

	# Prediction using entropy 
	y_pred_entropy = prediction(X_test, clf_entropy) 
	Accuracy_Report(y_test, y_pred_entropy) 
	
if __name__=="__main__": 
	DecisionTree() 