from sklearn.datasets import load_iris 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB
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

#............ Bayes Classifier Code .........#

def Bayes_Train(X_train, Y_train):
    BayesClassifier = GaussianNB()
    BayesClassifier.fit(X_train, Y_train)
    return BayesClassifier

#...................................................#

def BayesClassifiers(): 

	data = ImportDataset_Iris() 
	X, Y, X_train, X_test, y_train, y_test = SplitDataset(data) 
	BC = Bayes_Train(X_train, y_train)  
	
	print("Results BayesClassifier:")
	y_pred_bayes = prediction(X_test, BC) 
	Accuracy_Report(y_test, y_pred_bayes) 
	
	
if __name__=="__main__":
	BayesClassifiers()