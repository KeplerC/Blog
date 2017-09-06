## Training(chapter 4)
This is covered more by professor Andrews Ng, so I will study cs229 after this one. 

## SVM
SVM has hard classification and soft classification. Hard Classification is linearly separable, soft classification gives margin violations. 

The python code is simple 
```python 
svm_clf = Pipeline((
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge")), #applies Stochstic Gradient Descent rather than batch
))
svm_clf.fit(X_scaled, y)
```
and can add polynomial features by 
```python 
    ("poly_features", PolynomialFeatures(degree=3))
```
and then do linear SVC

>Adding polynomial features is simple to implement and can work great with all sorts
of Machine Learning algorithms (not just SVMs), but at a low polynomial degree it
cannot deal with very complex datasets, and with a high polynomial degree it creates
a huge number of features, making the model too slow.

### Gaussian Radial Basis  
```python 
("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
```

#### Computational complexity 
linear SVC m*n
SGD m*n
SVC m^2~m^3 * n

Decision and prediction 

decision function = 0 if theta^T . x < 0 and vise versa
The goal is to minimize ||theta|| conditioned on >1 on positive cases and <-1 on negative instances 

#### Hinge Loss
equivalent to max(0, 1 – t)

## Decision Trees
we can train a decision tree by 
```python 
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)
```
and visualize by 
```python 
from sklearn.tree import export_graphviz
export_graphviz(
tree_clf,
out_file=image_path("iris_tree.dot"),
feature_names=iris.feature_names[2:],
class_names=iris.target_names,
rounded=True,
filled=True
)
```
dot -Tpng iris_tree.dot -o iris_tree.png

