import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pandas as pd

data = pd.read_csv("car.data")
# print(data.head())

# Give Each label a numeric value so it can be worked as a classifier.
label = preprocessing.LabelEncoder()
buying = label.fit_transform(list (data["buying"]))
maint = label.fit_transform(list (data["maint"]))
door = label.fit_transform(list (data["door"]))
persons = label.fit_transform(list (data["persons"]))
lug_boot = label.fit_transform(list (data["lug_boot"]))
safety = label.fit_transform(list (data["safety"]))
cls = label.fit_transform(list (data["class"]))

predict = "class"

# X for Features & Y for Labels.
X = list(zip(buying, maint, door, persons, lug_boot, safety))
Y = list(cls)

# Training & Testing Values and we used 0.1 to minimize the sacrifice of the data.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# Begin Testing data.
model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

# Get the predictions and the distances between data.
for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    distances = model.kneighbors([x_test[x]], 9, True)
    print("Distances: ", distances)