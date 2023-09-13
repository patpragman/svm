#  Logistic Regression Model using sci-kit learn
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from skimage import color, io
import os
import pickle
from cm_handler import display_and_save_cm
from pprint import pformat
from pathlib import Path

sizes = [1024, 512, 256, 224, 128, 64]
sizes.reverse()  # start with the smaller problem

results_filename = 'results.md'
if os.path.isfile(results_filename):
    os.remove(results_filename)  # delete the old one, make a new one!

with open(results_filename, "w") as results_file:
    results_file.write("# Results for SVM:\n")

folder_paths = [f"{Path.home()}/tx_data/data_{size}" for size in sizes]
for size, dataset_path in zip(sizes, folder_paths):
    print(pformat({"msg": f"SVM Model for {size} x {size} images"}))

    # variables to hold our data
    data = []
    Y = []

    classes = os.listdir(dataset_path)

    if ".DS_Store" in classes:
        # for macs
        classes.remove(".DS_Store")

    for klass in classes:
        if ".pkl" in klass:
            classes.remove(klass)

    # create a mapping from the classes to each number class and demapping
    mapping = {n: i for i, n in enumerate(classes)}
    demapping = {i: n for i, n in enumerate(classes)}

    # now create an encoder
    encoder = lambda s: mapping[s]
    decoder = lambda i: demapping[i]

    # now walk through and load the data in the containers we constructed above
    for root, dirs, files in os.walk(dataset_path):

        for file in files:
            if ".JPEG" in file.upper() or ".JPG" in file.upper() or ".PNG" in file.upper():
                key = root.split("/")[-1]
                img = io.imread(f"{root}/{file}", as_gray=True)
                arr = np.asarray(img).reshape(size * size, )  # reshape into an array
                data.append(arr)

                Y.append(encoder(key))  # simple one hot encoding

    y = np.array(Y)
    X = np.array(data)

    # now we've loaded all the X values into a single array
    # and all the Y values into another one, let's do a train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=42)  # for consistency

    # grid search through the values
    # Define a parameter grid for hyperparameter tuning
    param_grid = {
        'C': [1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': [0.001, 0.01, 0.1],
    }

    # Create an SVM classifier
    svm_classifier = SVC()

    # Perform a grid search with cross-validation
    grid_search = GridSearchCV(estimator=svm_classifier,
                               param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Train an SVM classifier with the best hyperparameters
    best_svm_classifier = SVC(**best_params)
    best_svm_classifier.fit(X_train, y_train)

    y_pred = best_svm_classifier.predict(X_test)

    cr = classification_report(
        y_test, y_pred, target_names=[key for key in mapping.keys()], output_dict=True
    )

    # save it
    with open(f"{dataset_path}/log_reg_{size}.pkl", "wb") as file:
        pickle.dump(best_svm_classifier, file)

    display_and_save_cm(
        y_actual=y_test, y_pred=y_pred, labels=[key for key in mapping.keys()],
        name=f"SVM For Image Size {size}x{size}"
    )

    print(pformat({"msg": f"{size}x{size} complete!"}))

    for d in [cr[k] for k in mapping.keys()]:
        print(pformat(d))

    with open(results_filename, "a") as outfile:
        outfile.write(f"{size}x{size} images\n")
        outfile.write(classification_report(
            y_test, y_pred, target_names=[key for key in mapping.keys()]
        ))
        outfile.write("\n")
