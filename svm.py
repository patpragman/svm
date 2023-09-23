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
HOME_DIRECTORY = Path.home()
import argparse
parser = argparse.ArgumentParser(description="SVM Classifier Trainer Command Line Script!")
parser.add_argument("--target_path",
                    default=f"{HOME_DIRECTORY}/data/0.35_reduced_then_balanced",
                    type=str)
parser.add_argument("--specific_size", default=-1, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--quiet", default=False, action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()


    if args.specific_size < 0:
        sizes = [512, 256, 224, 128, 64]
    else:
        sizes = [args.specific_size]


    results_filename = f'{args.target_path}/results.md'
    if os.path.isfile(results_filename):
        os.remove(results_filename)  # delete the old one, make a new one!

    with open(results_filename, "w") as results_file:
        results_file.write("# Results for SVM:\n")

    folder_paths = [f"{args.target_path}/data_{size}" for size in sizes]
    for size, dataset_path in zip(sizes, folder_paths):
        model_title = f"SVM Model for {size} x {size} images"
        if not args.quiet:
            print(model_title)

        with open(results_filename, "a") as results_file:
            results_file.write(model_title)
            results_file.write("\n")

        # variables to hold our data
        data = []
        Y = []

        excluded = ["pkl", ".DS_Store", "md", "png"]
        classes = [str(klass).split("/")[-1] for klass in Path(dataset_path).iterdir()
                   if klass.is_dir()]



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
                    if "cm" in file:
                        continue
                    else:
                        img = io.imread(f"{root}/{file}", as_gray=True)

                        arr = np.asarray(img).reshape(size * size, )  # reshape into an array
                        data.append(arr)

                        Y.append(encoder(key))  # simple one hot encoding

        y = np.array(Y)
        X = np.array(data)

        # now we've loaded all the X values into a single array
        # and all the Y values into another one, let's do a train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                            random_state=args.seed)  # for consistency

        # grid search through the values
        # Define a parameter grid for hyperparameter tuning
        param_grid = {
            'C': [1, 10, 100],
            'kernel': ['linear', "rbf"],
        }

        # Create an SVM classifier
        svm_classifier = SVC()

        # Perform a grid search with cross-validation
        grid_search = GridSearchCV(estimator=svm_classifier,
                                   param_grid=param_grid,
                                   cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        # Get the best hyperparameters
        best_params = grid_search.best_params_
        if not args.quiet:
            print("Best Hyperparameters:", best_params)
        with open(results_filename, "a") as results_file:
            results_file.write(f"Best Hyperparameters:  {str(best_params)}")


        # Train an SVM classifier with the best hyperparameters
        best_svm_classifier = SVC(**best_params)
        best_svm_classifier.fit(X_train, y_train)

        y_pred = best_svm_classifier.predict(X_test)

        cr = classification_report(
            y_test, y_pred, target_names=[key for key in mapping.keys()], output_dict=True
        )

        # save it
        with open(f"{dataset_path}/svm_{size}.pkl", "wb") as file:
            pickle.dump(best_svm_classifier, file)

        display_and_save_cm(
            y_actual=y_test, y_pred=y_pred, labels=[key for key in mapping.keys()],
            name=f"SVM For Image Size {size}x{size}", output_path=f"{dataset_path}/cm.png", save_fig=True
        )

        status = f"{size}x{size} complete!"
        if not args.quiet:
            print(status)

            for d in [cr[k] for k in mapping.keys()]:
                print(pformat(d))

        with open(results_filename, "a") as results_file:
            results_file.write("\n")

            results_file.write(status)
            results_file.write(classification_report(
                y_test, y_pred, target_names=[key for key in mapping.keys()],
            ))
            results_file.write("\n")
