import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Cargar los conjuntos de datos
iris = load_iris()
wine = load_wine()
cancer = load_breast_cancer()

datasets = {
    "Iris": iris,
    "Wine": wine,
    "Breast Cancer": cancer
}
def evaluate_knn(dataset, dataset_name):
    X = dataset.data
    y = dataset.target

    # Hold-Out 70/30
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 1-NN Clasificador
    knn_1 = KNeighborsClassifier(n_neighbors=1)
    knn_1.fit(X_train, y_train)
    y_pred_1 = knn_1.predict(X_test)

    accuracy_1 = accuracy_score(y_test, y_pred_1)
    cm_1 = confusion_matrix(y_test, y_pred_1)
    
    # Mejor valor de K usando 10-Fold Cross Validation
    k_range = range(1, 31)
    k_scores = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
        k_scores.append(scores.mean())

    best_k = k_range[np.argmax(k_scores)]
    
    # Clasificador K-NN con el mejor K
    knn_best = KNeighborsClassifier(n_neighbors=best_k)
    knn_best.fit(X_train, y_train)
    y_pred_best = knn_best.predict(X_test)

    accuracy_best = accuracy_score(y_test, y_pred_best)
    cm_best = confusion_matrix(y_test, y_pred_best)
    
    # Resultados
    print(f"\n{dataset_name} Dataset Results:")
    print(f"1-NN clasificador con Accuracy (Hold-Out): {accuracy_1:.2f}")
    print(f"1-NN clasificador con Confusion Matrix (Hold-Out):\n{cm_1}")

    print(f"Best K found: {best_k} con Accuracy (10-Fold CV): {max(k_scores):.2f}")
    print(f"K-NN clasificador con K={best_k} Accuracy (Hold-Out): {accuracy_best:.2f}")
    print(f"K-NN clasificador con K={best_k} Confusion Matrix (Hold-Out):\n{cm_best}")

    # 10-Fold Cross Validation con el mejor K
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    cv_accuracy_scores = cross_val_score(knn_best, X, y, cv=kf, scoring='accuracy')
    
    print(f"K-NN clasificador con K={best_k} Accuracy (10-Fold CV): {cv_accuracy_scores.mean():.2f}")

for name, dataset in datasets.items():
    evaluate_knn(dataset, name)