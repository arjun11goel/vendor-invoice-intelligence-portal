from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

def train_decision_tree(X_train, y_train):

    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 4, 5, 6],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 2, 5]
    }

    dt = DecisionTreeClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=dt,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    return grid_search


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))