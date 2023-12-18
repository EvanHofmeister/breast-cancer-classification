from sklearn.model_selection import GridSearchCV

def train_model(model, param_grid, X_train, y_train, cv_folds):
    grid_search = GridSearchCV(model, param_grid, cv=cv_folds, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_
