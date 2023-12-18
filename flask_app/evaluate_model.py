import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

def evaluate_model(model, X_test, y_test, model_name, results_dict):
    predictions = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc_score = roc_auc_score(y_test, proba)

    results_dict[model_name] = [acc, precision, recall, f1, auc_score]

    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(5,5))
    ConfusionMatrixDisplay(cm).plot(values_format='d')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure(figsize=(5,5))
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_score).plot()
    plt.title(f'ROC Curve for {model_name}')
    plt.show()
