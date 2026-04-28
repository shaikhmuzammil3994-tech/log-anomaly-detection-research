from sklearn.metrics import roc_auc_score

def evaluate(y_true, y_pred):
    score = roc_auc_score(y_true, y_pred)
    print("ROC-AUC Score:", score)

if __name__ == "__main__":
    # dummy example
    y_true = [0, 1, 0, 1]
    y_pred = [0.1, 0.9, 0.2, 0.8]

    evaluate(y_true, y_pred)
