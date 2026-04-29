
from sklearn.metrics import roc_auc_score

def evaluate(model, dataloader):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, features, y in dataloader:
            preds = model(x, features)

            y_true.extend(y.tolist())
            y_pred.extend(preds.squeeze().tolist())

    score = roc_auc_score(y_true, y_pred)
    print("ROC-AUC:", score)
