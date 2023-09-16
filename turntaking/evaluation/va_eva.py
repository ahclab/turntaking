import torch
from sklearn.metrics import roc_auc_score

def safe_roc_auc_score(y_true, y_score, **kwargs):
    try:
        return roc_auc_score(y_true, y_score, **kwargs)
    except ValueError:
        return 0.5

def compute_confusion_matrix(preds, labels):
    """
    Compute the confusion matrix for the given predictions and labels.
    
    :param preds: Model's predictions, shape: (batch_size, num_classes)
    :param labels: True labels, shape: (batch_size)
    :return: Confusion matrix of shape (num_classes, num_classes)
    """
    preds = preds.to("cpu")  # Get the device of the preds tensor (could be 'cpu', 'cuda:0', etc.)
    num_classes = preds.size(-1)
    
    # Get the predicted classes
    _, predicted_classes = preds.max(dim=-1)
    
    # Ensure labels are on the same device as preds
    labels = labels.to("cpu")
    
    # Initialize confusion matrix on the same device
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64, device="cpu")
    
    # Populate confusion matrix
    for t, p in zip(labels, predicted_classes):
        confusion_matrix[t, p] += 1

    return confusion_matrix

def compute_classification_metrics(preds, labels, top_k=5):
    """
    Compute accuracy, precision, recall, f1-score, top-k accuracy, multi-class AUC, and micro/macro-averaged metrics.
    """
    preds = preds.to("cpu")
    labels = labels.to("cpu")
    num_classes = preds.size(-1)
    _, predicted_classes = preds.max(dim=-1)
    labels_squeezed = labels.squeeze(dim=-1)

    # Initialize metrics
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    TP = torch.zeros(num_classes)
    FP = torch.zeros(num_classes)
    FN = torch.zeros(num_classes)
    TN = torch.zeros(num_classes)

    # Populate confusion matrix
    for t, p in zip(labels_squeezed.view(-1), predicted_classes.view(-1)):
        confusion_matrix[t, p] += 1

    # Compute TP, FP, TN, FN for each class
    for i in range(num_classes):
        TP[i] = confusion_matrix[i, i]
        FP[i] = confusion_matrix[:, i].sum() - TP[i]
        FN[i] = confusion_matrix[i, :].sum() - TP[i]
        TN[i] = confusion_matrix.sum() - (FP[i] + FN[i] + TP[i])

    epsilon = 1e-10  # a small value to avoid division by zero
    # Macro-average metrics
    precision_macro = TP / (TP + FP + epsilon)
    recall_macro = TP / (TP + FN + epsilon)
    f1_score_macro = 2 * (precision_macro * recall_macro) / (precision_macro + recall_macro + epsilon)

    # Micro-average metrics
    precision_micro = TP.sum() / (TP.sum() + FP.sum())
    recall_micro = TP.sum() / (TP.sum() + FN.sum())
    f1_score_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro)

    # Top-k accuracy
    _, top_k_preds = preds.topk(top_k, dim=-1)
    correct_top_k = top_k_preds.eq(labels_squeezed.unsqueeze(-1).expand_as(top_k_preds))
    top_k_accuracy = correct_top_k.any(dim=-1).float().mean().item()
    
    # Multi-class AUC
    # Convert predictions to probability via softmax
    probs = torch.nn.functional.softmax(preds, dim=-1)
    one_hot_labels = torch.nn.functional.one_hot(labels_squeezed, num_classes=num_classes)

    # Reshape the tensors for AUC calculation
    one_hot_labels_reshaped = one_hot_labels.view(-1, num_classes).cpu().numpy()
    probs_reshaped = probs.view(-1, num_classes).cpu().numpy()

    multi_class_auc = safe_roc_auc_score(one_hot_labels_reshaped, probs_reshaped, multi_class='ovr')

    return {
        'accuracy': (TP.sum() / confusion_matrix.sum()).item(),
        'top_k_accuracy': top_k_accuracy,
        # 'macro_precision': precision_macro.mean().item(),
        # 'macro_recall': recall_macro.mean().item(),
        # 'macro_f1': f1_score_macro.mean().item(),
        # 'micro_precision': precision_micro.item(),
        # 'micro_recall': recall_micro.item(),
        # 'micro_f1': f1_score_micro.item(),
        # 'multi_class_auc': multi_class_auc
    }


def compute_regression_metrics(preds, labels):
    """
    Compute MAE, MSE, and RMSE for regression tasks.
    
    :param preds: Model's predictions, shape: (batch_size, num_data_points, feature_dim1, feature_dim2)
    :param labels: True labels, shape: (batch_size, num_data_points, 1, feature_dim1, feature_dim2)
    :return: Dictionary containing MAE, MSE, and RMSE
    """
    
    # Removing singleton dimensions
    labels_squeezed = labels.squeeze(dim=2)
    
    # Compute differences
    diff = preds - labels_squeezed
    
    # Calculate metrics
    mae = torch.abs(diff).mean()
    mse = (diff**2).mean()
    rmse = torch.sqrt(mse)
    
    return {
        'MAE': mae.item(),
        'MSE': mse.item(),
        'RMSE': rmse.item()
    }


def compute_comparative_metrics(preds, labels):
    """
    Compute MAE, MSE, and RMSE for regression tasks.
    
    :param preds: Model's predictions, shape: (batch_size, num_data_points, 1)
    :param labels: True labels, shape: (batch_size, num_data_points, 1)
    :return: Dictionary containing MAE, MSE, and RMSE
    """
    
    # Compute differences
    diff = preds - labels
    
    # Calculate metrics
    mae = torch.abs(diff).mean()
    mse = (diff**2).mean()
    rmse = torch.sqrt(mse)
    
    return {
        'MAE': mae.item(),
        'MSE': mse.item(),
        'RMSE': rmse.item()
    }
