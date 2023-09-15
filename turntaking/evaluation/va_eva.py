import torch

def compute_classification_metrics(preds, labels):
    """
    Compute accuracy, precision, recall, f1-score and confusion matrix for the given predictions and labels.
    
    :param predictions: Model's predictions, shape: (batch_size, num_frames, num_classes)
    :param labels: True labels, shape: (batch_size, num_frames, 1)
    :return: Dictionary containing accuracy, precision, recall, f1-score and confusion matrix
    """
    num_classes = preds.size(-1)
    
    # Get the predicted classes
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

    # Compute TP, FP, TN, FN
    for i in range(num_classes):
        TP[i] = confusion_matrix[i, i]
        FP[i] = confusion_matrix[:, i].sum() - TP[i]
        FN[i] = confusion_matrix[i, :].sum() - TP[i]
        TN[i] = confusion_matrix.sum() - (FP[i] + FN[i] + TP[i])
    
    # Compute precision, recall and F1-score
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    # Compute accuracy
    accuracy = TP.sum() / confusion_matrix.sum()
    
    return {
        'accuracy': accuracy.item(),
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'f1_score': f1_score.mean().item(),
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
