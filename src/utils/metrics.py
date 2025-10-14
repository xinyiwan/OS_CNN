import numpy as np
import sklearn.metrics as sk
import torch
from typing import Dict, List, Tuple, Any, Union
from scipy.special import softmax

def compute_expected_calibration_error(probs: Union[np.ndarray, torch.Tensor], 
                                     labels: Union[np.ndarray, torch.Tensor], 
                                     n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE) - measures how well calibrated the probabilities are.
    
    Args:
        probs: Probability scores (n_samples, n_classes) or (n_samples,) for binary
        labels: True labels (n_samples,)
        n_bins: Number of bins to use for calibration
    
    Returns:
        ECE score (lower is better)
    """
    # Convert to tensors for consistent processing
    if isinstance(probs, np.ndarray):
        probs = torch.from_numpy(probs)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    
    # For multiclass, use maximum probability as confidence
    if probs.dim() > 1 and probs.shape[1] > 1:
        confidences, predictions = torch.max(probs, dim=1)
    else:
        # Binary classification
        confidences = probs.squeeze()
        predictions = (confidences > 0.5).long()
    
    # Ensure labels are same type as predictions
    labels = labels.long()
    
    # Check if we have valid data
    if confidences.numel() == 0:
        return float('nan')
    
    # Compute ECE
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            # Calculate accuracy and average confidence in this bin
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            # Add to ECE
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.item()

def compute_classification_metrics(probs: Union[np.ndarray, torch.Tensor], 
                                 labels: Union[np.ndarray, torch.Tensor], 
                                 average: str = 'macro') -> Dict[str, float]:
    """
    Compute comprehensive classification metrics from probabilities and labels.
    
    Args:
        probs: Probability scores (n_samples, n_classes) or (n_samples,) for binary
        labels: True labels (n_samples,)
        average: Averaging method for multiclass metrics ('macro', 'micro', 'weighted')
    
    Returns:
        Dictionary containing various metrics
    """
    # Convert to numpy arrays if they're tensors
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Ensure probs is 2D (samples, classes)
    if probs.ndim == 1:
        # Binary classification - convert to 2D
        probs = np.column_stack([1 - probs, probs])
    
    # Get predicted classes
    pred_classes = np.argmax(probs, axis=1)
    
    # For binary classification
    if probs.shape[1] == 2:
        positive_probs = probs[:, 1]  # Probabilities for positive class
        
        metrics = {
            'auroc': sk.roc_auc_score(labels, positive_probs),
            'auprc': sk.average_precision_score(labels, positive_probs),
            'accuracy': sk.accuracy_score(labels, pred_classes),
            'precision': sk.precision_score(labels, pred_classes, zero_division=0),
            'recall': sk.recall_score(labels, pred_classes, zero_division=0),
            'f1': sk.f1_score(labels, pred_classes, zero_division=0),
            'brier_score': sk.brier_score_loss(labels, positive_probs),
        }
        
        # Confusion matrix components
        tn, fp, fn, tp = sk.confusion_matrix(labels, pred_classes).ravel()
        metrics.update({
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn,
            'true_positive': tp,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        })
    
    else:
        # Multiclass classification
        metrics = {
            'auroc': sk.roc_auc_score(labels, probs, multi_class='ovr', average=average),
            'accuracy': sk.accuracy_score(labels, pred_classes),
            'precision': sk.precision_score(labels, pred_classes, average=average, zero_division=0),
            'recall': sk.recall_score(labels, pred_classes, average=average, zero_division=0),
            'f1': sk.f1_score(labels, pred_classes, average=average, zero_division=0),
        }
    
    # Additional metrics that work for both binary and multiclass
    try:
        metrics['log_loss'] = sk.log_loss(labels, probs)
    except:
        metrics['log_loss'] = float('inf')
    
    return metrics

def compute_confidence_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute confidence-related metrics like Expected Calibration Error (ECE)
    """
    if probs.ndim == 1:
        probs = np.column_stack([1 - probs, probs])
    
    pred_classes = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    correct = (pred_classes == labels).astype(float)
    
    # Simple confidence metrics
    metrics = {
        'avg_confidence': float(np.mean(confidence)),
        'avg_confidence_correct': float(np.mean(confidence[correct == 1])) if np.sum(correct) > 0 else 0.0,
        'avg_confidence_incorrect': float(np.mean(confidence[correct == 0])) if np.sum(correct == 0) > 0 else 0.0,
        'confidence_gap': float(np.mean(confidence[correct == 1]) - np.mean(confidence[correct == 0])) 
                         if (np.sum(correct) > 0 and np.sum(correct == 0) > 0) else 0.0,
    }
    
    return metrics

def aggregate_metrics_across_folds(all_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple cross-validation folds
    
    Args:
        all_metrics: List of metric dictionaries from each fold
    
    Returns:
        Dictionary with mean and std for each metric
    """
    if not all_metrics:
        return {}
    
    # Get all metric names
    metric_names = all_metrics[0].keys()
    
    aggregated = {}
    for metric_name in metric_names:
        values = [metrics[metric_name] for metrics in all_metrics if metric_name in metrics]
        if values:
            aggregated[f'{metric_name}_mean'] = float(np.mean(values))
            aggregated[f'{metric_name}_std'] = float(np.std(values))
            aggregated[f'{metric_name}_min'] = float(np.min(values))
            aggregated[f'{metric_name}_max'] = float(np.max(values))
    
    return aggregated

__all__ = ['compute_classification_metrics', 'compute_confidence_metrics', 'aggregate_metrics_across_folds']