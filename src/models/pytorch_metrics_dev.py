'''
Common evaluation metrics used in uncertainty quantification
'''
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
import sklearn.metrics as sk
import numpy as np
import matplotlib.pyplot as plt

class Metrics():
    def __init__(self, probs, labels, num_bins, device=None, use_gp=False):

        self.probs = probs.clone().detach().to(device)
        self.labels = labels.clone().detach().to(device)
        self.num_bins = num_bins
        self.use_gp = use_gp
        self.given_k = self.probs.shape[-1]
        self.k = max(2, self.given_k)


        # Adjust probabilities if k < 2
        if self.given_k < 2:
            self.probs = torch.cat([1. - self.probs, self.probs], dim=-1)[:, -self.k:]

        # Get prediction probabilities
        if use_gp:    
            p_softmax = torch.softmax(self.probs, dim=-1)
            self.pred_probs = torch.max(p_softmax, dim=-1).values
        else:
            self.pred_probs = torch.max(self.probs, dim=-1).values

        # Bin indices for ECE calculation
        self.bin_indices = torch.bucketize(
            self.pred_probs,
            torch.linspace(0.0, 1.0, steps=self.num_bins + 1, device=device)
        ) - 1

    def compute(self):
        """Returns: acc, ece, nll, msp"""
        # Get predicted labels
        if self.use_gp:
            p_softmax = torch.softmax(self.probs, dim=-1)
            pred_labels = torch.argmax(p_softmax, dim=-1)
        else:
            pred_labels = torch.argmax(self.probs, dim=-1)
        
        correct_preds = (pred_labels == self.labels).float()

        # Bin calculations
        batch_counts = torch.zeros(self.num_bins, device=self.probs.device)
        batch_acc_sums = torch.zeros_like(batch_counts)
        batch_conf_sums = torch.zeros_like(batch_counts)

        for i in range(self.num_bins):
            mask = self.bin_indices == i
            batch_counts[i] = mask.float().sum()
            batch_acc_sums[i] = correct_preds[mask].sum()
            batch_conf_sums[i] = self.pred_probs[mask].sum()

        # Remove empty bins
        non_empty = batch_counts > 0
        counts = batch_counts[non_empty]
        accs = batch_acc_sums[non_empty] / counts
        confs = batch_conf_sums[non_empty] / counts

        # Compute metrics
        if self.use_gp:
            nll = F.cross_entropy(self.probs, self.labels)
            msp = torch.max(torch.softmax(self.probs, dim=-1), dim=-1).values
        else:
            nll = F.nll_loss(torch.log(self.probs), self.labels.long())
            msp = torch.max(self.probs, dim=-1).values

        acc = balanced_accuracy_score(
            self.labels.cpu().numpy(), 
            pred_labels.cpu().numpy()
        )
        ece = torch.sum((counts / counts.sum()) * torch.abs(accs - confs))
        
        return acc, ece.item(), nll.item(), msp
    
    
def compute_global_ece(confidences, labels, n_bins=10):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()


# Evaluate AUC, AUPR, and AUAC between the test set and OOD set
def fd_metrics(msp_ind, msp_fd):
    """
    Evaluate failure detection performance between in-distribution (test) and fd_task samples
    
    Args:
        msp_ind: MSP scores for in-distribution test samples
        msp_fd: MSP scores for fd_task samples
        
    Returns:
        Dictionary
    """
    # combine msp_ind and msp_fd
    combined_scores = np.concatenate([msp_fd, msp_ind])


    # Create labels (1 for in-distribution, 0 for OOD)
    labels = np.concatenate([
        np.ones_like(msp_fd),  # 1 fd
        np.zeros_like(msp_ind)   # 0 for IND
    ])

    # Check the distribution of combined_scores and labels
    print("Labels Distribution:", np.unique(labels, return_counts=True))

    # Handle NaN values
    valid_mask = ~np.isnan(combined_scores)
    combined_scores = combined_scores[valid_mask]
    labels = labels[valid_mask]

    if len(combined_scores) == 0:
        return {
            'aupr': np.nan,
            'auroc': np.nan,
            'auac': np.nan
        }

    # Print labels and combined_scores for debugging
    print("Labels:", labels)
    print("Combined Scores:", combined_scores)

    # Calculate AUPR and AUROC
    aupr = sk.average_precision_score(labels, combined_scores)
    auroc = sk.roc_auc_score(labels, combined_scores)

    # Calculate AUAC (Area Under Accuracy Curve)
    # Sort by confidence scores (descending)
    sort_idx = np.argsort(-combined_scores)
    sorted_scores = combined_scores[sort_idx]
    sorted_labels = labels[sort_idx]

    # Calculate cumulative accuracy at each threshold
    total = len(sorted_labels)
    correct_cumsum = np.cumsum(sorted_labels)
    accuracy_curve = correct_cumsum / (np.arange(total) + 1)

    # AUAC is the area under the accuracy curve (normalized to [0,1])
    auac = np.mean(1 - accuracy_curve)  # Lower is better

    return {
    'aupr': aupr,
    'auroc': auroc,
    'auac': auac
    }

def plot_auac_curve(scores, labels, accuracy_curve, auac, plot_path):
    """Plot the AUAC curve with thresholds"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Accuracy vs. Threshold (Percentage of samples included)
    thresholds = np.linspace(0, 100, len(accuracy_curve))
    ax1.plot(thresholds, accuracy_curve, 'b-', linewidth=2, label='Accuracy Curve')
    ax1.fill_between(thresholds, 0, accuracy_curve, alpha=0.3, label=f'AUAC = {auac:.3f}')
    ax1.set_xlabel('Percentage of Samples Included (Most Confident First)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs. Sample Inclusion Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy vs. Confidence Score Threshold
    unique_scores = np.unique(scores)
    accuracy_at_threshold = []
    for threshold in unique_scores:
        mask = scores >= threshold
        if np.sum(mask) > 0:
            acc = np.mean(labels[mask])
            accuracy_at_threshold.append(acc)
        else:
            accuracy_at_threshold.append(0)
    
    ax2.plot(unique_scores, accuracy_at_threshold, 'r-', linewidth=2)
    ax2.set_xlabel('Confidence Score Threshold')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs. Confidence Threshold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if path provided
    if plot_path:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"AUAC plot saved to: {plot_path}")
    
    return fig