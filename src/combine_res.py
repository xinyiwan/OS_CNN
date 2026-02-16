import pandas as pd
import numpy as np
import glob, os
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
import scipy.stats as st
import warnings
warnings.filterwarnings('ignore')


LABEL_DIS = '/projects/prjs1779/Osteosarcoma/exp_data/label_distributions_summary.csv'

def combine_performance_from_ensembles(exp_name, modality, model_type='resnet10_pretrained'):
    """Combine ensemble predictions from all folds (already at subject level)"""
    
    m = modality.lower()
    md = model_type
    all_fold_results = []
    all_metrics = []
    
    for fold in range(20):
        # Ensemble path pattern
        ensemble_res = f'/projects/prjs1779/Osteosarcoma/experiments/{exp_name}/{m}/{m}_{fold}_{md}/best_models_for_ensemble/predictions_ensemble.csv'
        
        if not os.path.exists(ensemble_res):
            # If ensemble file doesn't exist, create it by combining inner folds
            predictions_file = ensemble_res.replace('predictions_ensemble.csv', 'predictions.csv')
            if not os.path.exists(predictions_file):
                print(f"  ⚠️  Fold {fold}: Neither ensemble nor predictions file found")
                continue
                
            print(f"  🔧 Fold {fold}: Creating ensemble predictions...")
            d = combine_probabilities_by_position(predictions_file)
            if d is not None:
                d.to_csv(ensemble_res, index=False)
        else:
            # Load existing ensemble predictions
            d = pd.read_csv(ensemble_res)
        
        if d is not None and not d.empty:
            # Add fold information
            d['fold'] = fold
            
            # Calculate metrics for this fold
            fold_metrics = calculate_metrics(d)
            fold_metrics['fold'] = fold
            all_metrics.append(fold_metrics)
            
            # Store predictions for this fold
            all_fold_results.append(d)
    
    if not all_fold_results:
        print("❌ No valid fold data found")
        return None, None
    
    return all_metrics

def combine_probabilities_by_position(csv_path, threshold=0.5):
    """
    Combine probabilities by matching image positions across inner folds.
    Assumes images are in the same order within each inner_fold.
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Create a position index within each fold
        df['position'] = df.groupby('inner_fold').cumcount()
        
        # Group by position to get mean probability across folds
        position_stats = df.groupby(['subject_id', 'position']).agg({
            'probability': 'mean',
            'ground_truth': 'first'
        }).reset_index()
        
        # Calculate prediction
        position_stats['prediction'] = (position_stats['probability'] >= threshold).astype(int)
        
        # Drop position column
        result = position_stats.drop('position', axis=1)
        return result
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None

def calculate_metrics(predictions_df):
    """Calculate metrics from subject-level predictions"""
    
    probs = predictions_df['probability'].values
    labels = predictions_df['ground_truth'].values
    
    # Ensure we have valid data
    if len(np.unique(labels)) < 2:
        print("Warning: Only one class present in data")
        return {
            'auroc': np.nan,
            'accuracy': np.nan,
            'sensitivity': np.nan,
            'specificity': np.nan,
            'n_samples': len(labels),
            'n_positive': int(sum(labels)),
            'n_negative': len(labels) - int(sum(labels))
        }
    
    # Calculate predictions (threshold 0.5)
    preds = (probs > 0.5).astype(int)
    
    # Calculate AUC
    try:
        auroc = roc_auc_score(labels, probs)
    except Exception as e:
        print(f"Error calculating AUC: {e}")
        auroc = np.nan
    
    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)
    
    # Calculate confusion matrix metrics
    try:
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    except Exception as e:
        print(f"Error calculating confusion matrix: {e}")
        tp = tn = fp = fn = 0
        sensitivity = specificity = np.nan
    
    return {
        'auroc': auroc,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'n_samples': len(labels),
        'n_positive': int(tp + fn),
        'n_negative': int(tn + fp),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }

def compute_confidence(metric, N_train, N_test, alpha=0.95):
    """
    Function to calculate the adjusted confidence interval for cross-validation.
    metric: numpy array containing the result for a metric for the different cross validations
    (e.g. If 20 cross-validations are performed it is a list of length 20 with the calculated accuracy for
    each cross validation)
    N_train: Integer, number of training samples
    N_test: Integer, number of test_samples
    alpha: float ranging from 0 to 1 to calculate the alpha*100% CI, default 0.95
    """

    # Remove NaN values if they are there
    if np.isnan(metric).any():
        print('[WORC Warning] Array contains nan: removing.')
        metric = np.asarray(metric)
        metric = metric[np.logical_not(np.isnan(metric))]

    # Convert to floats, as python 2 rounds the divisions if we have integers
    N_train = float(N_train)
    N_test = float(N_test)
    N_iterations = float(len(metric))

    if N_iterations == 1.0:
        print('[WORC Warning] Cannot compute a confidence interval for a single iteration.')
        print('[WORC Warning] CI will be set to value of single iteration.')
        metric_average = np.mean(metric)
        CI = (metric_average, metric_average)
    else:
        metric_average = np.mean(metric)
        S_uj = 1.0 / (N_iterations - 1) * np.sum((metric_average - metric)**2.0)

        metric_std = np.sqrt((1.0/N_iterations + N_test/N_train)*S_uj)

        CI = st.t.interval(alpha, N_iterations-1, loc=metric_average, scale=metric_std)

    if np.isnan(CI[0]) and np.isnan(CI[1]):
        # When we cannot compute a CI, just give the averages
        CI = (metric_average, metric_average)
    return CI


def generate_roc_with_ci(exp_name, modality, model_type='resnet10_pretrained', alpha=0.95, n_thresholds=10):
    """
    Generate ROC curve data with confidence intervals across folds.

    Returns a DataFrame with FPR and TPR ranges for each threshold.
    """
    m = modality.lower()
    md = model_type

    # Collect all fold predictions
    all_predictions = []

    for fold in range(20):
        ensemble_res = f'/projects/prjs1779/Osteosarcoma/experiments/{exp_name}/{m}/{m}_{fold}_{md}/best_models_for_ensemble/predictions_ensemble.csv'

        if os.path.exists(ensemble_res):
            df = pd.read_csv(ensemble_res)
            if not df.empty and len(np.unique(df['ground_truth'])) == 2:
                all_predictions.append({
                    'fold': fold,
                    'probabilities': df['probability'].values,
                    'labels': df['ground_truth'].values
                })

    if not all_predictions:
        print("❌ No valid predictions found for ROC curve generation")
        return None

    print(f"✅ Found {len(all_predictions)} valid folds for ROC curve generation")

    # Get sample sizes for CI calculation
    distribution_df = pd.read_csv(LABEL_DIS)
    median_n_train = int(np.mean(distribution_df[distribution_df['modality'] == modality]['train_total'].values))
    median_n_test = int(np.mean(distribution_df[distribution_df['modality'] == modality]['test_total'].values))

    # Define thresholds
    thresholds = np.linspace(0, 1, n_thresholds)

    # Calculate FPR and TPR for each threshold across all folds
    roc_data = []

    for threshold in thresholds:
        fpr_values = []
        tpr_values = []

        for pred_data in all_predictions:
            probs = pred_data['probabilities']
            labels = pred_data['labels']

            # Make predictions with current threshold
            preds = (probs >= threshold).astype(int)

            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

            # Calculate TPR and FPR
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            fpr_values.append(fpr)
            tpr_values.append(tpr)

        # Compute confidence intervals
        fpr_values = np.array(fpr_values)
        tpr_values = np.array(tpr_values)

        fpr_ci = compute_confidence(fpr_values, median_n_train, median_n_test, alpha)
        tpr_ci = compute_confidence(tpr_values, median_n_train, median_n_test, alpha)

        roc_data.append({
            'threshold': threshold,
            'FPR': f"[{fpr_ci[0]:.8f} {fpr_ci[1]:.8f}]",
            'TPR': f"[{tpr_ci[0]:.8f} {tpr_ci[1]:.8f}]"
        })

    # Convert to DataFrame and reverse order (from threshold 1 to 0)
    roc_df = pd.DataFrame(roc_data)
    roc_df = roc_df.iloc[::-1].reset_index(drop=True)

    return roc_df


def main():
    parser = argparse.ArgumentParser(description="Combine ensemble predictions and compute CIs")
    parser.add_argument('--modality', type=str, default='T1W_FS_C', help='Image modality')
    parser.add_argument('--exp_name', type=str, default='pretrain', help='Experiment name')
    parser.add_argument('--model_type', type=str, default='resnet10_pretrained', help='Model type')
    parser.add_argument('--alpha', type=float, default=0.95, help='Confidence level')
    parser.add_argument('--n_thresholds', type=int, default=10, help='Number of thresholds for ROC curve')

    args = parser.parse_args()
    
    print(f"\nCombining predictions for:")
    print(f"  Modality: {args.modality}")
    print(f"  Experiment: {args.exp_name}")
    print(f"  Model type: {args.model_type}")
    print(f"  Confidence level: {args.alpha}")
    
    # Step 1: Combine predictions from all folds
    all_metrics = combine_performance_from_ensembles(
        args.exp_name, 
        args.modality, 
        args.model_type
    )
    
    if not all_metrics:
        print("❌ Failed to combine predictions. Exiting.")
        return
    
    # Convert metrics list to DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Step 2: Calculate overall metrics (average across folds)
    print(f"\nCalculating overall metrics across {len(metrics_df)} folds...")
    
    # Extract metric arrays from all folds
    auc_values = metrics_df['auroc'].dropna().values
    accuracy_values = metrics_df['accuracy'].dropna().values
    sensitivity_values = metrics_df['sensitivity'].dropna().values
    specificity_values = metrics_df['specificity'].dropna().values
    
    # Get sample sizes (use median across folds)
    # Is it fine?
    distribution_df = pd.read_csv(LABEL_DIS)
    median_n_train = int(np.mean(distribution_df[distribution_df['modality'] == args.modality]['train_total'].values))  
    median_n_test = int(np.mean(distribution_df[distribution_df['modality'] == args.modality]['test_total'].values))
    
    print(f"\nSample sizes (median across folds):")
    print(f"  Training samples: {median_n_train}")
    print(f"  Test samples: {median_n_test}")
    
    # Step 3: Compute CIs for each metric
    print(f"\nConfidence Intervals ({args.alpha*100:.0f}%):")
    print("-" * 50)
    
    # AUC
    if len(auc_values) > 0:
        auc_ci = compute_confidence(auc_values, median_n_train, median_n_test, args.alpha)
        print(f"AUC: {np.mean(auc_values):.3f} [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}] (n={len(auc_values)} folds)")
    else:
        print("AUC: No valid values")
    
    # Accuracy
    if len(accuracy_values) > 0:
        accuracy_ci = compute_confidence(accuracy_values, median_n_train, median_n_test, args.alpha)
        print(f"Accuracy: {np.mean(accuracy_values):.3f} [{accuracy_ci[0]:.3f}, {accuracy_ci[1]:.3f}] (n={len(accuracy_values)} folds)")
    else:
        print("Accuracy: No valid values")
    
    # Sensitivity
    if len(sensitivity_values) > 0:
        sensitivity_ci = compute_confidence(sensitivity_values, median_n_train, median_n_test, args.alpha)
        print(f"Sensitivity: {np.mean(sensitivity_values):.3f} [{sensitivity_ci[0]:.3f}, {sensitivity_ci[1]:.3f}] (n={len(sensitivity_values)} folds)")
    else:
        print("Sensitivity: No valid values")
    
    # Specificity
    if len(specificity_values) > 0:
        specificity_ci = compute_confidence(specificity_values, median_n_train, median_n_test, args.alpha)
        print(f"Specificity: {np.mean(specificity_values):.3f} [{specificity_ci[0]:.3f}, {specificity_ci[1]:.3f}] (n={len(specificity_values)} folds)")
    else:
        print("Specificity: No valid values")
    
    # Print summary statistics
    print(f"\nSummary statistics across folds:")
    print("-" * 50)
    if len(auc_values) > 0:
        print(f"AUC: {np.mean(auc_values):.3f} ± {np.std(auc_values):.3f}")
    if len(accuracy_values) > 0:
        print(f"Accuracy: {np.mean(accuracy_values):.3f} ± {np.std(accuracy_values):.3f}")
    
    # Save results
    res_dir = '/projects/prjs1779/Osteosarcoma/OS_CNN_res'
    metrics_output = os.path.join(res_dir, f"{args.exp_name}", f"{args.modality}", f"metrics_CI.csv")
    os.makedirs(os.path.dirname(metrics_output), exist_ok=True)

    metrics_df = metrics_df.round(2)
    metrics_df.to_csv(metrics_output, index=False)
    print(f"✅ Fold metrics saved to: {metrics_output}")

    # Step 4: Generate ROC curve data with confidence intervals
    print(f"\nGenerating ROC curve data with confidence intervals...")
    roc_df = generate_roc_with_ci(
        args.exp_name,
        args.modality,
        args.model_type,
        args.alpha,
        args.n_thresholds
    )

    if roc_df is not None:
        roc_output = os.path.join(res_dir, f"{args.exp_name}", f"{args.modality}", f"roc_curve_ci.csv")
        roc_df.to_csv(roc_output, index=False)
        print(f"✅ ROC curve data saved to: {roc_output}")
    else:
        print("❌ Failed to generate ROC curve data")


if __name__ == "__main__":
    main()