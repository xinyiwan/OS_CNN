import shutil
import os
import optuna
import pathlib as Path

class BestModelCallback:
    def __init__(self, exp_save_path: Path, prefix: str, n_folds: int = 5):
        self.exp_save_path = exp_save_path
        self.prefix = prefix
        self.n_folds = n_folds
    
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        """Callback to save best models after each trial"""
        try:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                return
            
            # Simplified best trial check
            if study.best_trial.number == trial.number:
                self._save_best_models(trial)
                    
        except Exception as e:
            print(f"Callback error for trial {trial.number}: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_best_models(self, trial: optuna.trial.FrozenTrial):
        """Helper method to save best models"""
        ensemble_save_dir = self.exp_save_path / "checkpoints"
        best_models_save_dir = self.exp_save_path / "best_models_for_ensemble"
        best_models_save_dir.mkdir(exist_ok=True)
        
        print(f"🎉 Trial {trial.number} is the new best! Saving models for ensemble...")
        
        # Copy fold checkpoints
        models_copied = 0
        for fold in range(self.n_folds):
            source_path = ensemble_save_dir / f"trial_{trial.number}_{self.prefix}_inner_{fold}_best.pth"
            destination_path = best_models_save_dir / f"best_ensemble_fold_{fold}.pth"
            
            if source_path.exists():
                shutil.copy(str(source_path), str(destination_path))
                models_copied += 1
                print(f"    Copied fold {fold} checkpoint")
            else:
                print(f"    Warning: Checkpoint file not found for fold {fold}: {source_path}")
        
        print(f"Successfully copied {models_copied}/{self.n_folds} model checkpoints")
        self._save_trial_info(trial, best_models_save_dir)
    
    def _save_trial_info(self, trial: optuna.trial.FrozenTrial, save_dir: Path):
        """Save trial information to file"""
        trial_info_path = save_dir / "best_trial_info.txt"
        with open(trial_info_path, 'w') as f:
            f.write(f"Best Trial Number: {trial.number}\n")
            f.write(f"Best Inner mean AUC: {trial.value:.4f}\n")
            f.write(f"Outer Ensemble AUC: {trial.user_attrs['Outer-Ensemble-AUC']:.4f}\n")
            f.write(f"Outer Ensemble accuracy: {trial.user_attrs['Outer-Ens-accuracy']:.4f}\n")
            f.write(f"Outer Ensemble sensitivity: {trial.user_attrs['Outer-Ens-sensitivity']:.4f}\n")
            f.write(f"Outer Ensemble specificity: {trial.user_attrs['Outer-Ens-specificity']:.4f}\n")
            f.write(f"Hyperparameters:\n")
            for key, value in trial.params.items():
                f.write(f"  {key}: {value}\n")
        
        print(f"Saved best trial info to {trial_info_path}")