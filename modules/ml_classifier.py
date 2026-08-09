# ============================================================
# MODULE 4: ML Classifier — Supervised Diagnosis
# Covers: Week 9 (Supervised Learning & Decision Trees)
# ============================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, List, Tuple, Any, Optional
import os
import sys

# Add parent directory to path for data module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')


class MLDiagnosticClassifier:
    """
    Ensemble ML-based diagnostic classifier.
    Uses Decision Trees, Random Forest, and Gradient Boosting 
    with voting ensemble for robust diagnosis.
    
    Loads data from CSV files in the data/ directory.
    """

    # 22 symptoms for comprehensive coverage (matches data/symptoms.csv)
    SYMPTOM_FEATURES = [
        'fever', 'cough', 'fatigue', 'headache',
        'body_aches', 'loss_of_smell', 'chest_pain',
        'rash', 'joint_pain', 'shortness_of_breath',
        'sweating', 'frequent_urination', 'excessive_thirst',
        'blurred_vision', 'night_sweats', 'weight_loss',
        'stiff_neck', 'light_sensitivity', 'sore_throat',
        'runny_nose', 'sneezing', 'nausea'
    ]

    DISEASE_LABELS = [
        'flu', 'covid19', 'dengue', 'cardiac_event',
        'diabetes', 'common_cold', 'tuberculosis', 'meningitis',
        'pneumonia', 'bronchitis', 'allergies',
        'gastroenteritis', 'strep_throat'
    ]

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the ML classifier.
        
        Args:
            data_dir: Path to the data directory containing CSV files
        """
        self.data_dir = data_dir
        self.models = {
            'Decision Tree': DecisionTreeClassifier(
                max_depth=8, 
                min_samples_split=5,
                min_samples_leaf=2,
                criterion='entropy', 
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=150, 
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150, 
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                random_state=42
            ),
        }
        
        # Voting ensemble
        self.ensemble = None
        self.best_model = None
        self.best_model_name = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = None
        self.cv_scores = {}
        
        # Store confusion matrices
        self.confusion_matrices = {}
        
        # Data
        self.X = None
        self.y = None
        self.df = None
        self._X_test = None
        self._y_test = None
        
        print(f"[MLClassifier] Initialized with data directory: {self.data_dir}")

    def _load_data(self) -> pd.DataFrame:
        """
        Load patient data from CSV file.
        
        Returns:
            DataFrame with patient records
        """
        # Try to load from data directory
        csv_path = os.path.join(self.data_dir, "patient_records.csv")
        
        if not os.path.exists(csv_path):
            print(f"[MLClassifier] Data file not found: {csv_path}")
            print("[MLClassifier] Attempting to load from parent directory...")
            csv_path = "../data/patient_records.csv"
            
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"Patient records file not found. Expected at: {self.data_dir}/patient_records.csv"
                )
        
        df = pd.read_csv(csv_path)
        print(f"[MLClassifier] Loaded {len(df)} patient records from {csv_path}")
        
        # Check if we have the expected columns
        missing_features = [f for f in self.SYMPTOM_FEATURES if f not in df.columns]
        if missing_features:
            print(f"[MLClassifier] Warning: Missing symptom columns: {missing_features}")
            # Add missing columns with default value 0
            for feature in missing_features:
                df[feature] = 0
        
        return df

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with patient records
            
        Returns:
            Tuple of (features, labels)
        """
        # Extract features - symptoms columns
        X = df[self.SYMPTOM_FEATURES].values
        
        # Extract labels - disease column
        y = df['disease'].values
        
        print(f"[MLClassifier] Prepared {len(X)} samples with {len(self.SYMPTOM_FEATURES)} features")
        print(f"[MLClassifier] Unique diseases: {np.unique(y)}")
        
        return X, y

    def train(self, verbose: bool = True, cv_folds: int = 5) -> Dict:
        """
        Train all models and select the best one with cross-validation.
        Loads data from CSV files.
        
        Args:
            verbose: Print progress if True
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary of model scores
        """
        # Load data from CSV
        self.df = self._load_data()
        
        # Prepare data
        X, y = self._prepare_data(self.df)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.X = X
        self.y = y_encoded
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        results = {}
        best_acc = 0.0
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        if verbose:
            print("\n" + "=" * 55)
            print("  ML Diagnostic Classifier — Training")
            print(f"  Training on {len(X_train)} samples, testing on {len(X_test)}")
            print("=" * 55)

        # Train each model and store confusion matrices
        self.confusion_matrices = {}
        
        for name, model in self.models.items():
            if verbose:
                print(f"\n  🌲 Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Train on full training set
            model.fit(X_train_scaled, y_train)
            
            # Test predictions
            y_pred = model.predict(X_test_scaled)
            test_acc = accuracy_score(y_test, y_pred)
            
            # Additional metrics
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Store confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            self.confusion_matrices[name] = {
                'matrix': cm,
                'labels': self.label_encoder.classes_,
                'y_true': y_test,
                'y_pred': y_pred
            }
            
            results[name] = {
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'test_acc': test_acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm
            }
            
            if verbose:
                print(f"     CV Accuracy    : {cv_mean:.4f} ± {cv_std:.4f}")
                print(f"     Test Accuracy  : {test_acc:.4f}")
                print(f"     F1 Score       : {f1:.4f}")

            # Track best model
            if test_acc > best_acc:
                best_acc = test_acc
                self.best_model = model
                self.best_model_name = name
                self.feature_importance = getattr(model, 'feature_importances_', None)

        # Build voting ensemble
        self.ensemble = VotingClassifier(
            estimators=[(name, model) for name, model in self.models.items()],
            voting='soft'
        )
        self.ensemble.fit(X_train_scaled, y_train)
        
        # Get ensemble predictions and confusion matrix
        y_pred_ensemble = self.ensemble.predict(X_test_scaled)
        cm_ensemble = confusion_matrix(y_test, y_pred_ensemble)
        self.confusion_matrices['Ensemble'] = {
            'matrix': cm_ensemble,
            'labels': self.label_encoder.classes_,
            'y_true': y_test,
            'y_pred': y_pred_ensemble
        }
        
        # Save test data for evaluation
        self._X_test = X_test_scaled
        self._y_test = y_test
        self._y_train = y_train
        self._X_train = X_train_scaled

        self.is_trained = True
        self.cv_scores = results

        if verbose:
            # Ensemble performance
            ensemble_acc = self.ensemble.score(X_test_scaled, y_test)
            print(f"\n  🏆 Best Model     : {self.best_model_name} ({best_acc:.4f})")
            print(f"  🏆 Ensemble Acc   : {ensemble_acc:.4f}")
            
        return results

    def get_confusion_matrices(self) -> Dict[str, Dict]:
        """
        Get all confusion matrices from training.
        
        Returns:
            Dictionary mapping model names to confusion matrix data
        """
        return self.confusion_matrices

    def plot_all_confusion_matrices(self, save_dir: str = "evaluation_output"):
        """
        Plot confusion matrices for all models.
        
        Args:
            save_dir: Directory to save the plots
        """
        if not self.is_trained:
            print("[MLClassifier] Model not trained. Please call train() first.")
            return
        
        if not self.confusion_matrices:
            print("[MLClassifier] No confusion matrices available.")
            return
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Get number of models
        n_models = len(self.confusion_matrices)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (name, data) in enumerate(self.confusion_matrices.items()):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            cm = data['matrix']
            labels = data['labels']
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels,
                       ax=ax, cbar=False)
            ax.set_title(f'{name}', fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            
            # Rotate labels if too many
            if len(labels) > 5:
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
        
        # Hide any unused subplots
        for idx in range(len(self.confusion_matrices), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Confusion Matrices for All Classifiers', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save the figure
        save_path = os.path.join(save_dir, 'all_confusion_matrices.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        
        plt.show()
        return fig

    def plot_single_confusion_matrix(self, model_name: str, save_dir: str = "evaluation_output"):
        """
        Plot a single confusion matrix for a specific model.
        
        Args:
            model_name: Name of the model ('Decision Tree', 'Random Forest', etc.)
            save_dir: Directory to save the plot
        """
        if not self.is_trained:
            print("[MLClassifier] Model not trained. Please call train() first.")
            return
        
        if model_name not in self.confusion_matrices:
            print(f"[MLClassifier] Model '{model_name}' not found.")
            print(f"  Available models: {list(self.confusion_matrices.keys())}")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        data = self.confusion_matrices[model_name]
        cm = data['matrix']
        labels = data['labels']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels,
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_title(f'Confusion Matrix - {model_name}', fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        if len(labels) > 5:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
        
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'confusion_matrix_{model_name.replace(" ", "_")}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        
        plt.show()
        return fig

    def predict(self, symptoms: List[str], use_ensemble: bool = True) -> Dict:
        """
        Predict disease from symptom list with detailed confidence.
        
        Args:
            symptoms: List of symptom strings
            use_ensemble: Use voting ensemble if True, else best model
            
        Returns:
            Dictionary with diagnosis and confidence details
        """
        if not self.is_trained:
            print("[MLClassifier] Model not trained. Training with default parameters...")
            self.train(verbose=False)

        # Create feature vector
        feature_vector = self._symptoms_to_vector(symptoms)
        features_scaled = self.scaler.transform(feature_vector.reshape(1, -1))

        # Select model
        model = self.ensemble if use_ensemble and self.ensemble else self.best_model
        
        # Get predictions
        y_pred = model.predict(features_scaled)[0]
        y_proba = model.predict_proba(features_scaled)[0]

        # Get diagnosis
        diagnosis = self.label_encoder.inverse_transform([y_pred])[0]
        confidence = float(y_proba[y_pred])

        # Get top 5 predictions
        classes = self.label_encoder.inverse_transform(range(len(y_proba)))
        prob_map = dict(zip(classes, y_proba))
        top5 = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)[:5]

        # Get disease severity (if available)
        severity = self._get_disease_severity(diagnosis, symptoms)

        return {
            'diagnosis': diagnosis,
            'confidence': confidence,
            'top5': top5,
            'model_used': 'Ensemble' if use_ensemble and self.ensemble else self.best_model_name,
            'symptom_vector': feature_vector.tolist(),
            'severity': severity,
            'all_probabilities': prob_map
        }

    def _symptoms_to_vector(self, symptoms: List[str]) -> np.ndarray:
        """Convert symptoms list to binary feature vector"""
        normalized = [s.lower().strip().replace(' ', '_') for s in symptoms]
        
        vector = np.zeros(len(self.SYMPTOM_FEATURES))
        for i, feature in enumerate(self.SYMPTOM_FEATURES):
            if feature in normalized:
                vector[i] = 1
        
        return vector

    def _get_disease_severity(self, disease: str, symptoms: List[str]) -> str:
        """Estimate severity based on disease and symptoms"""
        severe_diseases = ['cardiac_event', 'meningitis', 'pneumonia']
        moderate_diseases = ['covid19', 'dengue', 'tuberculosis']
        
        if disease in severe_diseases:
            return 'SEVERE'
        elif disease in moderate_diseases:
            if len(symptoms) >= 5:
                return 'MODERATE'
            return 'MILD'
        else:
            return 'MILD'

    def analyze(self, percept) -> Dict:
        """Module interface for the agent"""
        result = self.predict(percept.symptoms, use_ensemble=True)
        
        return {
            'diagnosis': result['diagnosis'],
            'confidence': result['confidence'],
            'details': {
                'top5': result['top5'],
                'model_used': result['model_used'],
                'severity': result.get('severity', 'UNKNOWN')
            }
        }

    def plot_evaluation(self, save_path: Optional[str] = None):
        """Visualize model performance with multiple plots"""
        if not self.is_trained:
            print("[MLClassifier] Model not trained. Please call train() first.")
            return

        y_pred = self.best_model.predict(self._X_test)
        y_pred_ensemble = self.ensemble.predict(self._X_test) if self.ensemble else y_pred
        labels = self.label_encoder.classes_

        fig = plt.figure(figsize=(18, 10))
        
        # 1. Confusion Matrix - Best Model
        ax1 = plt.subplot(2, 3, 1)
        cm = confusion_matrix(self._y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax1)
        ax1.set_title(f"Confusion Matrix\n{self.best_model_name}")
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("True")
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 2. Confusion Matrix - Ensemble
        ax2 = plt.subplot(2, 3, 2)
        cm_ensemble = confusion_matrix(self._y_test, y_pred_ensemble)
        sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Greens',
                    xticklabels=labels, yticklabels=labels, ax=ax2)
        ax2.set_title("Confusion Matrix\nEnsemble")
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("True")
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 3. Feature Importance
        ax3 = plt.subplot(2, 3, 3)
        if self.feature_importance is not None:
            importances = self.feature_importance
            sorted_idx = np.argsort(importances)[::-1][:12]
            top_features = [self.SYMPTOM_FEATURES[i] for i in sorted_idx]
            top_values = importances[sorted_idx]
            
            colors = plt.cm.RdYlGn(top_values / top_values.max())
            ax3.barh(range(len(top_features)), top_values[::-1], color=colors[::-1])
            ax3.set_yticks(range(len(top_features)))
            ax3.set_yticklabels(top_features[::-1])
            ax3.set_title("Feature Importances (Top 12)")
            ax3.set_xlabel("Importance Score")
            ax3.grid(True, alpha=0.3)

        # 4. Model Comparison
        ax4 = plt.subplot(2, 3, 4)
        model_names = list(self.cv_scores.keys()) + ['Ensemble']
        test_acc = [self.cv_scores[m]['test_acc'] for m in self.cv_scores.keys()]
        test_acc.append(accuracy_score(self._y_test, y_pred_ensemble))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = ax4.bar(model_names, test_acc, color=colors[:len(model_names)])
        ax4.set_ylabel("Test Accuracy")
        ax4.set_title("Model Comparison")
        ax4.set_ylim(0, 1)
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, acc in zip(bars, test_acc):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

        # 5. CV Scores Distribution
        ax5 = plt.subplot(2, 3, 5)
        cv_means = [self.cv_scores[m]['cv_mean'] for m in self.cv_scores.keys()]
        cv_stds = [self.cv_scores[m]['cv_std'] for m in self.cv_scores.keys()]
        
        ax5.bar(self.cv_scores.keys(), cv_means, yerr=cv_stds, capsize=5,
                color=colors[:len(cv_means)])
        ax5.set_ylabel("Cross-Validation Accuracy")
        ax5.set_title("CV Performance")
        ax5.set_ylim(0, 1)
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)

        # 6. Classification Report (text summary)
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        report = classification_report(self._y_test, y_pred, 
                                       target_names=labels, 
                                       zero_division=0,
                                       output_dict=True)
        
        summary_text = "📊 Classification Report (Best Model)\n"
        summary_text += "─" * 30 + "\n"
        summary_text += f"Accuracy  : {report['accuracy']:.3f}\n"
        summary_text += f"Precision : {report['weighted avg']['precision']:.3f}\n"
        summary_text += f"Recall    : {report['weighted avg']['recall']:.3f}\n"
        summary_text += f"F1 Score  : {report['weighted avg']['f1-score']:.3f}\n"
        summary_text += "\nPer-Class F1:\n"
        
        for label in labels[:8]:
            f1 = report.get(label, {}).get('f1-score', 0)
            summary_text += f"  {label[:12]:12}: {f1:.3f}\n"
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

        plt.suptitle("ML Diagnostic Model Evaluation", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✅ Saved: {save_path}")
        
        plt.show()


# ── Test the ML Classifier ─────────────────────────────────
def test_ml_classifier():
    """Test the MLDiagnosticClassifier module"""
    print("\n" + "=" * 60)
    print("  TESTING ML CLASSIFIER")
    print("=" * 60)
    
    clf = MLDiagnosticClassifier()
    
    # Train the model
    print("\n--- Training ---")
    scores = clf.train(verbose=True)
    
    # Print confusion matrices
    print("\n--- Confusion Matrices ---")
    cm_data = clf.get_confusion_matrices()
    for name, data in cm_data.items():
        print(f"\n  {name}:")
        print(f"    Matrix shape: {data['matrix'].shape}")
        print(f"    Labels: {data['labels']}")
    
    # Plot all confusion matrices
    print("\n--- Plotting Confusion Matrices ---")
    clf.plot_all_confusion_matrices(save_dir="evaluation_output")
    
    # Plot individual confusion matrix for best model
    if clf.best_model_name:
        print(f"\n--- Plotting {clf.best_model_name} Confusion Matrix ---")
        clf.plot_single_confusion_matrix(clf.best_model_name, save_dir="evaluation_output")
    
    # Test prediction
    print("\n--- Predictions ---")
    test_cases = [
        ["fever", "cough", "loss_of_smell", "fatigue"],
        ["fever", "cough", "body_aches", "headache"],
        ["runny_nose", "sneezing", "sore_throat"],
        ["high_fever", "joint_pain", "rash"],
    ]
    
    for symptoms in test_cases:
        result = clf.predict(symptoms, use_ensemble=True)
        print(f"\n  Symptoms: {symptoms}")
        print(f"    Diagnosis: {result['diagnosis']}")
        print(f"    Confidence: {result['confidence']:.2%}")
        print(f"    Top 5: {result['top5'][:3]}")
    
    print("\n✅ ML Classifier test passed!")


if __name__ == "__main__":
    test_ml_classifier()