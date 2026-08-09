# ============================================================
# EVALUATION METRICS
# Comprehensive performance metrics for AI system
# ============================================================

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import pandas as pd


class MetricsCalculator:
    """
    Calculate comprehensive performance metrics for the AI system.
    Supports classification, regression, and module comparison.
    """

    def __init__(self):
        self.results = {}
        self.module_metrics = defaultdict(dict)
        self.patient_results = []

    def reset(self):
        """Reset all stored metrics"""
        self.results = {}
        self.module_metrics = defaultdict(dict)
        self.patient_results = []

    def calculate_classification_metrics(self, 
                                         y_true: List[str], 
                                         y_pred: List[str],
                                         labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            labels: List of all possible labels
            
        Returns:
            Dictionary with all metrics
        """
        if not y_true or not y_pred:
            return {'error': 'Empty predictions provided'}
        
        if len(y_true) != len(y_pred):
            return {'error': 'Length mismatch between true and predicted'}
        
        # Get unique labels
        all_labels = labels or list(set(y_true + y_pred))
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred, labels=all_labels),
            'classification_report': classification_report(
                y_true, y_pred, 
                labels=all_labels,
                zero_division=0,
                output_dict=True
            ),
            'labels': all_labels,
            'n_samples': len(y_true),
            'n_correct': sum(1 for t, p in zip(y_true, y_pred) if t == p)
        }
        
        # Calculate per-class metrics
        per_class = {}
        report = metrics['classification_report']
        for label in all_labels:
            if label in report:
                per_class[label] = {
                    'precision': report[label]['precision'],
                    'recall': report[label]['recall'],
                    'f1': report[label]['f1-score'],
                    'support': report[label]['support']
                }
        metrics['per_class'] = per_class
        
        return metrics

    def calculate_module_comparison(self, 
                                    module_predictions: Dict[str, List[str]],
                                    y_true: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Compare performance across multiple modules.
        
        Args:
            module_predictions: Dict mapping module name to predictions
            y_true: Ground truth labels
            
        Returns:
            Dict mapping module name to metrics
        """
        comparison = {}
        
        for module_name, y_pred in module_predictions.items():
            metrics = self.calculate_classification_metrics(y_true, y_pred)
            comparison[module_name] = {
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1': metrics.get('f1', 0),
                'n_correct': metrics.get('n_correct', 0),
                'n_samples': metrics.get('n_samples', 0)
            }
        
        return comparison

    def calculate_confidence_metrics(self, 
                                     confidences: List[float],
                                     correctness: List[bool]) -> Dict[str, float]:
        """
        Calculate metrics related to confidence calibration.
        
        Args:
            confidences: List of confidence scores (0-1)
            correctness: List of booleans indicating correctness
            
        Returns:
            Dictionary with confidence metrics
        """
        if not confidences:
            return {'error': 'Empty confidences provided'}
        
        confidences = np.array(confidences)
        correctness = np.array(correctness)
        
        # Average confidence
        avg_confidence = np.mean(confidences)
        
        # Accuracy at different confidence thresholds
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        threshold_metrics = {}
        for thresh in thresholds:
            mask = confidences >= thresh
            if np.sum(mask) > 0:
                thresh_acc = np.mean(correctness[mask])
                thresh_count = np.sum(mask)
            else:
                thresh_acc = 0
                thresh_count = 0
            threshold_metrics[f'accuracy_at_{thresh:.1f}'] = thresh_acc
            threshold_metrics[f'count_at_{thresh:.1f}'] = thresh_count
        
        # Calibration error (simplified)
        # ECE: Expected Calibration Error
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(confidences, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 1)
        
        ece = 0
        for i in range(len(bins) - 1):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_acc = np.mean(correctness[mask])
                bin_conf = np.mean(confidences[mask])
                bin_weight = np.sum(mask) / len(confidences)
                ece += bin_weight * abs(bin_acc - bin_conf)
        
        return {
            'avg_confidence': avg_confidence,
            'overall_accuracy': np.mean(correctness),
            'ece': ece,
            **threshold_metrics
        }

    def calculate_diagnosis_metrics(self, 
                                    diagnoses: List[str],
                                    urgencies: List[str]) -> Dict[str, Any]:
        """
        Calculate metrics for diagnosis and urgency distributions.
        
        Args:
            diagnoses: List of diagnosis labels
            urgencies: List of urgency labels
            
        Returns:
            Dictionary with distribution metrics
        """
        from collections import Counter
        
        diagnosis_counts = Counter(diagnoses)
        urgency_counts = Counter(urgencies)
        
        return {
            'diagnosis_distribution': dict(diagnosis_counts),
            'urgency_distribution': dict(urgency_counts),
            'n_unique_diagnoses': len(diagnosis_counts),
            'n_unique_urgencies': len(urgency_counts),
            'most_common_diagnosis': diagnosis_counts.most_common(1)[0] if diagnosis_counts else None,
            'most_common_urgency': urgency_counts.most_common(1)[0] if urgency_counts else None
        }

    def calculate_processing_time_metrics(self, 
                                          times: List[float]) -> Dict[str, float]:
        """
        Calculate metrics for processing times.
        
        Args:
            times: List of processing times in seconds
            
        Returns:
            Dictionary with time metrics
        """
        if not times:
            return {'error': 'Empty times provided'}
        
        times = np.array(times)
        
        return {
            'mean_time': np.mean(times),
            'median_time': np.median(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'total_time': np.sum(times),
            'n_samples': len(times)
        }

    def add_patient_result(self, 
                           patient_id: str,
                           true_diagnosis: Optional[str],
                           predicted_diagnosis: str,
                           confidence: float,
                           urgency: str,
                           module_results: Dict[str, Dict],
                           processing_time: float = 0) -> None:
        """
        Add a single patient result for aggregate analysis.
        
        Args:
            patient_id: Patient identifier
            true_diagnosis: Ground truth diagnosis (optional)
            predicted_diagnosis: Predicted diagnosis
            confidence: Confidence score
            urgency: Urgency level
            module_results: Results from each module
            processing_time: Time taken for processing
        """
        result = {
            'patient_id': patient_id,
            'true_diagnosis': true_diagnosis,
            'predicted_diagnosis': predicted_diagnosis,
            'confidence': confidence,
            'urgency': urgency,
            'module_results': module_results,
            'processing_time': processing_time,
            'correct': true_diagnosis == predicted_diagnosis if true_diagnosis else None
        }
        self.patient_results.append(result)
        
        # Update module metrics
        for module_name, mod_result in module_results.items():
            if isinstance(mod_result, dict):
                self.module_metrics[module_name]['confidences'].append(
                    mod_result.get('confidence', 0)
                )
                self.module_metrics[module_name]['diagnoses'].append(
                    mod_result.get('diagnosis', 'unknown')
                )

    def get_patient_summary(self) -> pd.DataFrame:
        """
        Get patient results as a pandas DataFrame.
        
        Returns:
            DataFrame with patient results
        """
        return pd.DataFrame(self.patient_results)

    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """
        Calculate aggregate metrics from all patient results.
        
        Returns:
            Dictionary with aggregate metrics
        """
        if not self.patient_results:
            return {'error': 'No patient results available'}
        
        # Extract data
        predictions = [r['predicted_diagnosis'] for r in self.patient_results]
        confidences = [r['confidence'] for r in self.patient_results]
        urgencies = [r['urgency'] for r in self.patient_results]
        times = [r.get('processing_time', 0) for r in self.patient_results]
        
        # Check which have true diagnoses
        has_true = [r['true_diagnosis'] is not None for r in self.patient_results]
        true_diagnoses = [r['true_diagnosis'] for r in self.patient_results if r['true_diagnosis'] is not None]
        
        aggregate = {
            'n_patients': len(self.patient_results),
            'n_with_true_labels': sum(has_true),
            'avg_confidence': np.mean(confidences),
            'diagnosis_distribution': dict(zip(*np.unique(predictions, return_counts=True))),
            'urgency_distribution': dict(zip(*np.unique(urgencies, return_counts=True))),
        }
        
        # Calculate accuracy if we have true labels
        if true_diagnoses:
            correct = [r['correct'] for r in self.patient_results if r['correct'] is not None]
            aggregate['accuracy'] = np.mean(correct)
            aggregate['n_correct'] = sum(correct)
        
        # Time metrics
        if any(times):
            aggregate['time_metrics'] = self.calculate_processing_time_metrics(times)
        
        # Confidence calibration
        if confidences and true_diagnoses:
            correctness = [r['correct'] for r in self.patient_results if r['correct'] is not None]
            if correctness:
                aggregate['confidence_metrics'] = self.calculate_confidence_metrics(
                    confidences[:len(correctness)], correctness
                )
        
        # Module comparison
        if self.module_metrics:
            aggregate['module_summary'] = {
                name: {
                    'avg_confidence': np.mean(metrics.get('confidences', [0])),
                    'n_predictions': len(metrics.get('confidences', []))
                }
                for name, metrics in self.module_metrics.items()
            }
        
        return aggregate


# ── Test Metrics Calculator ─────────────────────────────
def test_metrics():
    """Test the MetricsCalculator module"""
    print("\n" + "=" * 60)
    print("  TESTING METRICS CALCULATOR")
    print("=" * 60)
    
    mc = MetricsCalculator()
    
    # Test classification metrics
    print("\n--- Classification Metrics ---")
    y_true = ['covid19', 'flu', 'covid19', 'common_cold', 'flu', 'covid19']
    y_pred = ['covid19', 'flu', 'covid19', 'flu', 'flu', 'covid19']
    
    metrics = mc.calculate_classification_metrics(y_true, y_pred)
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  F1 Score: {metrics['f1']:.2%}")
    print(f"  Correct: {metrics['n_correct']}/{metrics['n_samples']}")
    
    # Test module comparison
    print("\n--- Module Comparison ---")
    module_preds = {
        'KnowledgeBase': ['covid19', 'flu', 'covid19', 'common_cold', 'flu', 'covid19'],
        'BayesianNet': ['covid19', 'flu', 'covid19', 'flu', 'flu', 'covid19'],
        'MLClassifier': ['covid19', 'flu', 'flu', 'common_cold', 'flu', 'covid19']
    }
    comparison = mc.calculate_module_comparison(module_preds, y_true)
    for module, metrics in comparison.items():
        print(f"  {module}: Accuracy={metrics['accuracy']:.2%}")
    
    print("\n✅ Metrics test passed!")


if __name__ == "__main__":
    test_metrics()