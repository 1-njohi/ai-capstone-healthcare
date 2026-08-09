# ============================================================
# EVALUATION VISUALIZATIONS
# Charts and plots for system performance analysis
# ============================================================

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import matplotlib.gridspec as gridspec


class Visualizer:
    """
    Create visualizations for system evaluation.
    Supports confusion matrices, performance comparisons,
    confidence calibration, and more.
    """

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """Initialize with style settings"""
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        sns.set_palette("husl")
        self.figures = []

    def plot_confusion_matrix(self,
                              cm: np.ndarray,
                              labels: List[str],
                              title: str = "Confusion Matrix",
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 8)):
        """Plot confusion matrix as a heatmap."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Normalize for percentage display
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        cm_percent = np.nan_to_num(cm_percent)  # Handle division by zero
        
        # Create heatmap
        sns.heatmap(cm_percent, annot=cm, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    ax=ax, cbar_kws={'label': 'Percentage (%)'})
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(title, fontweight='bold')
        
        # Rotate labels if too many
        if len(labels) > 5:
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticklabels(labels, rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        
        self.figures.append(fig)
        
        try:
            plt.show()
        except:
            print("  (Plot displayed in non-interactive mode)")

    def plot_module_comparison(self,
                               comparison: Dict[str, Dict[str, float]],
                               title: str = "Module Performance Comparison",
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 6)):
        """Plot comparison of module performance."""
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        modules = list(comparison.keys())
        data = {metric: [comparison[m].get(metric, 0) for m in modules] 
                for metric in metrics_to_plot}
        
        x = np.arange(len(modules))
        width = 0.2
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, (metric, values) in enumerate(data.items()):
            offset = (i - 1.5) * width
            bars = ax.bar(x + offset, values, width, label=metric.capitalize(), 
                         color=colors[i], alpha=0.8)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Module')
        ax.set_ylabel('Score')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(modules, rotation=45, ha='right')
        ax.set_ylim(0, 1.15)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        
        self.figures.append(fig)
        
        try:
            plt.show()
        except:
            print("  (Plot displayed in non-interactive mode)")

    def plot_diagnosis_distribution(self,
                                    diagnoses: List[str],
                                    title: str = "Diagnosis Distribution",
                                    save_path: Optional[str] = None,
                                    figsize: Tuple[int, int] = (10, 6)):
        """Plot distribution of diagnoses."""
        counter = Counter(diagnoses)
        labels = list(counter.keys())
        values = list(counter.values())
        
        if not labels:
            print("  No diagnoses to plot")
            return
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Bar chart
        bars = ax1.bar(labels, values, color=colors, edgecolor='black', linewidth=1)
        ax1.set_xlabel('Diagnosis')
        ax1.set_ylabel('Count')
        ax1.set_title('Diagnosis Counts', fontweight='bold')
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(value), ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        ax2.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Diagnosis Proportions', fontweight='bold')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        
        self.figures.append(fig)
        
        try:
            plt.show()
        except:
            print("  (Plot displayed in non-interactive mode)")

    def plot_urgency_distribution(self,
                                  urgencies: List[str],
                                  title: str = "Urgency Distribution",
                                  save_path: Optional[str] = None,
                                  figsize: Tuple[int, int] = (8, 6)):
        """Plot distribution of urgencies."""
        urgency_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
        urgency_colors = {'CRITICAL': '#FF6B6B', 'HIGH': '#FFD93D', 
                          'MEDIUM': '#FFB347', 'LOW': '#6BCB77'}
        
        counter = Counter(urgencies)
        counts = [counter.get(u, 0) for u in urgency_order]
        
        colors = [urgency_colors.get(u, '#95A5A6') for u in urgency_order]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.bar(urgency_order, counts, color=colors, edgecolor='black', linewidth=2)
        
        ax.set_xlabel('Urgency Level')
        ax.set_ylabel('Count')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        
        self.figures.append(fig)
        
        try:
            plt.show()
        except:
            print("  (Plot displayed in non-interactive mode)")

    def plot_confidence_calibration(self,
                                    confidences: List[float],
                                    correctness: List[bool],
                                    title: str = "Confidence Calibration",
                                    save_path: Optional[str] = None,
                                    figsize: Tuple[int, int] = (10, 6)):
        """Plot confidence calibration curve."""
        if not confidences or not correctness:
            print("  No data for confidence calibration")
            return
        
        # Create bins
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_indices = np.digitize(confidences, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 1)
        
        # Calculate accuracy per bin
        bin_accuracies = []
        bin_counts = []
        for i in range(len(bins) - 1):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_accuracies.append(np.mean(correctness[mask]))
                bin_counts.append(np.sum(mask))
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot calibration curve
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        
        # Plot data points with size proportional to count
        sizes = [c * 20 for c in bin_counts]
        scatter = ax.scatter(bin_centers, bin_accuracies, s=sizes, 
                           c=bin_accuracies, cmap='RdYlGn', 
                           vmin=0, vmax=1, alpha=0.7)
        
        # Connect points
        valid_indices = [i for i, c in enumerate(bin_counts) if c > 0]
        if valid_indices:
            valid_centers = [bin_centers[i] for i in valid_indices]
            valid_accuracies = [bin_accuracies[i] for i in valid_indices]
            ax.plot(valid_centers, valid_accuracies, 'b-', alpha=0.5)
        
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(title, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Accuracy')
        
        # Add histogram of confidence distribution
        ax_hist = ax.inset_axes([0.7, 0.1, 0.25, 0.25])
        ax_hist.hist(confidences, bins=20, color='gray', alpha=0.7, edgecolor='black')
        ax_hist.set_xlim(0, 1)
        ax_hist.set_xlabel('Confidence')
        ax_hist.set_ylabel('Count')
        ax_hist.set_title('Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        
        self.figures.append(fig)
        
        try:
            plt.show()
        except:
            print("  (Plot displayed in non-interactive mode)")


# ── Test Visualizer ──────────────────────────────────────
def test_visualizer():
    """Test the Visualizer module"""
    print("\n" + "=" * 60)
    print("  TESTING VISUALIZER")
    print("=" * 60)
    
    viz = Visualizer()
    
    # Test confusion matrix
    print("\n--- Confusion Matrix ---")
    cm = np.array([[10, 2, 1], [1, 8, 3], [0, 1, 14]])
    labels = ['covid19', 'flu', 'common_cold']
    viz.plot_confusion_matrix(cm, labels, title="Test Confusion Matrix")
    
    # Test module comparison
    print("\n--- Module Comparison ---")
    comparison = {
        'KnowledgeBase': {'accuracy': 0.72, 'precision': 0.70, 'recall': 0.72, 'f1': 0.71},
        'BayesianNet': {'accuracy': 0.78, 'precision': 0.76, 'recall': 0.78, 'f1': 0.77},
        'MLClassifier': {'accuracy': 0.85, 'precision': 0.84, 'recall': 0.85, 'f1': 0.84},
        'NeuralNetwork': {'accuracy': 0.82, 'precision': 0.81, 'recall': 0.82, 'f1': 0.81}
    }
    viz.plot_module_comparison(comparison, title="Module Performance Test")
    
    # Test distributions
    print("\n--- Diagnosis Distribution ---")
    diagnoses = ['covid19', 'flu', 'covid19', 'common_cold', 'flu', 'covid19', 
                 'covid19', 'dengue', 'flu', 'common_cold']
    viz.plot_diagnosis_distribution(diagnoses, title="Test Diagnosis Distribution")
    
    # Test urgency distribution
    print("\n--- Urgency Distribution ---")
    urgencies = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'HIGH', 'MEDIUM', 
                 'LOW', 'CRITICAL', 'HIGH', 'MEDIUM']
    viz.plot_urgency_distribution(urgencies, title="Test Urgency Distribution")
    
    print("\n✅ Visualizer test passed!")


if __name__ == "__main__":
    test_visualizer()