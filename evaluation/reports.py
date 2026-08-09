# ============================================================
# EVALUATION REPORTS
# Generate comprehensive evaluation reports
# ============================================================

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class ReportGenerator:
    """
    Generate comprehensive evaluation reports in various formats.
    Supports JSON, CSV, and text reports.
    """

    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_json_report(self, 
                             metrics: Dict[str, Any],
                             filename: Optional[str] = None) -> str:
        """
        Generate a JSON report.
        
        Args:
            metrics: Dictionary of metrics
            filename: Output filename (optional)
            
        Returns:
            Path to the generated file
        """
        if filename is None:
            filename = f"report_{self.timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Add metadata
        report = {
            'timestamp': self.timestamp,
            'generated_at': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ JSON report saved: {filepath}")
        return str(filepath)

    def generate_csv_report(self,
                            patient_results: List[Dict],
                            filename: Optional[str] = None) -> str:
        """
        Generate a CSV report from patient results.
        
        Args:
            patient_results: List of patient result dictionaries
            filename: Output filename (optional)
            
        Returns:
            Path to the generated file
        """
        if filename is None:
            filename = f"patient_results_{self.timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        # Convert to DataFrame
        df = pd.DataFrame(patient_results)
        
        # Flatten module results if present
        if 'module_results' in df.columns:
            # Expand module results into separate columns
            expanded = []
            for _, row in df.iterrows():
                new_row = row.to_dict()
                if isinstance(row['module_results'], dict):
                    for module, result in row['module_results'].items():
                        if isinstance(result, dict):
                            new_row[f'module_{module}_diagnosis'] = result.get('diagnosis', '')
                            new_row[f'module_{module}_confidence'] = result.get('confidence', 0)
                del new_row['module_results']
                expanded.append(new_row)
            df = pd.DataFrame(expanded)
        
        df.to_csv(filepath, index=False)
        print(f"✅ CSV report saved: {filepath}")
        return str(filepath)

    def generate_text_report(self,
                            metrics: Dict[str, Any],
                            filename: Optional[str] = None) -> str:
        """
        Generate a human-readable text report.
        
        Args:
            metrics: Dictionary of metrics
            filename: Output filename (optional)
            
        Returns:
            Path to the generated file
        """
        if filename is None:
            filename = f"report_{self.timestamp}.txt"
        
        filepath = self.output_dir / filename
        
        lines = []
        lines.append("=" * 60)
        lines.append("  AI HEALTHCARE DIAGNOSTIC SYSTEM - EVALUATION REPORT")
        lines.append("=" * 60)
        lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)
        lines.append("")
        
        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 40)
        n_patients = metrics.get('n_patients', 0)
        lines.append(f"  Total Patients: {n_patients}")
        
        if 'accuracy' in metrics:
            lines.append(f"  Accuracy: {metrics['accuracy']:.2%}")
        if 'avg_confidence' in metrics:
            lines.append(f"  Avg Confidence: {metrics['avg_confidence']:.2%}")
        if 'n_correct' in metrics:
            lines.append(f"  Correct Predictions: {metrics['n_correct']}")
        lines.append("")
        
        # Diagnosis distribution
        diag_dist = metrics.get('diagnosis_distribution', {})
        if diag_dist:
            lines.append("DIAGNOSIS DISTRIBUTION")
            lines.append("-" * 40)
            for diagnosis, count in sorted(diag_dist.items(), key=lambda x: -x[1]):
                percentage = (count / n_patients) * 100 if n_patients > 0 else 0
                lines.append(f"  {diagnosis}: {count} ({percentage:.1f}%)")
            lines.append("")
        
        # Urgency distribution
        urgency_dist = metrics.get('urgency_distribution', {})
        if urgency_dist:
            lines.append("URGENCY DISTRIBUTION")
            lines.append("-" * 40)
            urgency_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
            for urgency in urgency_order:
                count = urgency_dist.get(urgency, 0)
                percentage = (count / n_patients) * 100 if n_patients > 0 else 0
                if count > 0:
                    lines.append(f"  {urgency}: {count} ({percentage:.1f}%)")
            lines.append("")
        
        # Module summary
        module_summary = metrics.get('module_summary', {})
        if module_summary:
            lines.append("MODULE PERFORMANCE")
            lines.append("-" * 40)
            for module, summary in module_summary.items():
                avg_conf = summary.get('avg_confidence', 0)
                n_pred = summary.get('n_predictions', 0)
                lines.append(f"  {module}: Avg Confidence={avg_conf:.2%}, Predictions={n_pred}")
            lines.append("")
        
        # Confidence metrics
        conf_metrics = metrics.get('confidence_metrics', {})
        if conf_metrics:
            lines.append("CONFIDENCE METRICS")
            lines.append("-" * 40)
            lines.append(f"  ECE (Expected Calibration Error): {conf_metrics.get('ece', 0):.3f}")
            lines.append(f"  Overall Accuracy: {conf_metrics.get('overall_accuracy', 0):.2%}")
            lines.append(f"  Avg Confidence: {conf_metrics.get('avg_confidence', 0):.2%}")
            lines.append("")
            lines.append("  Accuracy by Confidence Threshold:")
            for key, value in conf_metrics.items():
                if key.startswith('accuracy_at_'):
                    threshold = key.replace('accuracy_at_', '')
                    lines.append(f"    Confidence >= {threshold}: {value:.2%}")
        
        # Time metrics
        time_metrics = metrics.get('time_metrics', {})
        if time_metrics:
            lines.append("PROCESSING TIME METRICS")
            lines.append("-" * 40)
            lines.append(f"  Mean Time: {time_metrics.get('mean_time', 0):.3f}s")
            lines.append(f"  Median Time: {time_metrics.get('median_time', 0):.3f}s")
            lines.append(f"  Total Time: {time_metrics.get('total_time', 0):.3f}s")
            lines.append("")
        
        lines.append("=" * 60)
        lines.append("  End of Report")
        lines.append("=" * 60)
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ Text report saved: {filepath}")
        return str(filepath)

    def generate_html_report(self,
                             metrics: Dict[str, Any],
                             patient_results: List[Dict],
                             filename: Optional[str] = None) -> str:
        """
        Generate an HTML report.
        
        Args:
            metrics: Dictionary of metrics
            patient_results: List of patient result dictionaries
            filename: Output filename (optional)
            
        Returns:
            Path to the generated file
        """
        if filename is None:
            filename = f"report_{self.timestamp}.html"
        
        filepath = self.output_dir / filename
        
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append("  <meta charset='UTF-8'>")
        html.append("  <title>AI Healthcare System Evaluation Report</title>")
        html.append("  <style>")
        html.append("    body { font-family: Arial, sans-serif; margin: 20px; }")
        html.append("    h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }")
        html.append("    h2 { color: #34495e; margin-top: 20px; }")
        html.append("    .summary { background: #ecf0f1; padding: 15px; border-radius: 5px; }")
        html.append("    .metric { display: inline-block; margin: 10px 20px; }")
        html.append("    .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }")
        html.append("    .metric-label { color: #7f8c8d; }")
        html.append("    table { border-collapse: collapse; width: 100%; margin: 10px 0; }")
        html.append("    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("    th { background: #3498db; color: white; }")
        html.append("    tr:nth-child(even) { background: #f2f2f2; }")
        html.append("    .good { color: #27ae60; }")
        html.append("    .warning { color: #f39c12; }")
        html.append("    .danger { color: #e74c3c; }")
        html.append("  </style>")
        html.append("</head>")
        html.append("<body>")
        
        # Header
        html.append(f"<h1>AI Healthcare Diagnostic System</h1>")
        html.append(f"<p>Evaluation Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        
        # Summary
        html.append("<h2>Summary</h2>")
        html.append("<div class='summary'>")
        
        n_patients = metrics.get('n_patients', 0)
        accuracy = metrics.get('accuracy', 0)
        avg_conf = metrics.get('avg_confidence', 0)
        
        html.append(f"<div class='metric'><div class='metric-value'>{n_patients}</div><div class='metric-label'>Patients</div></div>")
        html.append(f"<div class='metric'><div class='metric-value'>{accuracy:.1%}</div><div class='metric-label'>Accuracy</div></div>")
        html.append(f"<div class='metric'><div class='metric-value'>{avg_conf:.1%}</div><div class='metric-label'>Avg Confidence</div></div>")
        html.append("</div>")
        
        # Patient Results Table
        html.append("<h2>Patient Results</h2>")
        html.append("<table>")
        html.append("  <tr><th>Patient ID</th><th>Predicted Diagnosis</th><th>Confidence</th><th>Urgency</th><th>Correct</th></tr>")
        
        for result in patient_results[:50]:  # Limit to 50 rows
            correct = result.get('correct')
            correct_str = "✅" if correct else "❌" if correct is not None else "—"
            html.append(f"  <tr>")
            html.append(f"    <td>{result.get('patient_id', 'N/A')}</td>")
            html.append(f"    <td>{result.get('predicted_diagnosis', 'N/A')}</td>")
            html.append(f"    <td>{result.get('confidence', 0):.1%}</td>")
            html.append(f"    <td>{result.get('urgency', 'N/A')}</td>")
            html.append(f"    <td>{correct_str}</td>")
            html.append(f"  </tr>")
        
        if len(patient_results) > 50:
            html.append(f"  <tr><td colspan='5'>... and {len(patient_results) - 50} more patients</td></tr>")
        
        html.append("</table>")
        
        # Module Comparison
        module_summary = metrics.get('module_summary', {})
        if module_summary:
            html.append("<h2>Module Performance</h2>")
            html.append("<table>")
            html.append("  <tr><th>Module</th><th>Avg Confidence</th><th>Predictions</th></tr>")
            for module, summary in module_summary.items():
                html.append(f"  <tr>")
                html.append(f"    <td>{module}</td>")
                html.append(f"    <td>{summary.get('avg_confidence', 0):.1%}</td>")
                html.append(f"    <td>{summary.get('n_predictions', 0)}</td>")
                html.append(f"  </tr>")
            html.append("</table>")
        
        # Distribution
        diag_dist = metrics.get('diagnosis_distribution', {})
        if diag_dist:
            html.append("<h2>Diagnosis Distribution</h2>")
            html.append("<ul>")
            for diagnosis, count in sorted(diag_dist.items(), key=lambda x: -x[1]):
                percentage = (count / n_patients) * 100 if n_patients > 0 else 0
                html.append(f"  <li>{diagnosis}: {count} ({percentage:.1f}%)</li>")
            html.append("</ul>")
        
        html.append("<hr>")
        html.append("<p><em>Generated by AI Healthcare Diagnostic System Evaluation Module</em></p>")
        html.append("</body>")
        html.append("</html>")
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(html))
        
        print(f"✅ HTML report saved: {filepath}")
        return str(filepath)


# ── Test Report Generator ────────────────────────────────
def test_report_generator():
    """Test the ReportGenerator module"""
    print("\n" + "=" * 60)
    print("  TESTING REPORT GENERATOR")
    print("=" * 60)
    
    rg = ReportGenerator()
    
    # Sample metrics
    metrics = {
        'n_patients': 10,
        'accuracy': 0.85,
        'avg_confidence': 0.78,
        'n_correct': 8,
        'diagnosis_distribution': {
            'covid19': 4,
            'flu': 3,
            'common_cold': 2,
            'dengue': 1
        },
        'urgency_distribution': {
            'CRITICAL': 1,
            'HIGH': 3,
            'MEDIUM': 4,
            'LOW': 2
        },
        'module_summary': {
            'KnowledgeBase': {'avg_confidence': 0.72, 'n_predictions': 10},
            'BayesianNet': {'avg_confidence': 0.76, 'n_predictions': 10},
            'MLClassifier': {'avg_confidence': 0.85, 'n_predictions': 10}
        }
    }
    
    # Sample patient results
    patient_results = [
        {
            'patient_id': 'P001',
            'predicted_diagnosis': 'covid19',
            'confidence': 0.85,
            'urgency': 'HIGH',
            'correct': True
        },
        {
            'patient_id': 'P002',
            'predicted_diagnosis': 'flu',
            'confidence': 0.72,
            'urgency': 'MEDIUM',
            'correct': True
        },
        {
            'patient_id': 'P003',
            'predicted_diagnosis': 'common_cold',
            'confidence': 0.65,
            'urgency': 'LOW',
            'correct': False
        }
    ]
    
    # Generate reports
    print("\n--- Generating Reports ---")
    rg.generate_json_report(metrics)
    rg.generate_text_report(metrics)
    rg.generate_csv_report(patient_results)
    rg.generate_html_report(metrics, patient_results)
    
    print("\n✅ Report Generator test passed!")


if __name__ == "__main__":
    test_report_generator()