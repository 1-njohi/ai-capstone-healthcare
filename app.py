# ============================================================
# CAPSTONE MAIN APPLICATION
# Intelligent Healthcare Diagnostic Assistant
# Introduction to AI — 13-Week Capstone
# ============================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
import json
import warnings
import numpy as np
import matplotlib
# Use non-interactive backend for headless environments
try:
    matplotlib.use('Agg')
except:
    pass
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
import os

warnings.filterwarnings('ignore')

# Import from modules
from modules.agent import HealthcareDiagnosticAgent, PatientPercept
from modules.knowledge_base import MedicalKnowledgeBase
from modules.bayesian_net import SimpleBayesianDiagnostics
from modules.ml_classifier import MLDiagnosticClassifier
from modules.neural_network import NeuralDiagnosticModel
from modules.fuzzy_controller import FuzzySeverityAssessor
from modules.planner import TreatmentPlanner

# Import evaluation module
from evaluation import MetricsCalculator, Visualizer, ReportGenerator

# ── ANSI Colors ──────────────────────────────────────────
class C:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def banner():
    print(f"""
{C.BOLD}{C.BLUE}╔══════════════════════════════════════════════════════════════════╗
║        🏥  INTELLIGENT HEALTHCARE DIAGNOSTIC ASSISTANT           ║
║         Introduction to Artificial Intelligence — Capstone       ║
║  Modules: Agents | Logic | Bayes | ML | DNN | Fuzzy | Plan       ║
╚══════════════════════════════════════════════════════════════════╝{C.END}
""")
    
def clean_module_results(report):
    """Clean up module results to show proper diagnoses"""
    if not isinstance(report, dict):
        return report
    
    if 'module_results' in report:
        nn = None
        # Try to get NeuralNetwork instance
        try:
            from modules.neural_network import NeuralDiagnosticModel
            nn = NeuralDiagnosticModel()
        except:
            pass
        
        for module, result in report['module_results'].items():
            if module == 'NeuralNetwork' and isinstance(result, dict):
                diag = result.get('diagnosis', '')
                # Check if it's a number
                try:
                    if isinstance(diag, (int, float)) or (isinstance(diag, str) and diag.isdigit()):
                        idx = int(diag)
                        if nn and hasattr(nn, 'DISEASE_LABELS') and 0 <= idx < len(nn.DISEASE_LABELS):
                            result['diagnosis'] = nn.DISEASE_LABELS[idx]
                except:
                    pass
    
    return report

def section(title):
    print(f"\n{C.BOLD}{C.YELLOW}{'═'*60}{C.END}")
    print(f"{C.BOLD}{C.YELLOW}  {title}{C.END}")
    print(f"{C.BOLD}{C.YELLOW}{'═'*60}{C.END}")

def safe_get(data, key, default='N/A'):
    """Safely get a value from a dict or object"""
    if data is None:
        return default
    if isinstance(data, dict):
        return data.get(key, default)
    if hasattr(data, key):
        return getattr(data, key, default)
    return default

def print_report(report):
    """Print diagnosis report - handles both dict and object"""
    print(f"\n{C.BOLD}{C.GREEN}{'═'*60}{C.END}")
    print(f"{C.BOLD}{C.GREEN}  📋 DIAGNOSIS REPORT{C.END}")
    print(f"{C.BOLD}{C.GREEN}{'═'*60}{C.END}")
    
    patient_id = safe_get(report, 'patient_id')
    diagnosis = safe_get(report, 'diagnosis')
    confidence = safe_get(report, 'confidence', 0)
    urgency = safe_get(report, 'urgency')
    
    print(f"\n{C.BOLD}Patient:{C.END} {patient_id}")
    print(f"{C.BOLD}Diagnosis:{C.END} {diagnosis}")
    if isinstance(confidence, (int, float)):
        print(f"{C.BOLD}Confidence:{C.END} {confidence:.1%}")
    else:
        print(f"{C.BOLD}Confidence:{C.END} {confidence}")
    print(f"{C.BOLD}Urgency:{C.END} {urgency}")
    
    recommendations = safe_get(report, 'recommendations', [])
    if recommendations:
        print(f"\n{C.BOLD}Recommendations:{C.END}")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"  {i}. {rec}")
    
    next_action = safe_get(report, 'next_action')
    if next_action:
        print(f"\n{C.BOLD}Next Action:{C.END} {next_action}")
    
    # Show module breakdown if available
    module_results = safe_get(report, 'module_results', {})
    if module_results:
        print(f"\n{C.BOLD}Module Breakdown:{C.END}")
        for module, result in module_results.items():
            if isinstance(result, dict):
                diag = result.get('diagnosis', 'N/A')
                conf = result.get('confidence', 0)
                print(f"  {module}: {diag} ({conf:.1%})")
    
    print(f"\n{C.GREEN}{'═'*60}{C.END}")

# ── Fix for Planner's analyze method ──────────────────────
def fix_planner_analyze(planner):
    """Wrap the planner's analyze method to handle errors gracefully"""
    original_analyze = planner.analyze
    
    def safe_analyze(percept):
        try:
            result = original_analyze(percept)
            if isinstance(result, dict):
                if 'steps' not in result:
                    result['steps'] = 0
                    result['summary'] = "Plan generation failed"
                    result['diagnosis'] = 'unknown'
                    result['confidence'] = 0.5
                return result
            return result
        except Exception as e:
            print(f"    [Planner] Error: {e}")
            return {
                'diagnosis': 'unknown',
                'confidence': 0.5,
                'steps': 0,
                'summary': f"Plan error: {e}",
                'plan': [],
                'error': str(e)
            }
    
    planner.analyze = safe_analyze
    return planner

def build_system():
    """Build and register all modules"""
    section("🔧 Building AI System")
    
    agent = HealthcareDiagnosticAgent()
    print("\n  Registering modules...")
    
    # Create modules with fixes
    modules = {
        'KnowledgeBase': MedicalKnowledgeBase(),
        'BayesianNet': SimpleBayesianDiagnostics(),
        'MLClassifier': MLDiagnosticClassifier(),
        'NeuralNetwork': NeuralDiagnosticModel(),
        'FuzzyController': FuzzySeverityAssessor(),
        'Planner': fix_planner_analyze(TreatmentPlanner())
    }
    
    for name, module in modules.items():
        agent.register_module(name, module)
    
    # Train ML models
    print("\n  Training ML models...")
    try:
        agent._modules['MLClassifier'].train(verbose=False)
        print(f"    {C.GREEN}✅ ML Classifier trained{C.END}")
    except Exception as e:
        print(f"    {C.YELLOW}⚠️ ML Classifier: {e}{C.END}")
    
    try:
        agent._modules['NeuralNetwork'].train(epochs=20, verbose=False)
        print(f"    {C.GREEN}✅ Neural Network trained{C.END}")
    except Exception as e:
        print(f"    {C.YELLOW}⚠️ Neural Network: {e}{C.END}")
    
    print(f"\n{C.GREEN}✅ System ready! {len(modules)} modules registered{C.END}")
    return agent

def run_tests(agent):
    """Run test patients"""
    section("🧪 Testing Sample Patients")
    
    test_patients = [
        # COVID-19
        PatientPercept("P001", ["fever", "cough", "loss_of_smell", "fatigue"], 34, 38.5, 98, "120/80"),
        # Flu
        PatientPercept("P002", ["fever", "cough", "body_aches", "headache"], 45, 39.0, 105, "130/85"),
        # Common Cold
        PatientPercept("P003", ["runny_nose", "sneezing", "sore_throat"], 28, 37.2, 72, "118/76"),
        # Dengue
        PatientPercept("P004", ["high_fever", "joint_pain", "rash"], 32, 39.8, 115, "125/82"),
        # Cardiac Event
        PatientPercept("P005", ["chest_pain", "shortness_of_breath", "sweating"], 55, 37.0, 125, "150/90"),
        # Diabetes
        PatientPercept("P006", ["fatigue", "frequent_urination", "excessive_thirst"], 52, 36.8, 82, "135/85"),
        # Tuberculosis
        PatientPercept("P007", ["cough", "weight_loss", "night_sweats", "fatigue"], 40, 38.2, 95, "125/80"),
        # Meningitis
        PatientPercept("P008", ["headache", "stiff_neck", "high_fever"], 26, 39.5, 110, "120/75"),
        # Healthy
        PatientPercept("P009", ["slight_fatigue"], 30, 36.5, 68, "115/70"),
    ]
    
    results = []
    for i, patient in enumerate(test_patients, 1):
        print(f"\n{C.GREEN}Patient {i}{C.END}")
        print(f"  Symptoms: {', '.join(patient.symptoms)}")
        print(f"  Temp: {patient.temperature}°C, HR: {patient.heart_rate} BPM")
        try:
            report = agent.run(patient)
            results.append({
                'patient_id': patient.patient_id,
                'report': report,
                'success': True,
                'symptoms': patient.symptoms,
                'temperature': patient.temperature,
                'heart_rate': patient.heart_rate
            })
            
            diagnosis = safe_get(report, 'diagnosis')
            confidence = safe_get(report, 'confidence', 0)
            urgency = safe_get(report, 'urgency')
            
            print(f"  {C.BOLD}Diagnosis:{C.END} {diagnosis}")
            if isinstance(confidence, (int, float)):
                print(f"  Confidence: {confidence:.1%}")
            else:
                print(f"  Confidence: {confidence}")
            print(f"  Urgency: {urgency}")
        except Exception as e:
            results.append({
                'patient_id': patient.patient_id,
                'success': False,
                'error': str(e)
            })
            print(f"  {C.RED}Error: {e}{C.END}")
    
    successful = sum(1 for r in results if r['success'])
    print(f"\n{C.GREEN}✅ {successful}/{len(results)} successful{C.END}")
    return results

def run_detailed_diagnosis(agent):
    """Run a detailed diagnosis for a single patient"""
    section("🔍 Detailed Diagnosis")
    
    print("\n  Enter patient information for detailed diagnosis")
    pid = input("  Patient ID: ").strip()
    symptoms_input = input("  Symptoms (comma-separated): ").strip()
    symptoms = [s.strip() for s in symptoms_input.split(',') if s.strip()]
    
    try:
        age = int(input("  Age: "))
        temp = float(input("  Temperature (°C): "))
        hr = int(input("  Heart Rate (BPM): "))
        bp = input("  Blood Pressure: ")
    except ValueError:
        print(f"{C.RED}Invalid input.{C.END}")
        return
    
    if not symptoms:
        print(f"{C.RED}Please enter at least one symptom.{C.END}")
        return
    
    patient = PatientPercept(pid, symptoms, age, temp, hr, bp)
    print(f"\n  {C.BOLD}Analyzing...{C.END}")
    
    try:
        report = agent.run(patient)
        print_report(report)
    except Exception as e:
        print(f"{C.RED}Error: {e}{C.END}")

def interactive_mode(agent):
    """Interactive CLI mode"""
    section("💬 Interactive Mode")
    print("\n  Enter patient data or 'quit' to exit\n")
    
    count = 0
    while True:
        pid = input("  Patient ID (or 'quit'): ").strip()
        if pid.lower() in ['quit', 'exit', 'q']:
            break
        
        symptoms_input = input("  Symptoms (comma-separated): ").strip()
        symptoms = [s.strip() for s in symptoms_input.split(',') if s.strip()]
        
        try:
            age = int(input("  Age: "))
            temp = float(input("  Temperature (°C): "))
            hr = int(input("  Heart Rate (BPM): "))
            bp = input("  Blood Pressure: ")
        except ValueError:
            print(f"{C.RED}Invalid input. Try again.{C.END}")
            continue
        
        if not symptoms:
            print(f"{C.RED}Please enter at least one symptom.{C.END}")
            continue
        
        patient = PatientPercept(pid, symptoms, age, temp, hr, bp)
        print(f"\n  {C.BOLD}Analyzing...{C.END}")
        
        try:
            report = agent.run(patient)
            count += 1
            print_report(report)
        except Exception as e:
            print(f"{C.RED}Error: {e}{C.END}")
    
    print(f"\n{C.GREEN}Diagnosed {count} patients{C.END}")

def run_evaluation(agent, results):
    """Run evaluation on test results and generate visualizations"""
    section("📊 System Evaluation")
    
    if not results:
        print(f"{C.YELLOW}No results to evaluate. Please run tests first.{C.END}")
        return
    
    successful_results = [r for r in results if r['success']]
    if not successful_results:
        print(f"{C.YELLOW}No successful results to evaluate.{C.END}")
        return
    
    print(f"\n  Evaluating {len(successful_results)} successful diagnoses...")
    
    # Collect metrics
    metrics_calc = MetricsCalculator()
    
    predictions = []
    confidences = []
    urgencies = []
    module_results = {}
    
    for r in successful_results:
        report = r['report']
        predictions.append(report.get('diagnosis', 'unknown'))
        confidences.append(report.get('confidence', 0))
        urgencies.append(report.get('urgency', 'LOW'))
        
        # Collect module results
        if 'module_results' in report:
            for module, result in report['module_results'].items():
                if module not in module_results:
                    module_results[module] = {'diagnoses': [], 'confidences': []}
                if isinstance(result, dict):
                    module_results[module]['diagnoses'].append(result.get('diagnosis', 'unknown'))
                    module_results[module]['confidences'].append(result.get('confidence', 0))
    
    # Calculate aggregate metrics
    aggregate = {
        'n_patients': len(successful_results),
        'avg_confidence': np.mean(confidences) if confidences else 0,
        'diagnosis_distribution': dict(Counter(predictions)),
        'urgency_distribution': dict(Counter(urgencies)),
        'module_summary': {
            name: {
                'avg_confidence': np.mean(data['confidences']) if data['confidences'] else 0,
                'n_predictions': len(data['confidences'])
            }
            for name, data in module_results.items()
        }
    }
    
    # Print summary
    print(f"\n{C.BOLD}Diagnosis Distribution:{C.END}")
    for diag, count in sorted(aggregate['diagnosis_distribution'].items(), key=lambda x: -x[1]):
        pct = count / aggregate['n_patients'] * 100
        print(f"  {diag}: {count} ({pct:.1f}%)")
    
    print(f"\n{C.BOLD}Urgency Distribution:{C.END}")
    urgency_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
    for urgency in urgency_order:
        count = aggregate['urgency_distribution'].get(urgency, 0)
        pct = count / aggregate['n_patients'] * 100 if aggregate['n_patients'] > 0 else 0
        if count > 0:
            print(f"  {urgency}: {count} ({pct:.1f}%)")
    
    print(f"\n{C.BOLD}Module Confidence Averages:{C.END}")
    for module, summary in aggregate['module_summary'].items():
        print(f"  {module}: {summary['avg_confidence']:.1%}")
    
    # Generate visualizations
    try:
        viz = Visualizer()
        
        # Create output directory
        os.makedirs('evaluation_output', exist_ok=True)
        
        # 1. Diagnosis distribution
        diagnoses = [r['report'].get('diagnosis', 'unknown') for r in successful_results]
        viz.plot_diagnosis_distribution(
            diagnoses,
            title="Diagnosis Distribution",
            save_path='evaluation_output/diagnosis_distribution.png'
        )
        
        # 2. Urgency distribution
        urgencies_list = [r['report'].get('urgency', 'LOW') for r in successful_results]
        viz.plot_urgency_distribution(
            urgencies_list,
            title="Urgency Distribution",
            save_path='evaluation_output/urgency_distribution.png'
        )
        
        # 3. Module comparison
        comparison = {}
        for module, data in module_results.items():
            # Calculate accuracy relative to final diagnosis
            final_diagnoses = [r['report'].get('diagnosis', 'unknown') for r in successful_results]
            module_diagnoses = data['diagnoses']
            # Simple accuracy: compare module diagnosis to final diagnosis
            correct = sum(1 for md, fd in zip(module_diagnoses, final_diagnoses) if md == fd)
            acc = correct / len(final_diagnoses) if final_diagnoses else 0
            comparison[module] = {
                'accuracy': acc,
                'precision': acc,  # Simplified for now
                'recall': acc,
                'f1': acc
            }
        
        viz.plot_module_comparison(
            comparison,
            title="Module Performance Comparison",
            save_path='evaluation_output/module_comparison.png'
        )
        
        # 4. Confidence calibration
        confidences_list = [r['report'].get('confidence', 0) for r in successful_results]
        # We don't have ground truth, so use confidence vs accuracy proxy
        # For now, skip calibration plot or use simple distribution
        if confidences_list:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(confidences_list, bins=20, color='#4ECDC4', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Count')
            ax.set_title('Confidence Distribution', fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('evaluation_output/confidence_distribution.png', dpi=150, bbox_inches='tight')
            print(f"✅ Saved: evaluation_output/confidence_distribution.png")
            plt.close()
        
        print(f"\n{C.GREEN}✅ Evaluation visualizations saved to 'evaluation_output/'{C.END}")
        
    except Exception as e:
        print(f"{C.YELLOW}⚠️ Could not generate visualizations: {e}{C.END}")
    
    # Generate reports
    try:
        report_gen = ReportGenerator('evaluation_output')
        
        # JSON report
        report_gen.generate_json_report(aggregate, 'metrics_report.json')
        
        # Text report
        report_gen.generate_text_report(aggregate, 'metrics_report.txt')
        
        # CSV report
        patient_data = []
        for r in successful_results:
            row = {
                'patient_id': r['patient_id'],
                'diagnosis': r['report'].get('diagnosis', 'unknown'),
                'confidence': r['report'].get('confidence', 0),
                'urgency': r['report'].get('urgency', 'LOW'),
                'temperature': r.get('temperature', 0),
                'heart_rate': r.get('heart_rate', 0)
            }
            # Add module results
            if 'module_results' in r['report']:
                for module, result in r['report']['module_results'].items():
                    if isinstance(result, dict):
                        row[f'module_{module}_diagnosis'] = result.get('diagnosis', '')
                        row[f'module_{module}_confidence'] = result.get('confidence', 0)
            patient_data.append(row)
        
        report_gen.generate_csv_report(patient_data, 'patient_results.csv')
        
        print(f"\n{C.GREEN}✅ Reports saved to 'evaluation_output/'{C.END}")
        
    except Exception as e:
        print(f"{C.YELLOW}⚠️ Could not generate reports: {e}{C.END}")

def show_info(agent):
    """Show system info"""
    section("ℹ️ System Information")
    perf = agent.get_performance()
    
    print(f"\n  Registered Modules: {len(agent._modules)}")
    for name in agent._modules:
        print(f"    - {name}")
    
    print(f"\n  Performance:")
    print(f"    Patients: {perf['total_patients']}")
    print(f"    Diagnoses: {perf['diagnoses_made']}")  # Already an int
    print(f"    Score: {perf['performance_score']}")
    
    print(f"\n  {C.BOLD}Recent Actions:{C.END}")
    for entry in agent.memory.action_log[-5:]:
        print(f"    {entry}")

def get_best_diagnosis(agent, patient):
    """Get the best diagnosis by polling all modules"""
    agent.perceive(patient)
    results = agent.think()
    
    # Extract diagnoses from each module
    diagnoses = {}
    for name, result in results.items():
        if isinstance(result, DiagnosisResult):
            diag = result.diagnosis
            conf = result.confidence
            if diag not in ['error', 'unknown']:
                # Skip fuzzy controller's severity output
                if name == 'FuzzyController' and 'Severity:' in diag:
                    continue
                if name == 'Planner' and conf < 0.5:
                    continue
                diagnoses[name] = {'diagnosis': diag, 'confidence': conf}
    
    # Find the most confident diagnosis
    if diagnoses:
        best_name = max(diagnoses, key=lambda x: diagnoses[x]['confidence'])
        best = diagnoses[best_name]
        return best['diagnosis'], best['confidence'], diagnoses
    return 'unknown', 0, diagnoses

def main():
    """Main application entry point"""
    banner()
    agent = build_system()
    results = None
    
    while True:
        print(f"\n{C.BOLD}{C.BLUE}{'─'*50}{C.END}")
        print(f"{C.BOLD}  MAIN MENU{C.END}")
        print(f"{C.BOLD}{C.BLUE}{'─'*50}{C.END}")
        print(f"  1. Run Test Patients (9 cases)")
        print(f"  2. Detailed Diagnosis (single patient)")
        print(f"  3. Interactive Mode")
        print(f"  4. System Info")
        print(f"  5. Run Evaluation")
        print(f"  6. Exit")
        print(f"{C.BOLD}{C.BLUE}{'─'*50}{C.END}")
        
        choice = input("  Choice: ").strip()
        
        if choice == '1':
            results = run_tests(agent)
        elif choice == '2':
            run_detailed_diagnosis(agent)
        elif choice == '3':
            interactive_mode(agent)
        elif choice == '4':
            show_info(agent)
        elif choice == '5':
            run_evaluation(agent, results)
        elif choice == '6':
            print(f"\n{C.GREEN}Thank you for using the Intelligent Healthcare Diagnostic Assistant!{C.END}")
            print(f"{C.BLUE}Dedan Kimathi University of Technology - AI Capstone Project{C.END}\n")
            break
        else:
            print(f"{C.RED}Invalid choice. Please enter 1-6.{C.END}")
        
        if choice != '6':
            input(f"\n{C.YELLOW}Press Enter to continue...{C.END}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{C.YELLOW}Exiting...{C.END}")
    except Exception as e:
        print(f"\n{C.RED}Error: {e}{C.END}")
        import traceback
        traceback.print_exc()