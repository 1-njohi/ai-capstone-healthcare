# ============================================================
# MODULE 1: Intelligent Agent — Healthcare Diagnostic Agent
# Covers: Week 2 (Intelligent Agents) + PEAS Framework
# ============================================================

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import datetime
from collections import Counter
import numpy as np


class AgentState(Enum):
    IDLE         = "idle"
    COLLECTING   = "collecting_symptoms"
    DIAGNOSING   = "diagnosing"
    RECOMMENDING = "recommending"
    PLANNING     = "planning_treatment"
    DONE         = "done"


class UrgencyLevel(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class PatientPercept:
    patient_id:   str
    symptoms:     List[str]
    age:          int
    temperature:  float
    heart_rate:   int
    blood_pressure: str
    timestamp:    str = field(
        default_factory=lambda: datetime.datetime.now().isoformat())


@dataclass
class AgentMemory:
    patient_history:  List[Dict]  = field(default_factory=list)
    current_patient:  Optional[PatientPercept] = None
    diagnosis_history: List[str]  = field(default_factory=list)
    action_log:       List[str]   = field(default_factory=list)


@dataclass
class DiagnosisResult:
    module_name: str
    diagnosis: str
    confidence: float
    details: Dict[str, Any]


@dataclass
class FinalReport:
    patient_id: str
    final_diagnosis: str
    confidence: float
    urgency: UrgencyLevel
    module_results: List[DiagnosisResult]
    recommendations: List[str]
    treatment_plan: Dict[str, Any]
    timestamp: str


class HealthcareDiagnosticAgent:
    """
    PEAS Definition:
    Performance : Diagnostic accuracy, patient safety, recommendation quality, response time
    Environment : Hospital/clinic, patient data, EMR
    Actuators   : Diagnosis report, treatment plan, referral recommendation, alerts
    Sensors     : Symptom input, vitals, lab results, patient history
    Agent Type  : Model-Based + Goal-Based + Learning
    """

    def __init__(self):
        self.state = AgentState.IDLE
        self.memory = AgentMemory()
        self.performance_score = 0
        self._modules = {}

    def register_module(self, name: str, module):
        self._modules[name] = module
        print(f"  🔌 Module registered: [{name}]")

    def perceive(self, percept: PatientPercept):
        self.memory.current_patient = percept
        self.memory.patient_history.append({
            'id': percept.patient_id,
            'symptoms': percept.symptoms,
            'time': percept.timestamp
        })
        self.state = AgentState.COLLECTING
        self._log(f"Perceived patient {percept.patient_id} with {len(percept.symptoms)} symptoms")
        return self

    def think(self) -> Dict[str, DiagnosisResult]:
        if self.state == AgentState.IDLE:
            raise ValueError("Must call perceive() before think()")
        
        self.state = AgentState.DIAGNOSING
        self._log("Agent thinking: running diagnostic modules...")

        results = {}
        patient = self.memory.current_patient

        for module_name, module in self._modules.items():
            try:
                self._log(f"  [{module_name}] consulting...")
                
                if hasattr(module, 'analyze'):
                    result = module.analyze(patient)
                    
                    if isinstance(result, dict):
                        diagnosis = result.get('diagnosis', 'unknown')
                        confidence = result.get('confidence', 0.0)
                        details = result.get('details', {})
                    else:
                        diagnosis = str(result)
                        confidence = 0.5
                        details = {}
                    
                    # Convert numpy types to Python types for safety
                    if hasattr(diagnosis, 'item'):
                        diagnosis = str(diagnosis.item())
                    elif isinstance(diagnosis, (int, float)):
                        diagnosis = str(int(diagnosis))
                    
                    if isinstance(confidence, (np.float32, np.float64)):
                        confidence = float(confidence)
                    
                    diagnosis_result = DiagnosisResult(
                        module_name=module_name,
                        diagnosis=diagnosis,
                        confidence=confidence,
                        details=details
                    )
                    results[module_name] = diagnosis_result
                    
                    self._log(f"  [{module_name}] → {diagnosis} ({confidence:.1%})")
                else:
                    self._log(f"  [{module_name}] has no analyze() method")
                    
            except Exception as e:
                self._log(f"  [{module_name}] ERROR: {str(e)}")
                results[module_name] = DiagnosisResult(
                    module_name=module_name,
                    diagnosis='error',
                    confidence=0.0,
                    details={'error': str(e)}
                )

        self.memory.diagnosis_history.append(results)
        self.state = AgentState.RECOMMENDING
        return results

    def act(self, diagnosis_results: Dict[str, DiagnosisResult]) -> Dict:
        self.state = AgentState.PLANNING
        patient = self.memory.current_patient
        
        if patient is None:
            raise ValueError("No patient data available. Call perceive() first.")

        # 1. Aggregate diagnoses - filter out irrelevant modules
        diagnosis_votes = {}
        diagnosis_confidences = {}
        module_diagnoses = []
        mapped_results = {}

        # Modules that should NOT contribute to diagnosis
        skip_for_diagnosis = ['FuzzyController', 'Planner']
        
        for name, result in diagnosis_results.items():
            # Skip if error or unknown
            if result.diagnosis == 'error' or result.diagnosis == 'unknown':
                mapped_results[name] = result.diagnosis
                continue
            
            # Skip modules that don't provide disease diagnoses
            if name in skip_for_diagnosis:
                mapped_results[name] = result.diagnosis
                continue
            
            # Get the diagnosis as a string
            diag = result.diagnosis
            original_diag = diag
            
            # Handle numpy.int64 or other numeric types from NeuralNetwork
            if name == 'NeuralNetwork':
                if hasattr(diag, 'item'):
                    diag = str(diag.item())
                elif isinstance(diag, (int, float)):
                    diag = str(int(diag))
                
                if isinstance(diag, str) and diag.isdigit():
                    idx = int(diag)
                    nn = self._modules.get('NeuralNetwork')
                    if nn and hasattr(nn, 'DISEASE_LABELS') and 0 <= idx < len(nn.DISEASE_LABELS):
                        mapped_diag = nn.DISEASE_LABELS[idx]
                        mapped_results[name] = mapped_diag
                        diag = mapped_diag
                    else:
                        mapped_results[name] = original_diag
                        continue
                else:
                    mapped_results[name] = diag
            else:
                mapped_results[name] = diag
            
            if not isinstance(diag, str) or diag == '':
                continue
                
            if name == 'FuzzyController' and 'Severity:' in diag:
                continue
            
            module_diagnoses.append(diag)
            diagnosis_votes[diag] = diagnosis_votes.get(diag, 0) + 1
            if diag not in diagnosis_confidences:
                diagnosis_confidences[diag] = []
            diagnosis_confidences[diag].append(result.confidence)

        if not diagnosis_votes:
            if 'Planner' in diagnosis_results:
                planner_result = diagnosis_results['Planner']
                if planner_result.diagnosis not in ['error', 'unknown']:
                    final_diagnosis = planner_result.diagnosis
                    avg_confidence = planner_result.confidence
                else:
                    final_diagnosis = 'unknown'
                    avg_confidence = 0.0
            else:
                final_diagnosis = 'unknown'
                avg_confidence = 0.0
        else:
            max_votes = max(diagnosis_votes.values())
            candidates = [d for d, v in diagnosis_votes.items() if v == max_votes]
            
            if len(candidates) == 1:
                final_diagnosis = candidates[0]
                avg_confidence = sum(diagnosis_confidences[final_diagnosis]) / len(diagnosis_confidences[final_diagnosis])
            else:
                best_conf = 0
                best_diag = candidates[0]
                for diag in candidates:
                    conf = sum(diagnosis_confidences[diag]) / len(diagnosis_confidences[diag])
                    if conf > best_conf:
                        best_conf = conf
                        best_diag = diag
                final_diagnosis = best_diag
                avg_confidence = best_conf

        urgency = self._assess_urgency(patient, avg_confidence)
        recommendations = self._generate_recommendations(patient, final_diagnosis, urgency)

        treatment_plan = {}
        if 'Planner' in self._modules:
            try:
                plan_result = self._modules['Planner'].analyze(patient)
                if isinstance(plan_result, dict):
                    treatment_plan = plan_result
                else:
                    treatment_plan = {'plan': [], 'steps': 0}
            except Exception as e:
                self._log(f"  [Planner] Error: {e}")
                treatment_plan = {'plan': [], 'steps': 0, 'error': str(e)}

        module_results_display = {}
        for name, result in diagnosis_results.items():
            if name in mapped_results:
                module_results_display[name] = {
                    'diagnosis': mapped_results[name],
                    'confidence': result.confidence
                }
            else:
                module_results_display[name] = {
                    'diagnosis': result.diagnosis,
                    'confidence': result.confidence
                }

        action_report = {
            'patient_id': patient.patient_id,
            'timestamp': patient.timestamp,
            'symptoms': patient.symptoms,
            'diagnosis': final_diagnosis,
            'confidence': round(avg_confidence, 3),
            'urgency': urgency.value,
            'recommendations': recommendations,
            'next_action': self._decide_next_action(urgency),
            'module_results': module_results_display,
            'treatment_plan': treatment_plan
        }

        self.performance_score += (10 if avg_confidence > 0.7 else 5)
        self.state = AgentState.DONE
        self._log(f"Action generated: {urgency.value} urgency, {final_diagnosis}")
        
        return action_report

    def run(self, percept: PatientPercept) -> Dict:
        """
        Full agent cycle: Perceive → Think → Act
        
        Args:
            percept: PatientPercept object
            
        Returns:
            Action report dictionary
        """
        self.perceive(percept)
        results = self.think()
        return self.act(results)

    def _assess_urgency(self, patient: PatientPercept, confidence: float) -> UrgencyLevel:
        if patient.temperature >= 40.0:
            return UrgencyLevel.CRITICAL
        elif patient.temperature >= 39.0:
            return UrgencyLevel.HIGH
        elif patient.temperature >= 38.0:
            return UrgencyLevel.MEDIUM
        
        if patient.heart_rate > 120 or patient.heart_rate < 40:
            return UrgencyLevel.CRITICAL
        elif patient.heart_rate > 100:
            return UrgencyLevel.HIGH
        
        if len(patient.symptoms) >= 7:
            return UrgencyLevel.HIGH
        elif len(patient.symptoms) >= 4:
            return UrgencyLevel.MEDIUM
        
        serious_diseases = ['covid19', 'cardiac_event', 'pneumonia', 'meningitis', 'tuberculosis']
        if confidence > 0.7:
            for disease in serious_diseases:
                if disease in str(patient.symptoms).lower():
                    return UrgencyLevel.HIGH
        
        return UrgencyLevel.LOW

    def _generate_recommendations(self, patient: PatientPercept, 
                                diagnosis: str, urgency: UrgencyLevel) -> List[str]:
        recommendations = []
        
        # Add urgency-based recommendations
        urgency_recs = {
            UrgencyLevel.CRITICAL: [
                "🚨 EMERGENCY: Immediate medical attention required!",
                "📞 Call emergency services or go to nearest ER",
                "🚑 Patient should not be moved without medical supervision"
            ],
            UrgencyLevel.HIGH: [
                "⚠️ URGENT: See a doctor within 24 hours",
                "📊 Monitor vitals closely",
                "📱 Keep emergency contact ready"
            ],
            UrgencyLevel.MEDIUM: [
                "📅 Schedule a doctor's appointment within 3-5 days",
                "🌡️ Rest and monitor symptoms",
                "💧 Stay well hydrated"
            ],
            UrgencyLevel.LOW: [
                "💚 Low urgency: Monitor symptoms at home",
                "💧 Stay hydrated and rest",
                "📱 Seek care if symptoms worsen"
            ]
        }
        recommendations.extend(urgency_recs.get(urgency, []))
        
        # Add diagnosis-specific recommendations
        diagnosis_recs = {
            'covid19': ["🧪 Get PCR test for confirmation", "🛑 Isolate from others", "📊 Monitor oxygen levels"],
            'flu': ["🛌 Get plenty of rest", "💊 Over-the-counter fever reducers", "💧 Stay hydrated"],
            'dengue': ["📊 Monitor for warning signs", "⚠️ Avoid NSAIDs", "💧 Stay well hydrated"],
            'common_cold': ["🛌 Rest and stay hydrated", "💊 Over-the-counter cold medications", "🧂 Saline nasal spray"],
            'cardiac_event': ["🚨 IMMEDIATE: This is a medical emergency!", "📞 Call emergency services immediately"],
            'diabetes': ["📊 Monitor blood sugar levels", "🥗 Follow diabetic diet plan", "💊 Take prescribed medications"],
            'tuberculosis': ["💊 Complete full course of antibiotics", "🛑 Follow isolation protocols", "📊 Monitor for side effects"],
            'meningitis': ["🚨 EMERGENCY: Immediate hospital care required", "💊 Antibiotics must be started immediately"],
            'pneumonia': ["💊 Antibiotics as prescribed", "🛌 Rest and stay hydrated", "📊 Monitor breathing"],
            'bronchitis': ["🛌 Rest and stay hydrated", "💊 Cough medication as needed", "🌡️ Monitor fever"],
            'allergies': ["💊 Antihistamines as needed", "🚫 Avoid known allergens", "🧹 Keep environment clean"],
        }
        
        for disease, recs in diagnosis_recs.items():
            if disease in diagnosis.lower():
                recommendations.extend(recs)
                break
        
        if diagnosis == 'unknown':
            recommendations.append("📋 Please provide more symptoms for accurate diagnosis")
        
        # Add general recommendations
        recommendations.extend([
            "📋 Continue monitoring symptoms for 48 hours",
            "📱 Follow up with healthcare provider if symptoms worsen"
        ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recs = []
        for rec in recommendations:
            # Normalize: remove extra spaces and make lowercase for comparison
            normalized = rec.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                unique_recs.append(rec)
        
        # Limit to 7 recommendations maximum
        return unique_recs[:7]
    def _decide_next_action(self, urgency: UrgencyLevel) -> str:
        actions = {
            UrgencyLevel.CRITICAL: "EMERGENCY_REFERRAL",
            UrgencyLevel.HIGH: "URGENT_APPOINTMENT",
            UrgencyLevel.MEDIUM: "SCHEDULE_FOLLOWUP",
            UrgencyLevel.LOW: "MONITOR_AT_HOME"
        }
        return actions.get(urgency, "MONITOR_AT_HOME")

    def _log(self, message: str):
        entry = f"[{self.state.value}] {message}"
        self.memory.action_log.append(entry)

    def print_log(self):
        print("\n📋 Agent Action Log:")
        print("─" * 50)
        for entry in self.memory.action_log:
            print(f"  {entry}")

    def get_performance(self) -> Dict:
        return {
            'total_patients': len(self.memory.patient_history),
            'performance_score': self.performance_score,
            'diagnoses_made': len(self.memory.diagnosis_history)  # This returns an int
        }
    
    def reset(self):
        self.state = AgentState.IDLE
        self.memory = AgentMemory()
        self._log("Agent reset")


def test_agent():
    """Test the HealthcareDiagnosticAgent"""
    print("\n" + "=" * 60)
    print("  TESTING HEALTHCARE DIAGNOSTIC AGENT")
    print("=" * 60)
    
    agent = HealthcareDiagnosticAgent()
    
    class MockModule:
        def analyze(self, percept):
            return {'diagnosis': 'flu', 'confidence': 0.85, 'details': {}}
    
    class MockPlanner:
        def analyze(self, percept):
            return {'diagnosis': 'flu', 'steps': 3, 'plan': [{'step': 1, 'action': 'Rest'}]}
    
    agent.register_module('MockModule', MockModule())
    agent.register_module('Planner', MockPlanner())
    
    patient = PatientPercept(
        patient_id="P001",
        symptoms=["fever", "cough", "fatigue"],
        age=34,
        temperature=38.5,
        heart_rate=98,
        blood_pressure="120/80"
    )
    
    print("\n  Testing agent workflow...")
    
    # Test the run() method
    report = agent.run(patient)
    
    print(f"\n  ✅ Diagnosis: {report['diagnosis']}")
    print(f"  Confidence: {report['confidence']:.1%}")
    print(f"  Urgency: {report['urgency']}")
    
    agent.print_log()
    print("\n  ✅ Agent test passed!")


if __name__ == "__main__":
    test_agent()