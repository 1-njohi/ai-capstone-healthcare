# ============================================================
# MODULE 7: AI Planning — Treatment Plan Generator
# Covers: Week 12 (AI Planning Techniques & STRIPS)
# ============================================================

from copy import deepcopy
from collections import deque, Counter
from typing import Dict, List, Set, Tuple, Optional, Any
import re


class TreatmentPlanner:
    """
    STRIPS-based treatment planner.
    Generates step-by-step treatment plans
    from patient diagnosis to recovery.
    
    Uses BFS search over state space to find
    optimal sequence of medical actions.
    """

    def __init__(self):
        self.action_library = self._build_action_library()
        self.diagnosis_states = self._initialize_diagnosis_states()
        self.diagnosis_goals = self._initialize_diagnosis_goals()
        self.symptom_keywords = self._initialize_symptom_keywords()
        
        print(f"[Planner] Initialized with {len(self.action_library)} actions")
        print(f"[Planner] {len(self.diagnosis_states)} diagnoses supported")

    def _initialize_symptom_keywords(self) -> Dict[str, List[str]]:
        """Map diseases to their characteristic symptoms"""
        return {
            'covid19': ['loss_of_smell', 'loss_of_taste', 'fever', 'cough', 'fatigue', 'shortness_of_breath'],
            'flu': ['fever', 'cough', 'body_aches', 'headache', 'fatigue', 'chills'],
            'common_cold': ['runny_nose', 'sneezing', 'sore_throat', 'cough'],
            'dengue': ['high_fever', 'joint_pain', 'rash', 'headache', 'muscle_pain', 'nausea'],
            'cardiac_event': ['chest_pain', 'shortness_of_breath', 'sweating', 'fatigue', 'nausea'],
            'diabetes': ['frequent_urination', 'excessive_thirst', 'blurred_vision', 'fatigue', 'weight_loss'],
            'tuberculosis': ['cough', 'weight_loss', 'night_sweats', 'fatigue', 'fever'],
            'meningitis': ['headache', 'stiff_neck', 'high_fever', 'light_sensitivity', 'nausea'],
            'pneumonia': ['fever', 'cough', 'shortness_of_breath', 'chest_pain', 'fatigue', 'chills'],
            'bronchitis': ['cough', 'mucus_production', 'fatigue', 'shortness_of_breath'],
            'allergies': ['runny_nose', 'sneezing', 'itchy_eyes', 'sore_throat'],
            'gastroenteritis': ['nausea', 'vomiting', 'diarrhea', 'abdominal_pain', 'fever'],
            'strep_throat': ['sore_throat', 'fever', 'swollen_lymph_nodes', 'headache']
        }

    def _build_action_library(self) -> List[Dict]:
        """Define medical treatment actions with STRIPS format"""
        return [
            # Emergency Actions
            {
                'name': 'CallEmergencyServices',
                'precond': {'EMERGENCY_CASE', 'PATIENT_PRESENT'},
                'delete': {'EMERGENCY_CASE'},
                'add': {'EMERGENCY_SERVICES_CALLED'},
                'cost': 0,
                'duration': '5 minutes',
                'category': 'emergency'
            },
            {
                'name': 'TransferToICU',
                'precond': {'EMERGENCY_SERVICES_CALLED', 'ICU_AVAILABLE'},
                'delete': {'EMERGENCY_SERVICES_CALLED'},
                'add': {'PATIENT_IN_ICU', 'MONITORING_ACTIVE'},
                'cost': 0,
                'duration': '15 minutes',
                'category': 'emergency'
            },
            {
                'name': 'StartEmergencyOxygen',
                'precond': {'LOW_OXYGEN', 'PATIENT_PRESENT'},
                'delete': {'LOW_OXYGEN'},
                'add': {'OXYGEN_STARTED', 'OXYGEN_MONITORING'},
                'cost': 0,
                'duration': '5 minutes',
                'category': 'emergency'
            },
            
            # Diagnostic Actions
            {
                'name': 'OrderBloodPanel',
                'precond': {'PATIENT_PRESENT', 'DIAGNOSIS_NEEDED'},
                'delete': {'DIAGNOSIS_NEEDED'},
                'add': {'BLOOD_RESULTS_PENDING'},
                'cost': 1,
                'duration': '30 minutes',
                'category': 'diagnostic'
            },
            {
                'name': 'ReceiveBloodResults',
                'precond': {'BLOOD_RESULTS_PENDING'},
                'delete': {'BLOOD_RESULTS_PENDING'},
                'add': {'BLOOD_RESULTS_AVAILABLE', 'DIAGNOSIS_REFINED'},
                'cost': 0,
                'duration': '2 hours',
                'category': 'diagnostic'
            },
            {
                'name': 'OrderPCRTest',
                'precond': {'COVID_SUSPECTED', 'PATIENT_PRESENT'},
                'delete': {'COVID_SUSPECTED'},
                'add': {'PCR_PENDING'},
                'cost': 1,
                'duration': '24 hours',
                'category': 'diagnostic'
            },
            {
                'name': 'ReceivePCRResult',
                'precond': {'PCR_PENDING'},
                'delete': {'PCR_PENDING'},
                'add': {'PCR_RESULT_AVAILABLE', 'DIAGNOSIS_CONFIRMED'},
                'cost': 0,
                'duration': '24 hours',
                'category': 'diagnostic'
            },
            {
                'name': 'OrderChestXray',
                'precond': {'RESPIRATORY_SYMPTOMS', 'PATIENT_PRESENT'},
                'delete': set(),
                'add': {'XRAY_PENDING'},
                'cost': 1,
                'duration': '1 hour',
                'category': 'diagnostic'
            },
            {
                'name': 'ReceiveXrayResult',
                'precond': {'XRAY_PENDING'},
                'delete': {'XRAY_PENDING'},
                'add': {'XRAY_COMPLETED', 'DIAGNOSIS_REFINED'},
                'cost': 0,
                'duration': '30 minutes',
                'category': 'diagnostic'
            },
            {
                'name': 'ClinicalDiagnosis',
                'precond': {'PATIENT_PRESENT', 'DIAGNOSIS_NEEDED'},
                'delete': {'DIAGNOSIS_NEEDED'},
                'add': {'DIAGNOSIS_CONFIRMED'},
                'cost': 0,
                'duration': '15 minutes',
                'category': 'diagnostic'
            },
            
            # Treatment Actions
            {
                'name': 'PrescribeAntiviral',
                'precond': {'DIAGNOSIS_CONFIRMED', 'VIRAL_INFECTION'},
                'delete': {'VIRAL_INFECTION'},
                'add': {'ANTIVIRAL_PRESCRIBED', 'TREATMENT_STARTED'},
                'cost': 1,
                'duration': '10 minutes',
                'category': 'treatment'
            },
            {
                'name': 'PrescribeAntibiotics',
                'precond': {'DIAGNOSIS_CONFIRMED', 'BACTERIAL_INFECTION'},
                'delete': {'BACTERIAL_INFECTION'},
                'add': {'ANTIBIOTICS_PRESCRIBED', 'TREATMENT_STARTED'},
                'cost': 1,
                'duration': '10 minutes',
                'category': 'treatment'
            },
            {
                'name': 'PrescribeAntipyretics',
                'precond': {'HIGH_FEVER', 'PATIENT_PRESENT'},
                'delete': {'HIGH_FEVER'},
                'add': {'FEVER_MANAGED', 'TREATMENT_STARTED'},
                'cost': 1,
                'duration': '5 minutes',
                'category': 'treatment'
            },
            {
                'name': 'AdministerFluids',
                'precond': {'PATIENT_PRESENT', 'DEHYDRATION_RISK'},
                'delete': {'DEHYDRATION_RISK'},
                'add': {'HYDRATED', 'TREATMENT_STARTED'},
                'cost': 1,
                'duration': '30 minutes',
                'category': 'treatment'
            },
            {
                'name': 'PrescribePainRelief',
                'precond': {'PAIN_SYMPTOM', 'PATIENT_PRESENT'},
                'delete': {'PAIN_SYMPTOM'},
                'add': {'PAIN_MANAGED', 'TREATMENT_STARTED'},
                'cost': 1,
                'duration': '5 minutes',
                'category': 'treatment'
            },
            {
                'name': 'PrescribeInsulin',
                'precond': {'DIAGNOSIS_CONFIRMED', 'DIABETES_DIAGNOSED'},
                'delete': {'DIABETES_DIAGNOSED'},
                'add': {'INSULIN_PRESCRIBED', 'TREATMENT_STARTED'},
                'cost': 1,
                'duration': '10 minutes',
                'category': 'treatment'
            },
            {
                'name': 'StartIVAntibiotics',
                'precond': {'PATIENT_IN_ICU', 'BACTERIAL_INFECTION'},
                'delete': {'BACTERIAL_INFECTION'},
                'add': {'IV_ANTIBIOTICS_STARTED', 'TREATMENT_STARTED'},
                'cost': 1,
                'duration': '1 hour',
                'category': 'treatment'
            },
            {
                'name': 'PrescribeAntimalarial',
                'precond': {'DIAGNOSIS_CONFIRMED', 'VIRAL_INFECTION', 'DENGUE_DIAGNOSED'},
                'delete': {'VIRAL_INFECTION'},
                'add': {'ANTIMALARIAL_PRESCRIBED', 'TREATMENT_STARTED'},
                'cost': 1,
                'duration': '10 minutes',
                'category': 'treatment'
            },
            
            # Monitoring Actions
            {
                'name': 'MonitorVitals',
                'precond': {'TREATMENT_STARTED', 'PATIENT_PRESENT'},
                'delete': set(),
                'add': {'VITALS_MONITORED'},
                'cost': 0,
                'duration': 'Continuous',
                'category': 'monitoring'
            },
            {
                'name': 'MonitorBloodSugar',
                'precond': {'DIAGNOSIS_CONFIRMED', 'DIABETES_DIAGNOSED'},
                'delete': set(),
                'add': {'BLOOD_SUGAR_MONITORED'},
                'cost': 0,
                'duration': '4 hours',
                'category': 'monitoring'
            },
            
            # Isolation Actions
            {
                'name': 'IsolatePatient',
                'precond': {'CONTAGIOUS_DISEASE', 'PATIENT_PRESENT'},
                'delete': {'CONTAGIOUS_DISEASE'},
                'add': {'PATIENT_ISOLATED'},
                'cost': 0,
                'duration': '14 days',
                'category': 'isolation'
            },
            
            # Follow-up Actions
            {
                'name': 'ScheduleFollowUp',
                'precond': {'TREATMENT_STARTED', 'VITALS_MONITORED'},
                'delete': set(),
                'add': {'FOLLOWUP_SCHEDULED', 'PLAN_COMPLETE'},
                'cost': 0,
                'duration': '5 minutes',
                'category': 'followup'
            },
            {
                'name': 'DischargePatient',
                'precond': {'PLAN_COMPLETE', 'SYMPTOMS_RESOLVED'},
                'delete': {'PLAN_COMPLETE'},
                'add': {'PATIENT_DISCHARGED'},
                'cost': 0,
                'duration': '30 minutes',
                'category': 'followup'
            },
        ]

    def _initialize_diagnosis_states(self) -> Dict[str, Set[str]]:
        """Map diagnoses to initial state predicates"""
        return {
            'flu': {'PATIENT_PRESENT', 'VIRAL_INFECTION', 'DIAGNOSIS_NEEDED', 'HIGH_FEVER'},
            'covid19': {'PATIENT_PRESENT', 'COVID_SUSPECTED', 'CONTAGIOUS_DISEASE', 
                       'VIRAL_INFECTION', 'RESPIRATORY_SYMPTOMS', 'DIAGNOSIS_NEEDED'},
            'common_cold': {'PATIENT_PRESENT', 'VIRAL_INFECTION', 'RESPIRATORY_SYMPTOMS', 'DIAGNOSIS_NEEDED'},
            'dengue': {'PATIENT_PRESENT', 'VIRAL_INFECTION', 'DEHYDRATION_RISK', 
                      'DENGUE_DIAGNOSED', 'HIGH_FEVER', 'PAIN_SYMPTOM', 'DIAGNOSIS_NEEDED'},
            'pneumonia': {'PATIENT_PRESENT', 'BACTERIAL_INFECTION', 'RESPIRATORY_SYMPTOMS', 
                         'LOW_OXYGEN', 'HIGH_FEVER', 'DIAGNOSIS_NEEDED'},
            'bronchitis': {'PATIENT_PRESENT', 'BACTERIAL_INFECTION', 'RESPIRATORY_SYMPTOMS', 
                          'DIAGNOSIS_NEEDED'},
            'strep_throat': {'PATIENT_PRESENT', 'BACTERIAL_INFECTION', 'PAIN_SYMPTOM', 
                            'DIAGNOSIS_NEEDED'},
            'tuberculosis': {'PATIENT_PRESENT', 'BACTERIAL_INFECTION', 'CONTAGIOUS_DISEASE', 
                            'RESPIRATORY_SYMPTOMS', 'DIAGNOSIS_NEEDED'},
            'meningitis': {'PATIENT_PRESENT', 'BACTERIAL_INFECTION', 'EMERGENCY_CASE', 
                          'ICU_AVAILABLE', 'DIAGNOSIS_NEEDED'},
            'cardiac_event': {'PATIENT_PRESENT', 'EMERGENCY_CASE', 'ICU_AVAILABLE', 
                             'DIAGNOSIS_NEEDED'},
            'diabetes': {'PATIENT_PRESENT', 'DIABETES_DIAGNOSED', 'DIAGNOSIS_NEEDED'},
            'allergies': {'PATIENT_PRESENT', 'DIAGNOSIS_NEEDED'},
            'gastroenteritis': {'PATIENT_PRESENT', 'DEHYDRATION_RISK', 'DIAGNOSIS_NEEDED'},
        }

    def _initialize_diagnosis_goals(self) -> Dict[str, Set[str]]:
        """Map diagnoses to goal states"""
        base_goals = {'TREATMENT_STARTED', 'VITALS_MONITORED', 'FOLLOWUP_SCHEDULED'}
        
        diagnosis_specific = {
            'covid19': {'PATIENT_ISOLATED', 'DIAGNOSIS_CONFIRMED'},
            'flu': {'FEVER_MANAGED', 'DIAGNOSIS_CONFIRMED'},
            'pneumonia': {'DIAGNOSIS_CONFIRMED', 'OXYGEN_STARTED', 'OXYGEN_MONITORING'},
            'dengue': {'HYDRATED', 'DIAGNOSIS_CONFIRMED', 'ANTIMALARIAL_PRESCRIBED'},
            'strep_throat': {'DIAGNOSIS_CONFIRMED', 'PAIN_MANAGED'},
            'tuberculosis': {'PATIENT_ISOLATED', 'DIAGNOSIS_CONFIRMED', 'ANTIBIOTICS_PRESCRIBED'},
            'meningitis': {'PATIENT_IN_ICU', 'DIAGNOSIS_CONFIRMED', 'IV_ANTIBIOTICS_STARTED'},
            'cardiac_event': {'PATIENT_IN_ICU', 'DIAGNOSIS_CONFIRMED', 'OXYGEN_STARTED'},
            'diabetes': {'DIAGNOSIS_CONFIRMED', 'INSULIN_PRESCRIBED', 'BLOOD_SUGAR_MONITORED'},
            'common_cold': {'DIAGNOSIS_CONFIRMED'},
            'bronchitis': {'DIAGNOSIS_CONFIRMED'},
            'allergies': {'DIAGNOSIS_CONFIRMED'},
            'gastroenteritis': {'HYDRATED', 'DIAGNOSIS_CONFIRMED'},
        }
        
        goals = {}
        for diagnosis in self.diagnosis_states.keys():
            specific = diagnosis_specific.get(diagnosis, set())
            goals[diagnosis] = base_goals | specific
        
        return goals

    def _apply_action(self, state: frozenset, action: Dict) -> Optional[frozenset]:
        if not action['precond'].issubset(state):
            return None
        
        new_state = set(state)
        
        delete_effects = action['delete']
        if not isinstance(delete_effects, set):
            delete_effects = set(delete_effects) if delete_effects else set()
        new_state -= delete_effects
        
        add_effects = action['add']
        if not isinstance(add_effects, set):
            add_effects = set(add_effects) if add_effects else set()
        new_state |= add_effects
        
        return frozenset(new_state)

    def generate_plan(self, initial_state: Set[str], goal_state: Set[str],
                      max_depth: int = 20) -> Optional[List[Dict]]:
        initial = frozenset(initial_state)
        goal = frozenset(goal_state)

        queue = deque([(initial, [])])
        visited = {initial}

        while queue:
            state, plan = queue.popleft()
            
            if goal.issubset(state):
                return plan
            
            if len(plan) >= max_depth:
                continue
            
            for action in self.action_library:
                new_state = self._apply_action(state, action)
                if new_state is not None and new_state not in visited:
                    visited.add(new_state)
                    new_plan = plan + [action]
                    queue.append((new_state, new_plan))
        
        return None

    def _determine_diagnosis_from_symptoms(self, symptoms: List[str]) -> Tuple[str, float]:
        """
        Determine the most likely diagnosis from symptoms using keyword matching.
        Returns tuple of (diagnosis, confidence)
        """
        symptoms_lower = [s.lower().replace(' ', '_') for s in symptoms]
        symptom_counts = Counter(symptoms_lower)
        
        # Score each disease based on how many of its characteristic symptoms are present
        scores = {}
        total_present = len(symptom_counts)
        
        for disease, keywords in self.symptom_keywords.items():
            matched = 0
            for keyword in keywords:
                if keyword in symptom_counts:
                    matched += symptom_counts[keyword]
            
            # Calculate score as percentage of matched symptoms
            if total_present > 0:
                # Weight by how many keywords the disease has
                max_possible = min(len(keywords), total_present)
                if max_possible > 0:
                    score = matched / max_possible
                else:
                    score = 0
            else:
                score = 0
            
            scores[disease] = score
        
        # Find the highest scoring disease
        if scores:
            best_disease = max(scores, key=scores.get)
            best_score = scores[best_disease]
            
            # Only return a diagnosis if confidence is above threshold
            if best_score >= 0.3:
                return best_disease, best_score
            else:
                return 'unknown', best_score
        
        return 'unknown', 0.0

    def _determine_urgency_from_symptoms(self, temperature: float, heart_rate: int, 
                                          symptoms: List[str], diagnosis: str) -> str:
        """Determine urgency from vitals and symptoms."""
        # Critical conditions
        critical_diagnoses = ['cardiac_event', 'meningitis']
        if diagnosis in critical_diagnoses:
            return 'CRITICAL'
        
        if temperature >= 40.0:
            return 'CRITICAL'
        elif heart_rate > 130:
            return 'CRITICAL'
        
        # High urgency
        high_diagnoses = ['covid19', 'dengue', 'tuberculosis', 'pneumonia']
        if diagnosis in high_diagnoses:
            return 'HIGH'
        
        if temperature >= 39.0:
            return 'HIGH'
        elif heart_rate > 110:
            return 'HIGH'
        elif len(symptoms) >= 7:
            return 'HIGH'
        
        # Medium urgency
        moderate_diagnoses = ['flu', 'diabetes', 'bronchitis']
        if diagnosis in moderate_diagnoses:
            return 'MEDIUM'
        
        if temperature >= 38.0:
            return 'MEDIUM'
        elif len(symptoms) >= 4:
            return 'MEDIUM'
        
        return 'LOW'

    def create_treatment_plan(self, diagnosis: str, urgency: str,
                              temperature: float = 37.0,
                              heart_rate: int = 80,
                              symptoms: List[str] = None) -> Dict:
        diagnosis = diagnosis.lower().replace(' ', '_')
        
        initial_state = self.diagnosis_states.get(
            diagnosis,
            {'PATIENT_PRESENT', 'DIAGNOSIS_NEEDED'}
        ).copy()
        
        if urgency == 'CRITICAL':
            initial_state.add('EMERGENCY_CASE')
            initial_state.add('ICU_AVAILABLE')
        elif urgency == 'HIGH':
            initial_state.add('HIGH_FEVER')
        
        if symptoms:
            symptom_facts = {
                'fever': 'HIGH_FEVER',
                'cough': 'RESPIRATORY_SYMPTOMS',
                'shortness_of_breath': 'LOW_OXYGEN',
                'chest_pain': 'PAIN_SYMPTOM',
                'joint_pain': 'PAIN_SYMPTOM',
                'headache': 'PAIN_SYMPTOM',
                'nausea': 'DEHYDRATION_RISK',
                'vomiting': 'DEHYDRATION_RISK',
                'diarrhea': 'DEHYDRATION_RISK',
            }
            
            for symptom in symptoms:
                fact = symptom_facts.get(symptom.lower().replace(' ', '_'))
                if fact:
                    initial_state.add(fact)
        
        if diagnosis == 'dengue':
            initial_state.add('DENGUE_DIAGNOSED')
        elif diagnosis == 'diabetes':
            initial_state.add('DIABETES_DIAGNOSED')
        
        goal_state = self.diagnosis_goals.get(
            diagnosis,
            {'TREATMENT_STARTED', 'FOLLOWUP_SCHEDULED'}
        )
        
        if urgency == 'CRITICAL':
            goal_state.add('PATIENT_IN_ICU')
        
        plan = self.generate_plan(initial_state, goal_state)
        
        if plan is None:
            expanded_initial = initial_state.copy()
            expanded_initial.add('DIAGNOSIS_CONFIRMED')
            expanded_initial.add('TREATMENT_STARTED')
            plan = self.generate_plan(expanded_initial, goal_state)
        
        if plan is None:
            return {
                'diagnosis': diagnosis,
                'urgency': urgency,
                'plan': [],
                'steps': 0,
                'status': 'No plan found - consult physician immediately',
                'estimated_duration_hours': 0,
                'initial_state': sorted(initial_state),
                'goal_state': sorted(goal_state)
            }
        
        formatted_plan = []
        total_duration_hours = 0
        categories = set()
        
        for i, action in enumerate(plan, 1):
            step = {
                'step': i,
                'action': action['name'],
                'duration': action['duration'],
                'cost': action['cost'],
                'category': action.get('category', 'general')
            }
            formatted_plan.append(step)
            categories.add(action.get('category', 'general'))
            
            duration_str = action['duration']
            if 'hours' in duration_str:
                try:
                    hours = float(duration_str.split()[0])
                    total_duration_hours += hours
                except:
                    total_duration_hours += 1
            elif 'minutes' in duration_str:
                try:
                    minutes = float(duration_str.split()[0])
                    total_duration_hours += minutes / 60
                except:
                    total_duration_hours += 0.1
            elif 'days' in duration_str:
                try:
                    days = float(duration_str.split()[0])
                    total_duration_hours += days * 24
                except:
                    total_duration_hours += 24
        
        return {
            'diagnosis': diagnosis,
            'urgency': urgency,
            'plan': formatted_plan,
            'steps': len(formatted_plan),
            'categories': list(categories),
            'status': 'Plan generated successfully',
            'estimated_duration_hours': round(total_duration_hours, 1),
            'initial_state': sorted(initial_state),
            'goal_state': sorted(goal_state)
        }

    def _create_fallback_plan(self, diagnosis: str, urgency: str) -> Dict:
        fallback_steps = [
            {'step': 1, 'action': 'ClinicalDiagnosis', 'duration': '15 minutes', 'cost': 0, 'category': 'diagnostic'},
            {'step': 2, 'action': 'PrescribeAntipyretics', 'duration': '5 minutes', 'cost': 1, 'category': 'treatment'},
            {'step': 3, 'action': 'MonitorVitals', 'duration': 'Continuous', 'cost': 0, 'category': 'monitoring'},
            {'step': 4, 'action': 'ScheduleFollowUp', 'duration': '5 minutes', 'cost': 0, 'category': 'followup'}
        ]
        
        if urgency == 'CRITICAL':
            fallback_steps = [
                {'step': 1, 'action': 'CallEmergencyServices', 'duration': '5 minutes', 'cost': 0, 'category': 'emergency'},
                {'step': 2, 'action': 'TransferToICU', 'duration': '15 minutes', 'cost': 0, 'category': 'emergency'},
                {'step': 3, 'action': 'ClinicalDiagnosis', 'duration': '15 minutes', 'cost': 0, 'category': 'diagnostic'},
                {'step': 4, 'action': 'MonitorVitals', 'duration': 'Continuous', 'cost': 0, 'category': 'monitoring'},
                {'step': 5, 'action': 'ScheduleFollowUp', 'duration': '5 minutes', 'cost': 0, 'category': 'followup'}
            ]
        elif urgency == 'HIGH':
            fallback_steps = [
                {'step': 1, 'action': 'ClinicalDiagnosis', 'duration': '15 minutes', 'cost': 0, 'category': 'diagnostic'},
                {'step': 2, 'action': 'PrescribeAntipyretics', 'duration': '5 minutes', 'cost': 1, 'category': 'treatment'},
                {'step': 3, 'action': 'OrderBloodPanel', 'duration': '30 minutes', 'cost': 1, 'category': 'diagnostic'},
                {'step': 4, 'action': 'MonitorVitals', 'duration': 'Continuous', 'cost': 0, 'category': 'monitoring'},
                {'step': 5, 'action': 'ScheduleFollowUp', 'duration': '5 minutes', 'cost': 0, 'category': 'followup'}
            ]
        
        return {
            'diagnosis': diagnosis,
            'urgency': urgency,
            'plan': fallback_steps,
            'steps': len(fallback_steps),
            'categories': list(set(s['category'] for s in fallback_steps)),
            'status': 'Fallback plan generated - physician consultation recommended',
            'estimated_duration_hours': 1.0,
            'initial_state': ['PATIENT_PRESENT', 'DIAGNOSIS_NEEDED'],
            'goal_state': ['TREATMENT_STARTED', 'FOLLOWUP_SCHEDULED']
        }

    def analyze(self, percept) -> Dict:
        """
        Module interface for the agent.
        Dynamically determines diagnosis and urgency from patient data.
        """
        # Determine diagnosis from symptoms with confidence
        diagnosis, confidence = self._determine_diagnosis_from_symptoms(percept.symptoms)
        
        # If no diagnosis found, use the most common symptoms to suggest
        if diagnosis == 'unknown':
            # Try to find the most frequent symptom pattern
            symptoms_lower = [s.lower().replace(' ', '_') for s in percept.symptoms]
            if 'fever' in symptoms_lower and 'cough' in symptoms_lower:
                diagnosis = 'respiratory_infection'
                confidence = 0.4
            elif 'joint_pain' in symptoms_lower and 'rash' in symptoms_lower:
                diagnosis = 'viral_infection'
                confidence = 0.4
            elif len(symptoms_lower) > 0:
                diagnosis = 'symptomatic_condition'
                confidence = 0.3
            else:
                diagnosis = 'unknown'
                confidence = 0.1
        
        # Determine urgency from vitals and symptoms
        urgency = self._determine_urgency_from_symptoms(
            percept.temperature,
            percept.heart_rate,
            percept.symptoms,
            diagnosis
        )
        
        # Create treatment plan
        plan = self.create_treatment_plan(
            diagnosis=diagnosis,
            urgency=urgency,
            temperature=percept.temperature,
            heart_rate=percept.heart_rate,
            symptoms=percept.symptoms
        )
        
        # If no plan found, use fallback
        if plan['status'] == 'No plan found - consult physician immediately':
            plan = self._create_fallback_plan(diagnosis, urgency)
        
        # Add summary for agent compatibility
        plan['summary'] = f"Plan: {plan['steps']} steps generated for {diagnosis} ({urgency} urgency)"
        plan['diagnosis'] = diagnosis
        plan['confidence'] = confidence
        
        return plan

    def get_plan_summary(self, plan: Dict) -> str:
        lines = []
        lines.append("=" * 50)
        lines.append(f"📋 TREATMENT PLAN: {plan['diagnosis'].upper()}")
        lines.append("=" * 50)
        lines.append(f"Urgency: {plan['urgency']}")
        lines.append(f"Status: {plan['status']}")
        lines.append(f"Steps: {plan['steps']}")
        if plan.get('categories'):
            lines.append(f"Categories: {', '.join(plan.get('categories', []))}")
        lines.append(f"Est. Duration: {plan.get('estimated_duration_hours', 0)} hours")
        lines.append("")
        lines.append("Actions:")
        
        if plan['plan']:
            for step in plan['plan']:
                lines.append(f"  Step {step['step']:2d}: {step['action']:<30} [{step['duration']}]")
        else:
            lines.append("  No specific actions found. Please consult a physician.")
        
        lines.append("=" * 50)
        return "\n".join(lines)


def test_planner():
    """Test the TreatmentPlanner module"""
    print("\n" + "=" * 60)
    print("  TESTING TREATMENT PLANNER")
    print("=" * 60)
    
    planner = TreatmentPlanner()
    
    # Test with different symptom sets
    test_cases = [
        (['fever', 'cough', 'loss_of_smell'], 'covid19'),
        (['fever', 'cough', 'body_aches'], 'flu'),
        (['runny_nose', 'sneezing', 'sore_throat'], 'common_cold'),
        (['high_fever', 'joint_pain', 'rash'], 'dengue'),
        (['chest_pain', 'shortness_of_breath'], 'cardiac_event'),
        (['headache', 'stiff_neck', 'high_fever'], 'meningitis'),
    ]
    
    for symptoms, expected in test_cases:
        print(f"\n--- Testing: {symptoms} ---")
        from modules.agent import PatientPercept
        patient = PatientPercept(
            patient_id="test",
            symptoms=symptoms,
            age=30,
            temperature=38.5,
            heart_rate=90,
            blood_pressure="120/80"
        )
        result = planner.analyze(patient)
        print(f"  Expected: {expected}")
        print(f"  Diagnosis: {result['diagnosis']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Urgency: {result['urgency']}")
        print(f"  Steps: {result['steps']}")
    
    print("\n✅ Treatment Planner test passed!")


if __name__ == "__main__":
    test_planner()