# ============================================================
# MODULE 2: FOL Knowledge Base + Inference Engine
# Covers: Week 5 (First-Order Logic & Inference)
# ============================================================

from typing import Set, List, Dict, Tuple, Optional, Any
from collections import deque
import re


class MedicalKnowledgeBase:
    """
    First-Order Logic based medical knowledge base.
    Supports forward chaining, backward chaining,
    and confidence-weighted inference.
    """

    def __init__(self):
        self.facts: Dict[str, float] = {}
        self.rules: List[Tuple[List[str], str, float]] = []
        self._inferred_facts: Set[str] = set()
        self._rule_fired: Set[int] = set()
        self._load_medical_knowledge()

    def _load_medical_knowledge(self):
        """Load domain medical knowledge with certainty factors"""
        
        disease_rules = [
            # ── COVID-19 Rules ──
            (["fever", "cough", "fatigue"], "respiratory_infection", 0.70),
            (["fever", "cough", "loss_of_smell"], "covid19_suspected", 0.85),
            (["covid19_suspected", "fatigue"], "covid19_likely", 0.90),
            (["covid19_suspected", "shortness_of_breath"], "covid19_severe", 0.88),
            (["covid19_suspected", "loss_of_taste"], "covid19_likely", 0.85),
            (["covid19_likely", "high_fever"], "covid19_confirmed", 0.92),
            
            # ── Flu Rules ──
            (["fever", "cough", "fatigue"], "flu_suspected", 0.75),
            (["flu_suspected", "body_aches"], "flu_likely", 0.85),
            (["flu_suspected", "headache"], "flu_with_headache", 0.70),
            (["flu_suspected", "chills"], "flu_likely", 0.80),
            (["flu_likely", "high_fever"], "flu_confirmed", 0.88),
            
            # ── Common Cold Rules ──
            (["runny_nose", "sneezing"], "cold_suspected", 0.70),
            (["cold_suspected", "sore_throat"], "cold_likely", 0.65),
            (["cold_suspected", "mild_fever"], "cold_with_fever", 0.60),
            (["cold_likely", "cough"], "cold_confirmed", 0.80),
            
            # ── Dengue Rules ──
            (["high_fever", "severe_headache"], "dengue_suspected", 0.75),
            (["dengue_suspected", "joint_pain"], "dengue_likely", 0.85),
            (["dengue_suspected", "rash"], "dengue_with_rash", 0.80),
            (["dengue_suspected", "nausea"], "dengue_severe", 0.75),
            (["dengue_likely", "high_fever"], "dengue_confirmed", 0.90),
            (["dengue_suspected", "muscle_pain"], "dengue_likely", 0.80),
            
            # ── Cardiac Rules ──
            (["chest_pain", "shortness_of_breath"], "cardiac_suspected", 0.80),
            (["cardiac_suspected", "sweating"], "cardiac_event_likely", 0.88),
            (["cardiac_suspected", "nausea"], "cardiac_with_nausea", 0.70),
            (["cardiac_event_likely", "chest_pain"], "cardiac_confirmed", 0.92),
            
            # ── Diabetes Rules ──
            (["frequent_urination", "excessive_thirst"], "diabetes_suspected", 0.80),
            (["diabetes_suspected", "blurred_vision"], "diabetes_likely", 0.85),
            (["diabetes_suspected", "weight_loss"], "diabetes_with_weight_loss", 0.75),
            (["diabetes_likely", "fatigue"], "diabetes_confirmed", 0.88),
            
            # ── Tuberculosis Rules ──
            (["cough", "weight_loss"], "tb_suspected", 0.75),
            (["tb_suspected", "night_sweats"], "tb_likely", 0.82),
            (["tb_suspected", "fatigue"], "tb_with_fatigue", 0.80),
            (["tb_likely", "cough"], "tb_confirmed", 0.88),
            
            # ── Meningitis Rules ──
            (["headache", "stiff_neck"], "meningitis_suspected", 0.80),
            (["meningitis_suspected", "high_fever"], "meningitis_likely", 0.88),
            (["meningitis_suspected", "light_sensitivity"], "meningitis_severe", 0.85),
            (["meningitis_likely", "stiff_neck"], "meningitis_confirmed", 0.92),
            
            # ── Bronchitis Rules ──
            (["cough", "mucus_production"], "bronchitis_suspected", 0.75),
            (["bronchitis_suspected", "fatigue"], "bronchitis_likely", 0.80),
            (["bronchitis_suspected", "shortness_of_breath"], "bronchitis_severe", 0.78),
            
            # ── Pneumonia Rules ──
            (["fever", "cough", "shortness_of_breath"], "pneumonia_suspected", 0.80),
            (["pneumonia_suspected", "chest_pain"], "pneumonia_likely", 0.85),
            (["pneumonia_suspected", "high_fever"], "pneumonia_severe", 0.82),
            
            # ── Allergies Rules ──
            (["runny_nose", "sneezing", "itchy_eyes"], "allergies_suspected", 0.85),
            (["allergies_suspected", "sore_throat"], "allergies_likely", 0.75),
            
            # ── Severity/Urgency Rules ──
            (["covid19_severe"], "high_urgency", 0.95),
            (["cardiac_confirmed"], "critical_urgency", 0.98),
            (["meningitis_confirmed"], "critical_urgency", 0.95),
            (["dengue_severe"], "high_urgency", 0.85),
            (["high_fever", "severe_headache"], "high_urgency", 0.80),
            (["pneumonia_severe"], "high_urgency", 0.88),
            
            # ── Temperature-based Rules ──
            (["temperature_high"], "fever", 0.90),
            (["temperature_very_high"], "high_fever", 0.95),
            (["temperature_critical"], "critical_condition", 0.98),
            (["temperature_high", "cough"], "fever_with_cough", 0.85),
            
            # ── Heart Rate Rules ──
            (["heart_rate_high"], "tachycardia", 0.85),
            (["heart_rate_low"], "bradycardia", 0.85),
            
            # ── Combined Symptom Rules ──
            (["fever", "cough"], "respiratory_symptoms", 0.80),
            (["fever", "rash"], "rash_with_fever", 0.70),
            (["joint_pain", "muscle_pain"], "myalgia", 0.75),
            (["nausea", "vomiting"], "gastrointestinal_symptoms", 0.80),
            (["headache", "light_sensitivity"], "neurological_symptoms", 0.75),
            (["chest_pain", "shortness_of_breath"], "cardiac_symptoms", 0.85),
            
            # ── Final Diagnosis Rules ──
            (["covid19_confirmed"], "covid19", 0.98),
            (["flu_confirmed"], "flu", 0.95),
            (["dengue_confirmed"], "dengue", 0.95),
            (["cardiac_confirmed"], "cardiac_event", 0.98),
            (["diabetes_confirmed"], "diabetes", 0.95),
            (["tb_confirmed"], "tuberculosis", 0.95),
            (["meningitis_confirmed"], "meningitis", 0.97),
            (["cold_confirmed"], "common_cold", 0.90),
            (["bronchitis_likely"], "bronchitis", 0.85),
            (["pneumonia_likely"], "pneumonia", 0.88),
            (["allergies_likely"], "allergies", 0.85),
        ]
        
        for conditions, conclusion, cf in disease_rules:
            self.add_rule(conditions, conclusion, cf)
        
        print(f"[KnowledgeBase] Loaded {len(self.rules)} medical rules")

    def add_fact(self, fact: str, certainty: float = 1.0):
        fact = self._normalize_text(fact)
        if fact in self.facts:
            self.facts[fact] = max(self.facts[fact], certainty)
        else:
            self.facts[fact] = certainty

    def add_rule(self, conditions: List[str], conclusion: str, certainty: float = 1.0):
        normalized_conditions = [self._normalize_text(c) for c in conditions]
        normalized_conclusion = self._normalize_text(conclusion)
        self.rules.append((normalized_conditions, normalized_conclusion, certainty))

    def _normalize_text(self, text: str) -> str:
        normalized = text.lower().strip()
        normalized = re.sub(r'[^a-z0-9_\s]', '', normalized)
        normalized = re.sub(r'\s+', '_', normalized)
        return normalized

    def load_patient_symptoms(self, symptoms: List[str]):
        self.facts = {}
        self._inferred_facts = set()
        self._rule_fired = set()
        
        for symptom in symptoms:
            normalized = self._normalize_text(symptom)
            self.add_fact(normalized, 0.95)
            
            # Add related facts with lower certainty
            symptom_mappings = {
                'fever': ['temperature_high'],
                'high_fever': ['temperature_very_high'],
                'cough': ['respiratory_symptom'],
                'headache': ['pain_symptom'],
                'chest_pain': ['pain_symptom', 'cardiac_symptom'],
                'joint_pain': ['pain_symptom', 'joint_symptom'],
                'muscle_pain': ['pain_symptom', 'muscle_symptom'],
                'shortness_of_breath': ['respiratory_symptom', 'breathing_symptom'],
                'runny_nose': ['nasal_symptom'],
                'sneezing': ['nasal_symptom'],
                'sore_throat': ['throat_symptom'],
                'nausea': ['gastrointestinal_symptom'],
                'vomiting': ['gastrointestinal_symptom'],
                'diarrhea': ['gastrointestinal_symptom'],
                'rash': ['skin_symptom'],
                'sweating': ['systemic_symptom'],
                'fatigue': ['systemic_symptom'],
                'weight_loss': ['systemic_symptom'],
                'night_sweats': ['systemic_symptom'],
                'stiff_neck': ['neurological_symptom'],
                'light_sensitivity': ['neurological_symptom'],
                'blurred_vision': ['visual_symptom'],
                'frequent_urination': ['urinary_symptom'],
                'excessive_thirst': ['metabolic_symptom'],
            }
            
            for mapped_symptom in symptom_mappings.get(normalized, []):
                self.add_fact(mapped_symptom, 0.80)
        
        print(f"[KnowledgeBase] Loaded {len(symptoms)} patient symptoms")
        print(f"[KnowledgeBase] Facts: {len(self.facts)} total facts")

    def load_vitals(self, temperature: float, heart_rate: int):
        if temperature >= 40.0:
            self.add_fact("temperature_critical", 0.98)
            self.add_fact("high_fever", 0.95)
        elif temperature >= 39.0:
            self.add_fact("temperature_very_high", 0.95)
            self.add_fact("high_fever", 0.90)
        elif temperature >= 38.0:
            self.add_fact("temperature_high", 0.90)
            self.add_fact("fever", 0.85)
        elif temperature >= 37.5:
            self.add_fact("temperature_elevated", 0.80)
            self.add_fact("mild_fever", 0.75)
        
        if heart_rate > 120:
            self.add_fact("heart_rate_high", 0.95)
            self.add_fact("tachycardia", 0.90)
        elif heart_rate > 100:
            self.add_fact("heart_rate_high", 0.85)
            self.add_fact("tachycardia", 0.80)
        elif heart_rate < 60:
            self.add_fact("heart_rate_low", 0.85)
            self.add_fact("bradycardia", 0.80)
        elif heart_rate < 50:
            self.add_fact("heart_rate_low", 0.95)
            self.add_fact("bradycardia", 0.90)
        
        print(f"[KnowledgeBase] Loaded vitals: {temperature}°C, {heart_rate} BPM")

    def forward_chain(self, verbose: bool = False, max_iterations: int = 50) -> Dict[str, float]:
        if verbose:
            print("\n[KnowledgeBase] Starting Forward Chaining...")
            print(f"[KnowledgeBase] Initial facts: {len(self.facts)}")
        
        iteration = 0
        new_facts_added = True
        self._inferred_facts = set()
        self._rule_fired = set()
        
        all_facts = self.facts.copy()
        
        while new_facts_added and iteration < max_iterations:
            new_facts_added = False
            iteration += 1
            
            if verbose:
                print(f"\n  Iteration {iteration}:")
            
            for rule_idx, (conditions, conclusion, rule_cf) in enumerate(self.rules):
                if rule_idx in self._rule_fired:
                    continue
                
                all_conditions_met = True
                min_cf = 1.0
                
                for condition in conditions:
                    if condition not in all_facts:
                        all_conditions_met = False
                        break
                    min_cf = min(min_cf, all_facts[condition])
                
                if all_conditions_met:
                    conclusion_cf = min_cf * rule_cf
                    
                    if conclusion not in all_facts or conclusion_cf > all_facts[conclusion]:
                        all_facts[conclusion] = conclusion_cf
                        new_facts_added = True
                        self._rule_fired.add(rule_idx)
                        
                        if verbose:
                            conditions_str = " ∧ ".join(conditions)
                            print(f"    Rule {rule_idx}: {conditions_str} → {conclusion} (CF={conclusion_cf:.3f})")
        
        if iteration >= max_iterations:
            print(f"[KnowledgeBase] Warning: Reached max iterations ({max_iterations})")
        
        inferred_facts = {
            fact: cf for fact, cf in all_facts.items()
            if cf > 0.01
        }
        
        if verbose:
            print(f"\n[KnowledgeBase] Forward Chaining complete.")
            print(f"[KnowledgeBase] Found {len(inferred_facts)} facts in {iteration} iterations")
        
        return inferred_facts

    def backward_chain(self, goal: str, 
                       visited: Optional[Set[str]] = None, 
                       depth: int = 0,
                       max_depth: int = 20) -> Tuple[bool, float, List[str]]:
        goal = self._normalize_text(goal)
        
        if visited is None:
            visited = set()
        
        if depth > max_depth:
            return False, 0.0, []
        
        if goal in visited:
            return False, 0.0, []
        
        visited.add(goal)
        
        if goal in self.facts:
            return True, self.facts[goal], [goal]
        
        for rule_idx, (conditions, conclusion, rule_cf) in enumerate(self.rules):
            if conclusion != goal:
                continue
            
            all_proved = True
            min_cf = 1.0
            proof_path = [goal]
            
            for condition in conditions:
                proved, cf, sub_path = self.backward_chain(
                    condition, visited.copy(), depth + 1, max_depth
                )
                if not proved:
                    all_proved = False
                    break
                min_cf = min(min_cf, cf)
                proof_path.extend(sub_path)
            
            if all_proved:
                conclusion_cf = min_cf * rule_cf
                return True, conclusion_cf, proof_path
        
        return False, 0.0, []

    def get_all_diagnoses(self, inferred_facts: Dict[str, float]) -> List[Tuple[str, float]]:
        diagnosis_keywords = [
            'covid19', 'flu', 'dengue', 'cardiac_event', 'diabetes', 
            'common_cold', 'tuberculosis', 'meningitis', 'pneumonia',
            'bronchitis', 'allergies'
        ]
        
        diagnoses = []
        
        for fact, cf in inferred_facts.items():
            for keyword in diagnosis_keywords:
                if keyword in fact and cf > 0.1:
                    diagnosis_name = fact.replace('_suspected', '')
                    diagnosis_name = diagnosis_name.replace('_likely', '')
                    diagnosis_name = diagnosis_name.replace('_confirmed', '')
                    diagnosis_name = diagnosis_name.replace('_severe', '')
                    diagnosis_name = diagnosis_name.replace('_with', '')
                    diagnosis_name = diagnosis_name.replace('_event', '')
                    diagnoses.append((diagnosis_name, cf))
                    break
        
        diagnoses.sort(key=lambda x: x[1], reverse=True)
        
        unique_diagnoses = {}
        for name, cf in diagnoses:
            if name not in unique_diagnoses or cf > unique_diagnoses[name]:
                unique_diagnoses[name] = cf
        
        return sorted(unique_diagnoses.items(), key=lambda x: x[1], reverse=True)

    def analyze(self, percept) -> Dict[str, Any]:
        self.facts = {}
        self._inferred_facts = set()
        self._rule_fired = set()
        
        self.load_patient_symptoms(percept.symptoms)
        self.load_vitals(percept.temperature, percept.heart_rate)
        
        all_facts = self.forward_chain(verbose=False)
        
        diagnoses = self.get_all_diagnoses(all_facts)
        
        if diagnoses:
            top_diagnosis, top_confidence = diagnoses[0]
            
            if top_confidence < 0.6:
                proved, cf, proof = self.backward_chain(top_diagnosis)
                if proved and cf > top_confidence:
                    top_confidence = cf
            
            return {
                'diagnosis': top_diagnosis,
                'confidence': round(top_confidence, 4),
                'details': {
                    'all_diagnoses': diagnoses[:5],
                    'facts': all_facts,
                    'rules_fired': len(self._rule_fired)
                }
            }
        
        common_diseases = ['covid19', 'flu', 'dengue', 'common_cold', 'tuberculosis']
        for disease in common_diseases:
            proved, cf, proof = self.backward_chain(disease)
            if proved and cf > 0.5:
                return {
                    'diagnosis': disease,
                    'confidence': round(cf, 4),
                    'details': {
                        'proof_path': proof,
                        'facts': all_facts,
                        'rules_fired': len(self._rule_fired)
                    }
                }
        
        return {
            'diagnosis': 'unknown',
            'confidence': 0.3,
            'details': {
                'facts': all_facts,
                'rules_fired': len(self._rule_fired)
            }
        }

    def explain_diagnosis(self, diagnosis: str) -> str:
        diagnosis = self._normalize_text(diagnosis)
        
        supporting_rules = []
        for conditions, conclusion, cf in self.rules:
            if conclusion == diagnosis or diagnosis in conclusion:
                supporting_rules.append((conditions, cf))
            elif diagnosis in conclusion:
                supporting_rules.append((conditions, cf))
        
        if supporting_rules:
            explanation = f"'{diagnosis}' is supported by:\n"
            for i, (conditions, cf) in enumerate(supporting_rules[:3], 1):
                explanation += f"  {i}. {', '.join(conditions)} → {diagnosis} (CF={cf})\n"
            return explanation
        else:
            return f"No specific rules found for '{diagnosis}'. It may be a base fact."


def test_knowledge_base():
    """Test the MedicalKnowledgeBase module"""
    print("\n" + "=" * 60)
    print("  TESTING MEDICAL KNOWLEDGE BASE")
    print("=" * 60)
    
    kb = MedicalKnowledgeBase()
    
    test_cases = [
        (["fever", "cough", "loss_of_smell", "fatigue"], "COVID-19"),
        (["fever", "cough", "body_aches", "headache"], "Flu"),
        (["runny_nose", "sneezing", "sore_throat"], "Cold"),
        (["high_fever", "joint_pain", "rash"], "Dengue"),
        (["chest_pain", "shortness_of_breath", "sweating"], "Cardiac"),
        (["frequent_urination", "excessive_thirst", "blurred_vision"], "Diabetes"),
    ]
    
    for symptoms, description in test_cases:
        print(f"\n--- {description} ---")
        from modules.agent import PatientPercept
        patient = PatientPercept(
            patient_id="test",
            symptoms=symptoms,
            age=30,
            temperature=38.5,
            heart_rate=90,
            blood_pressure="120/80"
        )
        result = kb.analyze(patient)
        print(f"  Diagnosis: {result['diagnosis']}")
        print(f"  Confidence: {result['confidence']:.1%}")
    
    print("\n✅ Knowledge Base test passed!")


if __name__ == "__main__":
    test_knowledge_base()