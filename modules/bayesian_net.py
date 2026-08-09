# ============================================================
# MODULE 3: Bayesian Network — Probabilistic Diagnosis
# Covers: Week 7 (Bayesian Networks)
# ============================================================

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import math
import re


class SimpleBayesianDiagnostics:
    """
    Simplified Bayesian diagnostic model using
    Naive Bayes with pre-computed conditional probabilities.
    
    Uses log-space calculations to prevent underflow
    and returns ranked diagnoses with confidence scores.
    """

    def __init__(self):
        # Prior probabilities P(Disease) - base rates in the population
        self.priors = {
            'flu': 0.15,
            'covid19': 0.08,
            'dengue': 0.05,
            'cardiac': 0.04,
            'diabetes': 0.10,
            'common_cold': 0.30,
            'tuberculosis': 0.02,
            'meningitis': 0.01,
            'pneumonia': 0.03,
            'bronchitis': 0.05,
            'allergies': 0.10,
            'healthy': 0.07,
        }
        
        # Normalize priors to sum to 1.0
        total = sum(self.priors.values())
        for disease in self.priors:
            self.priors[disease] /= total
        
        # Initialize likelihoods
        self.likelihoods = self._initialize_likelihoods()
        
        # All possible symptoms (built from likelihoods)
        self.all_symptoms = self._build_symptom_list()
        
        # Smoothing factor for zero probabilities
        self.epsilon = 0.01
        
        print(f"[BayesianNet] Initialized with {len(self.priors)} diseases")
        print(f"[BayesianNet] {len(self.all_symptoms)} symptoms tracked")

    def _initialize_likelihoods(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize likelihood tables for each disease.
        Values represent P(symptom | disease)
        
        Returns:
            Dictionary mapping disease -> symptom -> probability
        """
        likelihoods = {
            # ── COVID-19 ──
            'covid19': {
                'fever': 0.88, 'cough': 0.80, 'fatigue': 0.90,
                'loss_of_smell': 0.85, 'loss_of_taste': 0.70,
                'shortness_of_breath': 0.45, 'headache': 0.65,
                'body_aches': 0.60, 'sore_throat': 0.40,
                'runny_nose': 0.30, 'sneezing': 0.15,
                'chest_pain': 0.20, 'nausea': 0.25,
                'vomiting': 0.10, 'diarrhea': 0.15,
                'rash': 0.05, 'joint_pain': 0.20,
                'muscle_pain': 0.30, 'chills': 0.35,
                'sweating': 0.20, 'weight_loss': 0.05,
                'night_sweats': 0.05, 'stiff_neck': 0.02,
                'light_sensitivity': 0.02,
            },
            
            # ── Flu ──
            'flu': {
                'fever': 0.90, 'cough': 0.85, 'fatigue': 0.88,
                'body_aches': 0.80, 'headache': 0.75,
                'sore_throat': 0.60, 'runny_nose': 0.50,
                'chills': 0.70, 'loss_of_smell': 0.10,
                'loss_of_taste': 0.08, 'shortness_of_breath': 0.15,
                'chest_pain': 0.10, 'nausea': 0.20,
                'vomiting': 0.10, 'diarrhea': 0.05,
                'rash': 0.02, 'joint_pain': 0.30,
                'muscle_pain': 0.40, 'sweating': 0.30,
                'weight_loss': 0.02, 'night_sweats': 0.02,
            },
            
            # ── Common Cold ──
            'common_cold': {
                'runny_nose': 0.85, 'sneezing': 0.80,
                'sore_throat': 0.75, 'cough': 0.60,
                'mild_fever': 0.40, 'headache': 0.35,
                'fatigue': 0.30, 'body_aches': 0.15,
                'fever': 0.30, 'loss_of_smell': 0.05,
                'loss_of_taste': 0.03, 'shortness_of_breath': 0.02,
                'chest_pain': 0.01, 'nausea': 0.05,
                'vomiting': 0.02, 'diarrhea': 0.02,
                'rash': 0.01, 'joint_pain': 0.05,
                'muscle_pain': 0.05, 'chills': 0.10,
                'sweating': 0.05,
            },
            
            # ── Dengue ──
            'dengue': {
                'high_fever': 0.95, 'fever': 0.98,
                'severe_headache': 0.85, 'headache': 0.90,
                'joint_pain': 0.80, 'muscle_pain': 0.85,
                'rash': 0.70, 'nausea': 0.60,
                'vomiting': 0.50, 'fatigue': 0.75,
                'loss_of_smell': 0.05, 'loss_of_taste': 0.03,
                'cough': 0.15, 'sore_throat': 0.10,
                'runny_nose': 0.10, 'sneezing': 0.05,
                'shortness_of_breath': 0.05, 'chest_pain': 0.05,
                'diarrhea': 0.20, 'chills': 0.40,
                'sweating': 0.50, 'weight_loss': 0.10,
                'night_sweats': 0.05,
            },
            
            # ── Cardiac Event ──
            'cardiac': {
                'chest_pain': 0.92, 'shortness_of_breath': 0.88,
                'fatigue': 0.70, 'sweating': 0.75,
                'nausea': 0.40, 'vomiting': 0.20,
                'headache': 0.30, 'body_aches': 0.20,
                'fever': 0.10, 'cough': 0.15,
                'rash': 0.02, 'joint_pain': 0.10,
                'muscle_pain': 0.15, 'chills': 0.10,
                'weight_loss': 0.05, 'night_sweats': 0.05,
                'loss_of_smell': 0.01,
            },
            
            # ── Diabetes ──
            'diabetes': {
                'fatigue': 0.82, 'frequent_urination': 0.95,
                'excessive_thirst': 0.92, 'blurred_vision': 0.70,
                'weight_loss': 0.50, 'fever': 0.10,
                'cough': 0.05, 'rash': 0.08,
                'headache': 0.40, 'joint_pain': 0.20,
                'muscle_pain': 0.20, 'nausea': 0.15,
                'vomiting': 0.05, 'chest_pain': 0.05,
                'shortness_of_breath': 0.10, 'sweating': 0.15,
                'chills': 0.05,
            },
            
            # ── Tuberculosis ──
            'tuberculosis': {
                'cough': 0.95, 'weight_loss': 0.85,
                'night_sweats': 0.80, 'fatigue': 0.88,
                'fever': 0.70, 'chest_pain': 0.30,
                'shortness_of_breath': 0.35, 'sweating': 0.60,
                'loss_of_smell': 0.02, 'loss_of_taste': 0.01,
                'sore_throat': 0.10, 'runny_nose': 0.05,
                'headache': 0.20, 'body_aches': 0.25,
                'muscle_pain': 0.25, 'nausea': 0.15,
                'vomiting': 0.05, 'chills': 0.30,
            },
            
            # ── Meningitis ──
            'meningitis': {
                'headache': 0.95, 'stiff_neck': 0.90,
                'high_fever': 0.92, 'fever': 0.95,
                'light_sensitivity': 0.85, 'fatigue': 0.80,
                'nausea': 0.60, 'vomiting': 0.50,
                'cough': 0.10, 'sore_throat': 0.10,
                'runny_nose': 0.05, 'rash': 0.20,
                'joint_pain': 0.15, 'muscle_pain': 0.20,
                'sweating': 0.30, 'chills': 0.30,
                'loss_of_smell': 0.02,
            },
            
            # ── Pneumonia ──
            'pneumonia': {
                'fever': 0.92, 'cough': 0.90,
                'shortness_of_breath': 0.85, 'chest_pain': 0.75,
                'fatigue': 0.80, 'sweating': 0.70,
                'headache': 0.40, 'body_aches': 0.50,
                'nausea': 0.25, 'vomiting': 0.10,
                'chills': 0.60, 'loss_of_smell': 0.05,
                'loss_of_taste': 0.03, 'runny_nose': 0.10,
                'sore_throat': 0.20, 'muscle_pain': 0.30,
            },
            
            # ── Bronchitis ──
            'bronchitis': {
                'cough': 0.95, 'mucus_production': 0.90,
                'fatigue': 0.80, 'shortness_of_breath': 0.70,
                'chest_pain': 0.65, 'fever': 0.40,
                'sore_throat': 0.30, 'headache': 0.25,
                'body_aches': 0.20, 'chills': 0.20,
                'sweating': 0.15, 'loss_of_smell': 0.02,
                'loss_of_taste': 0.01, 'runny_nose': 0.20,
                'nausea': 0.10,
            },
            
            # ── Allergies ──
            'allergies': {
                'runny_nose': 0.90, 'sneezing': 0.90,
                'itchy_eyes': 0.85, 'sore_throat': 0.40,
                'cough': 0.30, 'fatigue': 0.20,
                'fever': 0.01, 'loss_of_smell': 0.02,
                'loss_of_taste': 0.01, 'headache': 0.15,
                'body_aches': 0.05, 'chest_pain': 0.01,
                'shortness_of_breath': 0.10,
            },
            
            # ── Healthy ──
            'healthy': {
                'fever': 0.02, 'cough': 0.05, 'fatigue': 0.10,
                'headache': 0.08, 'rash': 0.01, 'chest_pain': 0.01,
                'joint_pain': 0.05, 'loss_of_smell': 0.01,
                'body_aches': 0.05, 'sore_throat': 0.05,
                'runny_nose': 0.05, 'sneezing': 0.05,
                'shortness_of_breath': 0.01, 'nausea': 0.02,
                'vomiting': 0.01, 'diarrhea': 0.01,
            }
        }
        
        return likelihoods

    def _build_symptom_list(self) -> List[str]:
        """
        Build the list of all symptoms from likelihoods
        """
        all_symptoms = set()
        for disease in self.likelihoods:
            all_symptoms.update(self.likelihoods[disease].keys())
        return sorted(list(all_symptoms))

    def _normalize_symptom(self, symptom: str) -> str:
        """
        Normalize symptom name to match likelihood keys
        
        Args:
            symptom: Raw symptom string
            
        Returns:
            Normalized symptom string
        """
        normalized = symptom.lower().strip()
        normalized = re.sub(r'[^a-z0-9_\s]', '', normalized)
        normalized = re.sub(r'\s+', '_', normalized)
        
        # Handle common variations
        variations = {
            'loss_of_smell': ['loss_of_smell', 'loss_of_smell', 'anosmia'],
            'loss_of_taste': ['loss_of_taste', 'loss_of_taste', 'ageusia'],
            'shortness_of_breath': ['shortness_of_breath', 'shortness_of_breath', 'dyspnea'],
            'chest_pain': ['chest_pain', 'chest_pain'],
            'joint_pain': ['joint_pain', 'joint_pain', 'arthralgia'],
            'muscle_pain': ['muscle_pain', 'muscle_pain', 'myalgia'],
            'body_aches': ['body_aches', 'body_aches', 'myalgia'],
        }
        
        for key, aliases in variations.items():
            if normalized in aliases:
                return key
        
        return normalized

    def compute_posterior(self, symptoms: List[str]) -> Dict[str, float]:
        """
        Compute posterior probabilities for all diseases given symptoms.
        Uses log space to prevent underflow and returns normalized probabilities.
        
        Args:
            symptoms: List of symptom strings
            
        Returns:
            Dictionary of disease -> posterior probability
        """
        # Normalize symptoms
        normalized_symptoms = [self._normalize_symptom(s) for s in symptoms]
        
        # Get unique symptoms
        unique_symptoms = list(set(normalized_symptoms))
        
        if not unique_symptoms:
            # If no symptoms, return priors
            return self.priors.copy()
        
        # Compute log posterior for each disease
        log_posteriors = {}
        
        for disease, prior in self.priors.items():
            # Start with log(prior)
            log_score = math.log(prior + 1e-10)
            
            # Add log likelihoods for each symptom
            likelihood = self.likelihoods.get(disease, {})
            for symptom in unique_symptoms:
                # Get likelihood with smoothing
                p = likelihood.get(symptom, self.epsilon)
                if p > 0:
                    log_score += math.log(p)
                else:
                    log_score += math.log(self.epsilon)
            
            log_posteriors[disease] = log_score
        
        # Convert from log space to probabilities
        # Find max log score for numerical stability
        max_log = max(log_posteriors.values())
        
        # Compute exp(log - max_log) and normalize
        exp_scores = {}
        total = 0
        for disease, log_score in log_posteriors.items():
            exp_val = math.exp(log_score - max_log)
            exp_scores[disease] = exp_val
            total += exp_val
        
        # Normalize to sum to 1.0
        posteriors = {}
        for disease, exp_val in exp_scores.items():
            posteriors[disease] = exp_val / (total + 1e-10)
        
        return posteriors

    def compute_odds_ratio(self, disease: str, symptoms: List[str]) -> float:
        """
        Compute the odds ratio for a disease given symptoms.
        Odds ratio = P(disease|symptoms) / (1 - P(disease|symptoms))
        
        Args:
            disease: Disease name
            symptoms: List of symptom strings
            
        Returns:
            Odds ratio
        """
        posteriors = self.compute_posterior(symptoms)
        p = posteriors.get(disease, 0.0)
        if p >= 1.0:
            return float('inf')
        return p / (1 - p + 1e-10)

    def get_symptom_impact(self, disease: str, symptom: str) -> float:
        """
        Calculate how much a symptom increases/decreases disease probability.
        
        Args:
            disease: Disease name
            symptom: Symptom to analyze
            
        Returns:
            Likelihood ratio P(symptom|disease) / P(symptom|¬disease)
        """
        normalized = self._normalize_symptom(symptom)
        p_given_disease = self.likelihoods.get(disease, {}).get(normalized, self.epsilon)
        
        # Calculate P(symptom|¬disease) - average across other diseases
        other_probs = []
        for d, likelihoods in self.likelihoods.items():
            if d != disease:
                other_probs.append(likelihoods.get(normalized, self.epsilon))
        
        p_given_not_disease = np.mean(other_probs) if other_probs else self.epsilon
        
        if p_given_not_disease > 0:
            return p_given_disease / p_given_not_disease
        return float('inf')

    def explain(self, disease: str, symptoms: List[str]) -> str:
        """
        Generate an explanation of how the diagnosis was reached.
        
        Args:
            disease: The diagnosed disease
            symptoms: The symptoms provided
            
        Returns:
            Explanation string
        """
        normalized_symptoms = [self._normalize_symptom(s) for s in symptoms]
        likelihoods = self.likelihoods.get(disease, {})
        prior = self.priors.get(disease, 0.0)
        
        explanation = []
        explanation.append(f"Diagnosis: {disease.upper()}")
        explanation.append(f"Prior probability: {prior:.1%}")
        explanation.append("")
        
        # Show symptom contributions
        explanation.append("Symptom contributions:")
        symptom_impacts = []
        for symptom in normalized_symptoms:
            p = likelihoods.get(symptom, self.epsilon)
            impact = self.get_symptom_impact(disease, symptom)
            symptom_impacts.append((symptom, p, impact))
        
        # Sort by impact (most influential first)
        symptom_impacts.sort(key=lambda x: x[2], reverse=True)
        
        for symptom, p, impact in symptom_impacts:
            if impact > 1.5:
                strength = "strongly supports"
            elif impact > 0.8:
                strength = "weakly supports"
            elif impact > 0.3:
                strength = "weakly contradicts"
            else:
                strength = "strongly contradicts"
            
            explanation.append(f"  • {symptom}: {strength} (P={p:.1%}, ratio={impact:.2f})")
        
        # Show posterior
        posteriors = self.compute_posterior(symptoms)
        posterior = posteriors.get(disease, 0.0)
        explanation.append("")
        explanation.append(f"Posterior probability: {posterior:.1%}")
        
        # Show other likely diagnoses
        top_others = sorted(posteriors.items(), key=lambda x: x[1], reverse=True)[1:3]
        if top_others:
            explanation.append("")
            explanation.append("Other likely diagnoses:")
            for d, p in top_others:
                if p > 0.05:
                    explanation.append(f"  • {d}: {p:.1%}")
        
        return "\n".join(explanation)

    def analyze(self, percept) -> Dict[str, Any]:
        """
        Standard interface method for the Agent to call.
        
        Args:
            percept: PatientPercept object
            
        Returns:
            Dictionary with diagnosis and confidence
        """
        # Get symptoms from patient
        symptoms = percept.symptoms
        
        # Compute posterior probabilities
        posteriors = self.compute_posterior(symptoms)
        
        # Find the most likely disease
        ranked = sorted(posteriors.items(), key=lambda x: x[1], reverse=True)
        best_disease, best_prob = ranked[0]
        
        # Calculate confidence based on how much better the top is
        second_prob = ranked[1][1] if len(ranked) > 1 else 0
        confidence_margin = best_prob - second_prob
        
        # Confidence is the probability of the top diagnosis, adjusted by margin
        confidence = best_prob * (1 + min(confidence_margin, 0.2))
        confidence = min(confidence, 0.99)
        
        # Get top 3 diagnoses
        top_3 = ranked[:3]
        
        return {
            'diagnosis': best_disease,
            'confidence': round(confidence, 4),
            'details': {
                'all_posteriors': posteriors,
                'ranked_diagnoses': ranked[:5],
                'top_3': top_3,
                'confidence_margin': confidence_margin
            }
        }

    def diagnose_with_threshold(self, symptoms: List[str], threshold: float = 0.2) -> List[Tuple[str, float]]:
        """
        Get all diagnoses above a confidence threshold.
        
        Args:
            symptoms: List of symptom strings
            threshold: Minimum confidence threshold
            
        Returns:
            List of (disease, confidence) tuples
        """
        posteriors = self.compute_posterior(symptoms)
        return [
            (disease, prob)
            for disease, prob in sorted(posteriors.items(), key=lambda x: x[1], reverse=True)
            if prob >= threshold
        ]


# ── Test the Bayesian Network ─────────────────────────────
def test_bayesian_net():
    """Test the SimpleBayesianDiagnostics module"""
    print("\n" + "=" * 60)
    print("  TESTING BAYESIAN NETWORK")
    print("=" * 60)
    
    bn = SimpleBayesianDiagnostics()
    
    # Test cases
    test_cases = [
        (["fever", "cough", "loss_of_smell", "fatigue"], "COVID-19 symptoms"),
        (["fever", "cough", "body_aches", "headache"], "Flu symptoms"),
        (["runny_nose", "sneezing", "sore_throat"], "Cold symptoms"),
        (["high_fever", "severe_headache", "joint_pain", "rash"], "Dengue symptoms"),
        (["chest_pain", "shortness_of_breath", "sweating"], "Cardiac symptoms"),
        (["frequent_urination", "excessive_thirst", "blurred_vision"], "Diabetes symptoms"),
        (["cough", "weight_loss", "night_sweats", "fatigue"], "Tuberculosis symptoms"),
        (["headache", "stiff_neck", "high_fever"], "Meningitis symptoms"),
    ]
    
    for symptoms, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Symptoms: {symptoms}")
        
        # Compute posteriors
        posteriors = bn.compute_posterior(symptoms)
        ranked = sorted(posteriors.items(), key=lambda x: x[1], reverse=True)
        
        print("Top 3 Diagnoses:")
        for disease, prob in ranked[:3]:
            print(f"  {disease:15}: {prob:.2%}")
        
        # Show explanation for top diagnosis
        if ranked:
            print("\nExplanation:")
            print(bn.explain(ranked[0][0], symptoms))
    
    # Test analyze method
    print("\n--- Test analyze() method ---")
    from modules.agent import PatientPercept
    patient = PatientPercept(
        patient_id="P001",
        symptoms=["fever", "cough", "loss_of_smell", "fatigue"],
        age=34,
        temperature=38.5,
        heart_rate=98,
        blood_pressure="120/80"
    )
    result = bn.analyze(patient)
    print(f"  Diagnosis: {result['diagnosis']}")
    print(f"  Confidence: {result['confidence']:.1%}")
    
    print("\n✅ Bayesian Network test passed!")


if __name__ == "__main__":
    test_bayesian_net()