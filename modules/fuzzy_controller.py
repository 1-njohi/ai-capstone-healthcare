# ============================================================
# MODULE 6: Fuzzy Logic — Patient Severity Assessment
# Covers: Week 12 (Fuzzy Logic & Fuzzy Control Systems)
# ============================================================

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt


class FuzzySeverityAssessor:
    """
    Fuzzy logic system for patient severity assessment.
    
    Inputs: Temperature, Heart Rate, Symptom Count
    Output: Severity Score (0-100) and Severity Label
    """

    def __init__(self):
        self.severity_ranges = {
            'LOW': (0, 20),
            'MILD': (20, 40),
            'MODERATE': (40, 60),
            'HIGH': (60, 80),
            'CRITICAL': (80, 100)
        }
        
        self.severity_centers = {
            'low': 15,
            'mild': 35,
            'moderate': 55,
            'high': 75,
            'critical': 92
        }
        
        self.last_result = None
        print("[FuzzyController] Initialized severity assessor")

    def _membership_temp(self, temp: float) -> Dict[str, float]:
        memberships = {
            'normal': 0.0,
            'mild': 0.0,
            'high': 0.0,
            'critical': 0.0
        }
        
        if 36.0 <= temp <= 37.5:
            if temp <= 36.8:
                memberships['normal'] = (temp - 36.0) / 0.8
            else:
                memberships['normal'] = (37.5 - temp) / 0.7
            memberships['normal'] = max(0, min(1, memberships['normal']))
        
        if 37.3 <= temp <= 38.5:
            if temp <= 38.0:
                memberships['mild'] = (temp - 37.3) / 0.7
            else:
                memberships['mild'] = (38.5 - temp) / 0.5
            memberships['mild'] = max(0, min(1, memberships['mild']))
        
        if 38.0 <= temp <= 39.5:
            if temp <= 39.0:
                memberships['high'] = (temp - 38.0) / 1.0
            else:
                memberships['high'] = (39.5 - temp) / 0.5
            memberships['high'] = max(0, min(1, memberships['high']))
        
        if temp >= 39.0:
            memberships['critical'] = min(1.0, (temp - 39.0) / 2.0)
            memberships['critical'] = max(0, min(1, memberships['critical']))
        
        return memberships

    def _membership_hr(self, hr: int) -> Dict[str, float]:
        memberships = {
            'low': 0.0,
            'normal': 0.0,
            'elevated': 0.0,
            'high': 0.0,
            'critical': 0.0
        }
        
        if hr < 60:
            memberships['low'] = max(0, min(1, (60 - hr) / 20.0))
        
        if 60 <= hr <= 100:
            if hr <= 80:
                memberships['normal'] = (hr - 60) / 20.0
            else:
                memberships['normal'] = (100 - hr) / 20.0
            memberships['normal'] = max(0, min(1, memberships['normal']))
        
        if 90 <= hr <= 110:
            if hr <= 100:
                memberships['elevated'] = (hr - 90) / 10.0
            else:
                memberships['elevated'] = (110 - hr) / 10.0
            memberships['elevated'] = max(0, min(1, memberships['elevated']))
        
        if 105 <= hr <= 125:
            if hr <= 115:
                memberships['high'] = (hr - 105) / 10.0
            else:
                memberships['high'] = (125 - hr) / 10.0
            memberships['high'] = max(0, min(1, memberships['high']))
        
        if hr >= 120:
            memberships['critical'] = min(1.0, (hr - 120) / 30.0)
            memberships['critical'] = max(0, min(1, memberships['critical']))
        
        return memberships

    def _membership_symptoms(self, count: int) -> Dict[str, float]:
        memberships = {
            'few': 0.0,
            'several': 0.0,
            'many': 0.0,
            'extensive': 0.0
        }
        
        if 0 <= count <= 3:
            if count <= 1:
                memberships['few'] = 1.0
            else:
                memberships['few'] = (3 - count) / 2.0
            memberships['few'] = max(0, min(1, memberships['few']))
        
        if 2 <= count <= 5:
            if count <= 3.5:
                memberships['several'] = (count - 2) / 1.5
            else:
                memberships['several'] = (5 - count) / 1.5
            memberships['several'] = max(0, min(1, memberships['several']))
        
        if 4 <= count <= 8:
            if count <= 6:
                memberships['many'] = (count - 4) / 2.0
            else:
                memberships['many'] = (8 - count) / 2.0
            memberships['many'] = max(0, min(1, memberships['many']))
        
        if count >= 7:
            memberships['extensive'] = min(1.0, (count - 7) / 4.0)
            memberships['extensive'] = max(0, min(1, memberships['extensive']))
        
        return memberships

    def _evaluate_rules(self, temp_mf: Dict[str, float], 
                         hr_mf: Dict[str, float], 
                         symptom_mf: Dict[str, float]) -> Dict[str, float]:
        rules = {
            'critical': 0.0,
            'high': 0.0,
            'moderate': 0.0,
            'mild': 0.0,
            'low': 0.0
        }
        
        # CRITICAL Rules
        rules['critical'] = max(rules['critical'], min(temp_mf.get('critical', 0), hr_mf.get('high', 0)))
        rules['critical'] = max(rules['critical'], min(temp_mf.get('critical', 0), symptom_mf.get('extensive', 0)))
        rules['critical'] = max(rules['critical'], min(hr_mf.get('critical', 0), symptom_mf.get('many', 0)))
        rules['critical'] = max(rules['critical'], min(temp_mf.get('high', 0), hr_mf.get('high', 0), symptom_mf.get('many', 0)))
        
        # HIGH Rules
        rules['high'] = max(rules['high'], min(temp_mf.get('high', 0), hr_mf.get('elevated', 0)))
        rules['high'] = max(rules['high'], min(temp_mf.get('high', 0), symptom_mf.get('many', 0)))
        rules['high'] = max(rules['high'], min(hr_mf.get('elevated', 0), symptom_mf.get('many', 0)))
        rules['high'] = max(rules['high'], min(temp_mf.get('mild', 0), hr_mf.get('high', 0), symptom_mf.get('many', 0)))
        
        # MODERATE Rules
        rules['moderate'] = max(rules['moderate'], min(temp_mf.get('mild', 0), hr_mf.get('elevated', 0)))
        rules['moderate'] = max(rules['moderate'], min(temp_mf.get('mild', 0), symptom_mf.get('several', 0)))
        rules['moderate'] = max(rules['moderate'], min(hr_mf.get('normal', 0), symptom_mf.get('many', 0)))
        rules['moderate'] = max(rules['moderate'], min(temp_mf.get('high', 0), hr_mf.get('normal', 0), symptom_mf.get('several', 0)))
        
        # MILD Rules
        rules['mild'] = max(rules['mild'], min(temp_mf.get('mild', 0), hr_mf.get('normal', 0)))
        rules['mild'] = max(rules['mild'], min(temp_mf.get('normal', 0), symptom_mf.get('several', 0)))
        rules['mild'] = max(rules['mild'], min(hr_mf.get('elevated', 0), symptom_mf.get('few', 0)))
        rules['mild'] = max(rules['mild'], min(temp_mf.get('mild', 0), hr_mf.get('elevated', 0), symptom_mf.get('few', 0)))
        
        # LOW Rules
        rules['low'] = max(rules['low'], min(temp_mf.get('normal', 0), hr_mf.get('normal', 0)))
        rules['low'] = max(rules['low'], min(temp_mf.get('normal', 0), symptom_mf.get('few', 0)))
        rules['low'] = max(rules['low'], min(hr_mf.get('normal', 0), symptom_mf.get('few', 0)))
        rules['low'] = max(rules['low'], min(temp_mf.get('normal', 0), hr_mf.get('normal', 0), symptom_mf.get('few', 0)))
        
        for key in rules:
            rules[key] = max(0.0, min(1.0, rules[key]))
        
        return rules

    def _defuzzify(self, rules: Dict[str, float], method: str = 'centroid') -> float:
        if sum(rules.values()) < 0.01:
            return 10.0
        
        if method == 'centroid':
            numerator = 0.0
            denominator = 0.0
            
            for severity, strength in rules.items():
                center = self.severity_centers.get(severity, 0)
                numerator += center * strength
                denominator += strength
            
            denominator = max(denominator, 1e-10)
            score = numerator / denominator
        else:
            score = sum(
                self.severity_centers.get(severity, 0) * (strength ** 1.5)
                for severity, strength in rules.items()
            ) / sum(strength ** 1.5 for strength in rules.values())
        
        score = max(0, min(100, score))
        return score

    def _classify_severity(self, score: float) -> Tuple[str, str, str]:
        if score >= 80:
            return "CRITICAL", "red", "🚨"
        elif score >= 60:
            return "HIGH", "orange", "⚠️"
        elif score >= 40:
            return "MODERATE", "yellow", "📋"
        elif score >= 20:
            return "MILD", "blue", "💚"
        else:
            return "LOW", "green", "✅"

    def assess(self, temperature: float, heart_rate: int,
               symptom_count: int, verbose: bool = False) -> Dict:
        temp_mf = self._membership_temp(temperature)
        hr_mf = self._membership_hr(heart_rate)
        symptom_mf = self._membership_symptoms(symptom_count)
        
        rules = self._evaluate_rules(temp_mf, hr_mf, symptom_mf)
        
        severity_score = self._defuzzify(rules)
        severity_label, color, emoji = self._classify_severity(severity_score)
        
        self.last_result = {
            'temp_memberships': temp_mf,
            'hr_memberships': hr_mf,
            'symptom_memberships': symptom_mf,
            'rules': rules,
            'score': severity_score,
            'label': severity_label,
            'color': color,
            'emoji': emoji
        }
        
        response = {
            'severity_score': round(severity_score, 2),
            'severity_label': severity_label,
            'severity_color': color,
            'severity_emoji': emoji,
            'details': {
                'temperature': temperature,
                'heart_rate': heart_rate,
                'symptom_count': symptom_count,
                'memberships': {
                    'temperature': temp_mf,
                    'heart_rate': hr_mf,
                    'symptoms': symptom_mf
                },
                'rule_activations': rules,
                'dominant_rules': self._get_dominant_rules(rules)
            }
        }
        
        return response

    def _get_dominant_rules(self, rules: Dict[str, float]) -> List[Tuple[str, float]]:
        sorted_rules = sorted(rules.items(), key=lambda x: x[1], reverse=True)
        return [(name, strength) for name, strength in sorted_rules if strength > 0.1][:3]

    def explain(self, result: Dict) -> str:
        score = result['severity_score']
        label = result['severity_label']
        emoji = result['severity_emoji']
        details = result['details']
        
        explanation = []
        explanation.append("=" * 50)
        explanation.append(f"{emoji} SEVERITY ASSESSMENT EXPLANATION")
        explanation.append("=" * 50)
        explanation.append(f"\nOverall Severity: {label} ({score:.1f}/100)")
        explanation.append("")
        explanation.append("Input Values:")
        explanation.append(f"  • Temperature: {details['temperature']}°C")
        explanation.append(f"  • Heart Rate: {details['heart_rate']} BPM")
        explanation.append(f"  • Symptom Count: {details['symptom_count']}")
        explanation.append("")
        explanation.append("Key Contributing Factors:")
        
        for rule_name, strength in details.get('dominant_rules', []):
            explanation.append(f"  • {rule_name.upper()}: {strength:.2%}")
        
        explanation.append("")
        explanation.append("Recommendation:")
        
        if label == "CRITICAL":
            explanation.append("  🚨 IMMEDIATE EMERGENCY CARE REQUIRED!")
        elif label == "HIGH":
            explanation.append("  ⚠️ URGENT: See a doctor within 24 hours.")
        elif label == "MODERATE":
            explanation.append("  📋 Schedule a doctor's appointment within 3-5 days.")
        elif label == "MILD":
            explanation.append("  💚 Low urgency: Monitor symptoms at home.")
        else:
            explanation.append("  ✅ No immediate action required.")
        
        explanation.append("")
        explanation.append("=" * 50)
        return "\n".join(explanation)

    def visualize_memberships(self, temperature: float, heart_rate: int, 
                              symptom_count: int, save_path: Optional[str] = None):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1 = axes[0]
        temps = np.linspace(35, 42, 100)
        for temp in temps:
            mf = self._membership_temp(temp)
            for label, val in mf.items():
                if val > 0:
                    ax1.scatter(temp, val, s=10, label=label if temp == 35 else "")
        ax1.axvline(x=temperature, color='red', linestyle='--', label=f'Input: {temperature}°C')
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Membership Degree')
        ax1.set_title('Temperature Membership Functions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        hrs = np.linspace(30, 160, 100)
        for hr in hrs:
            mf = self._membership_hr(int(hr))
            for label, val in mf.items():
                if val > 0:
                    ax2.scatter(hr, val, s=10, label=label if hr == 30 else "")
        ax2.axvline(x=heart_rate, color='red', linestyle='--', label=f'Input: {heart_rate} BPM')
        ax2.set_xlabel('Heart Rate (BPM)')
        ax2.set_ylabel('Membership Degree')
        ax2.set_title('Heart Rate Membership Functions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[2]
        counts = np.linspace(0, 14, 100)
        for count in counts:
            mf = self._membership_symptoms(int(count))
            for label, val in mf.items():
                if val > 0:
                    ax3.scatter(count, val, s=10, label=label if count == 0 else "")
        ax3.axvline(x=symptom_count, color='red', linestyle='--', label=f'Input: {symptom_count}')
        ax3.set_xlabel('Symptom Count')
        ax3.set_ylabel('Membership Degree')
        ax3.set_title('Symptom Count Membership Functions')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Fuzzy Membership Functions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✅ Saved: {save_path}")
        
        plt.show()

    def analyze(self, percept) -> Dict:
        """
        Module interface for the agent.
        Returns severity as the primary output.
        """
        result = self.assess(
            percept.temperature,
            percept.heart_rate,
            len(percept.symptoms),
            verbose=False
        )
        
        return {
            'diagnosis': f"Severity: {result['severity_label']}",
            'confidence': result['severity_score'] / 100,
            'severity': result['severity_label'],
            'severity_score': result['severity_score'],
            'details': result['details'],
            'summary': f"{result['severity_emoji']} Severity: {result['severity_label']} ({result['severity_score']:.1f}/100)"
        }


def test_fuzzy_controller():
    """Test the FuzzySeverityAssessor module"""
    print("\n" + "=" * 60)
    print("  TESTING FUZZY LOGIC SEVERITY ASSESSOR")
    print("=" * 60)
    
    fa = FuzzySeverityAssessor()
    
    test_cases = [
        (37.0, 72, 2, "Normal patient"),
        (38.5, 95, 4, "Mild illness"),
        (39.8, 115, 7, "Severe case"),
        (40.2, 130, 9, "Critical case"),
    ]
    
    print("\n--- Severity Assessments ---")
    for temp, hr, count, desc in test_cases:
        print(f"\n{desc}:")
        result = fa.assess(temp, hr, count, verbose=False)
        print(f"  Severity Score: {result['severity_score']:.1f}")
        print(f"  Severity Label: {result['severity_emoji']} {result['severity_label']}")
    
    print("\n✅ Fuzzy Controller test passed!")


if __name__ == "__main__":
    test_fuzzy_controller()