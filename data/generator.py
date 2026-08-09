# ============================================================
# DATA GENERATOR
# Programmatically generate all data files for the system
# ============================================================

import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime, timedelta
import random


class DataGenerator:
    """
    Generate all data files for the healthcare diagnostic system.
    Creates symptoms.csv, diseases.csv, patient_records.csv,
    and disease_symptom_matrix.csv
    """

    def __init__(self, data_dir: str = "data", seed: int = 42):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(seed)
        random.seed(seed)
        
        # Define symptom features for binary columns
        self.symptom_features = [
            'fever', 'cough', 'fatigue', 'headache',
            'body_aches', 'loss_of_smell', 'chest_pain',
            'rash', 'joint_pain', 'shortness_of_breath',
            'sweating', 'frequent_urination', 'excessive_thirst',
            'blurred_vision', 'night_sweats', 'weight_loss',
            'stiff_neck', 'light_sensitivity', 'sore_throat',
            'runny_nose', 'sneezing', 'nausea'
        ]
        
        print(f"[DataGenerator] Initialized with data directory: {self.data_dir}")

    def generate_all(self, n_patients: int = 2000):
        """Generate all data files"""
        print("\n" + "=" * 60)
        print("  GENERATING DATA FILES")
        print("=" * 60)
        
        self.generate_symptoms()
        self.generate_diseases()
        self.generate_disease_symptom_matrix()
        self.generate_patient_records(n_patients=n_patients)
        
        print("\n" + "=" * 60)
        print("  ✅ ALL DATA FILES GENERATED")
        print("=" * 60)

    def generate_symptoms(self):
        """Generate symptoms.csv"""
        symptoms = [
            # ID, Name, Category, Severity Weight (1-10)
            ("S001", "fever", "general", 8),
            ("S002", "cough", "respiratory", 7),
            ("S003", "fatigue", "general", 6),
            ("S004", "headache", "neurological", 5),
            ("S005", "body_aches", "musculoskeletal", 6),
            ("S006", "loss_of_smell", "neurological", 7),
            ("S007", "chest_pain", "cardiac", 9),
            ("S008", "rash", "dermatological", 5),
            ("S009", "joint_pain", "musculoskeletal", 6),
            ("S010", "shortness_of_breath", "respiratory", 8),
            ("S011", "sweating", "general", 5),
            ("S012", "frequent_urination", "urinary", 6),
            ("S013", "excessive_thirst", "metabolic", 6),
            ("S014", "blurred_vision", "neurological", 6),
            ("S015", "night_sweats", "general", 6),
            ("S016", "weight_loss", "metabolic", 7),
            ("S017", "stiff_neck", "neurological", 7),
            ("S018", "light_sensitivity", "neurological", 6),
            ("S019", "sore_throat", "respiratory", 4),
            ("S020", "runny_nose", "respiratory", 3),
            ("S021", "sneezing", "respiratory", 3),
            ("S022", "nausea", "gastrointestinal", 5),
            ("S023", "vomiting", "gastrointestinal", 6),
            ("S024", "diarrhea", "gastrointestinal", 6),
            ("S025", "chills", "general", 5),
            ("S026", "mucus_production", "respiratory", 4),
            ("S027", "itchy_eyes", "ophthalmic", 3),
            ("S028", "swollen_lymph_nodes", "immune", 5),
            ("S029", "abdominal_pain", "gastrointestinal", 6),
            ("S030", "high_fever", "general", 9),
        ]
        
        df = pd.DataFrame(symptoms, columns=['symptom_id', 'symptom_name', 'category', 'severity_weight'])
        filepath = self.data_dir / "symptoms.csv"
        df.to_csv(filepath, index=False)
        print(f"  ✅ Generated symptoms.csv ({len(df)} symptoms)")
        return df

    def generate_diseases(self):
        """Generate diseases.csv"""
        diseases = [
            # ID, Name, Category, Severity, Contagious, Treatable
            ("D001", "flu", "viral", "MEDIUM", True, True),
            ("D002", "covid19", "viral", "HIGH", True, True),
            ("D003", "dengue", "viral", "HIGH", False, True),
            ("D004", "cardiac_event", "cardiac", "CRITICAL", False, True),
            ("D005", "diabetes", "metabolic", "MEDIUM", False, True),
            ("D006", "common_cold", "viral", "LOW", True, True),
            ("D007", "tuberculosis", "bacterial", "HIGH", True, True),
            ("D008", "meningitis", "bacterial", "CRITICAL", True, True),
            ("D009", "pneumonia", "bacterial", "HIGH", True, True),
            ("D010", "bronchitis", "bacterial", "MEDIUM", True, True),
            ("D011", "allergies", "immune", "LOW", False, True),
            ("D012", "gastroenteritis", "viral", "MEDIUM", True, True),
            ("D013", "strep_throat", "bacterial", "MEDIUM", True, True),
        ]
        
        df = pd.DataFrame(diseases, columns=['disease_id', 'disease_name', 'category', 'severity', 'contagious', 'treatable'])
        filepath = self.data_dir / "diseases.csv"
        df.to_csv(filepath, index=False)
        print(f"  ✅ Generated diseases.csv ({len(df)} diseases)")
        return df

    def generate_disease_symptom_matrix(self):
        """Generate disease_symptom_matrix.csv with probability mappings"""
        
        # Disease to symptom probabilities
        # Format: (disease_id, symptom_id, probability)
        matrix = [
            # ── Flu (D001) ──
            ("D001", "S001", 0.90),  # fever
            ("D001", "S002", 0.85),  # cough
            ("D001", "S003", 0.88),  # fatigue
            ("D001", "S004", 0.70),  # headache
            ("D001", "S005", 0.80),  # body_aches
            ("D001", "S025", 0.70),  # chills
            ("D001", "S019", 0.60),  # sore_throat
            ("D001", "S020", 0.50),  # runny_nose
            ("D001", "S021", 0.30),  # sneezing
            ("D001", "S022", 0.20),  # nausea
            ("D001", "S009", 0.40),  # joint_pain
            ("D001", "S011", 0.30),  # sweating
            
            # ── COVID-19 (D002) ──
            ("D002", "S001", 0.88),  # fever
            ("D002", "S002", 0.80),  # cough
            ("D002", "S003", 0.90),  # fatigue
            ("D002", "S006", 0.85),  # loss_of_smell
            ("D002", "S004", 0.65),  # headache
            ("D002", "S005", 0.60),  # body_aches
            ("D002", "S010", 0.45),  # shortness_of_breath
            ("D002", "S007", 0.20),  # chest_pain
            ("D002", "S011", 0.20),  # sweating
            ("D002", "S019", 0.40),  # sore_throat
            ("D002", "S020", 0.30),  # runny_nose
            ("D002", "S021", 0.15),  # sneezing
            ("D002", "S022", 0.25),  # nausea
            ("D002", "S009", 0.20),  # joint_pain
            
            # ── Dengue (D003) ──
            ("D003", "S030", 0.98),  # high_fever
            ("D003", "S008", 0.75),  # rash
            ("D003", "S009", 0.80),  # joint_pain
            ("D003", "S004", 0.90),  # headache
            ("D003", "S003", 0.80),  # fatigue
            ("D003", "S005", 0.88),  # body_aches
            ("D003", "S011", 0.50),  # sweating
            ("D003", "S022", 0.60),  # nausea
            ("D003", "S023", 0.50),  # vomiting
            ("D003", "S024", 0.20),  # diarrhea
            
            # ── Cardiac Event (D004) ──
            ("D004", "S007", 0.92),  # chest_pain
            ("D004", "S010", 0.88),  # shortness_of_breath
            ("D004", "S003", 0.70),  # fatigue
            ("D004", "S011", 0.75),  # sweating
            ("D004", "S004", 0.30),  # headache
            ("D004", "S005", 0.20),  # body_aches
            ("D004", "S022", 0.40),  # nausea
            
            # ── Diabetes (D005) ──
            ("D005", "S003", 0.82),  # fatigue
            ("D005", "S012", 0.95),  # frequent_urination
            ("D005", "S013", 0.92),  # excessive_thirst
            ("D005", "S014", 0.70),  # blurred_vision
            ("D005", "S016", 0.50),  # weight_loss
            
            # ── Common Cold (D006) ──
            ("D006", "S020", 0.85),  # runny_nose
            ("D006", "S021", 0.80),  # sneezing
            ("D006", "S019", 0.75),  # sore_throat
            ("D006", "S002", 0.60),  # cough
            ("D006", "S001", 0.30),  # fever
            ("D006", "S004", 0.35),  # headache
            ("D006", "S003", 0.30),  # fatigue
            ("D006", "S005", 0.15),  # body_aches
            ("D006", "S006", 0.30),  # loss_of_smell
            
            # ── Tuberculosis (D007) ──
            ("D007", "S002", 0.95),  # cough
            ("D007", "S016", 0.85),  # weight_loss
            ("D007", "S015", 0.80),  # night_sweats
            ("D007", "S003", 0.88),  # fatigue
            ("D007", "S001", 0.70),  # fever
            ("D007", "S007", 0.30),  # chest_pain
            ("D007", "S010", 0.35),  # shortness_of_breath
            ("D007", "S011", 0.60),  # sweating
            ("D007", "S004", 0.20),  # headache
            ("D007", "S005", 0.25),  # body_aches
            
            # ── Meningitis (D008) ──
            ("D008", "S004", 0.95),  # headache
            ("D008", "S017", 0.90),  # stiff_neck
            ("D008", "S001", 0.92),  # fever
            ("D008", "S018", 0.85),  # light_sensitivity
            ("D008", "S003", 0.80),  # fatigue
            ("D008", "S022", 0.60),  # nausea
            ("D008", "S023", 0.50),  # vomiting
            ("D008", "S008", 0.20),  # rash
            
            # ── Pneumonia (D009) ──
            ("D009", "S001", 0.92),  # fever
            ("D009", "S002", 0.90),  # cough
            ("D009", "S010", 0.85),  # shortness_of_breath
            ("D009", "S007", 0.75),  # chest_pain
            ("D009", "S003", 0.80),  # fatigue
            ("D009", "S011", 0.70),  # sweating
            ("D009", "S004", 0.40),  # headache
            ("D009", "S005", 0.50),  # body_aches
            ("D009", "S022", 0.25),  # nausea
            ("D009", "S025", 0.60),  # chills
            
            # ── Bronchitis (D010) ──
            ("D010", "S002", 0.95),  # cough
            ("D010", "S026", 0.90),  # mucus_production
            ("D010", "S003", 0.80),  # fatigue
            ("D010", "S010", 0.70),  # shortness_of_breath
            ("D010", "S007", 0.65),  # chest_pain
            ("D010", "S001", 0.40),  # fever
            ("D010", "S019", 0.30),  # sore_throat
            ("D010", "S004", 0.25),  # headache
            ("D010", "S005", 0.20),  # body_aches
            ("D010", "S025", 0.20),  # chills
            
            # ── Allergies (D011) ──
            ("D011", "S020", 0.90),  # runny_nose
            ("D011", "S021", 0.90),  # sneezing
            ("D011", "S027", 0.85),  # itchy_eyes
            ("D011", "S019", 0.40),  # sore_throat
            ("D011", "S002", 0.30),  # cough
            ("D011", "S003", 0.20),  # fatigue
            
            # ── Gastroenteritis (D012) ──
            ("D012", "S022", 0.90),  # nausea
            ("D012", "S023", 0.85),  # vomiting
            ("D012", "S024", 0.85),  # diarrhea
            ("D012", "S029", 0.80),  # abdominal_pain
            ("D012", "S001", 0.55),  # fever
            ("D012", "S003", 0.50),  # fatigue
            
            # ── Strep Throat (D013) ──
            ("D013", "S019", 0.95),  # sore_throat
            ("D013", "S001", 0.80),  # fever
            ("D013", "S028", 0.85),  # swollen_lymph_nodes
            ("D013", "S004", 0.60),  # headache
            ("D013", "S022", 0.40),  # nausea
        ]
        
        df = pd.DataFrame(matrix, columns=['disease_id', 'symptom_id', 'probability'])
        filepath = self.data_dir / "disease_symptom_matrix.csv"
        df.to_csv(filepath, index=False)
        print(f"  ✅ Generated disease_symptom_matrix.csv ({len(df)} mappings)")
        return df

    def generate_patient_records(self, n_patients: int = 2000):
        """
        Generate patient_records.csv with realistic patient data.
        Creates binary symptom columns for easy ML training.
        
        Args:
            n_patients: Number of patient records to generate
        """
        print(f"\n  Generating {n_patients} patient records...")
        
        # Load disease and symptom data
        diseases_df = pd.read_csv(self.data_dir / "diseases.csv")
        symptoms_df = pd.read_csv(self.data_dir / "symptoms.csv")
        matrix_df = pd.read_csv(self.data_dir / "disease_symptom_matrix.csv")
        
        # Build symptom probability lookup
        disease_symptoms = {}
        for disease_id in diseases_df['disease_id']:
            disease_name = diseases_df[diseases_df['disease_id'] == disease_id].iloc[0]['disease_name']
            probs = matrix_df[matrix_df['disease_id'] == disease_id]
            symptom_probs = {}
            for _, row in probs.iterrows():
                symptom_id = row['symptom_id']
                symptom_name = symptoms_df[symptoms_df['symptom_id'] == symptom_id].iloc[0]['symptom_name']
                symptom_probs[symptom_name] = row['probability']
            disease_symptoms[disease_name] = symptom_probs
        
        # Generate patient records
        records = []
        start_date = datetime(2024, 1, 1)
        
        # Define age ranges for diseases
        age_ranges = {
            'flu': (18, 65),
            'covid19': (18, 80),
            'dengue': (15, 50),
            'cardiac_event': (45, 80),
            'diabetes': (35, 70),
            'common_cold': (5, 80),
            'tuberculosis': (25, 60),
            'meningitis': (15, 40),
            'pneumonia': (40, 80),
            'bronchitis': (30, 65),
            'allergies': (10, 60),
            'gastroenteritis': (15, 50),
            'strep_throat': (5, 40),
        }
        
        # Generate base vitals for each disease
        vitals_base = {
            'flu': {'temp': 38.5, 'hr': 95, 'sys': 125, 'dia': 80},
            'covid19': {'temp': 38.5, 'hr': 95, 'sys': 125, 'dia': 80},
            'dengue': {'temp': 39.0, 'hr': 100, 'sys': 120, 'dia': 75},
            'cardiac_event': {'temp': 37.0, 'hr': 115, 'sys': 145, 'dia': 90},
            'diabetes': {'temp': 36.8, 'hr': 80, 'sys': 130, 'dia': 85},
            'common_cold': {'temp': 37.2, 'hr': 72, 'sys': 118, 'dia': 76},
            'tuberculosis': {'temp': 38.2, 'hr': 95, 'sys': 125, 'dia': 80},
            'meningitis': {'temp': 39.5, 'hr': 110, 'sys': 120, 'dia': 75},
            'pneumonia': {'temp': 39.2, 'hr': 102, 'sys': 130, 'dia': 85},
            'bronchitis': {'temp': 38.0, 'hr': 90, 'sys': 125, 'dia': 80},
            'allergies': {'temp': 36.8, 'hr': 72, 'sys': 118, 'dia': 76},
            'gastroenteritis': {'temp': 38.0, 'hr': 85, 'sys': 120, 'dia': 75},
            'strep_throat': {'temp': 38.5, 'hr': 90, 'sys': 120, 'dia': 78},
        }
        
        # Disease weights for realistic distribution
        disease_weights = {
            'flu': 15,
            'covid19': 10,
            'dengue': 5,
            'cardiac_event': 8,
            'diabetes': 10,
            'common_cold': 20,
            'tuberculosis': 3,
            'meningitis': 2,
            'pneumonia': 5,
            'bronchitis': 7,
            'allergies': 10,
            'gastroenteritis': 4,
            'strep_throat': 6,
        }
        
        diseases = list(disease_weights.keys())
        weights = [disease_weights[d] for d in diseases]
        
        # Generate records
        for i in range(n_patients):
            # Select disease with weighting
            disease = random.choices(diseases, weights=weights)[0]
            
            age_min, age_max = age_ranges.get(disease, (18, 65))
            age = random.randint(age_min, age_max)
            gender = random.choice(['M', 'F'])
            
            # Generate symptoms based on disease profile
            probs = disease_symptoms.get(disease, {})
            symptoms = []
            for symptom, prob in probs.items():
                actual_prob = prob + random.uniform(-0.05, 0.05)
                actual_prob = max(0, min(1, actual_prob))
                if random.random() < actual_prob:
                    symptoms.append(symptom)
            
            # Ensure at least 2 symptoms
            if len(symptoms) < 2:
                if probs:
                    symptom = random.choices(
                        list(probs.keys()),
                        weights=list(probs.values())
                    )[0]
                    if symptom not in symptoms:
                        symptoms.append(symptom)
                if len(symptoms) < 2:
                    common_symptoms = ['fatigue', 'headache']
                    for s in common_symptoms:
                        if s not in symptoms:
                            symptoms.append(s)
                            break
            
            # Generate vitals
            vitals = vitals_base.get(disease, {'temp': 37.0, 'hr': 75, 'sys': 120, 'dia': 80})
            temp = np.random.normal(vitals['temp'], 0.4)
            hr = int(np.random.normal(vitals['hr'], 10))
            sys = int(np.random.normal(vitals['sys'], 10))
            dia = int(np.random.normal(vitals['dia'], 6))
            
            # Generate date
            date = start_date + timedelta(days=random.randint(0, 365))
            
            # Create record with symptoms as comma-separated string
            record = {
                'patient_id': f"P{i+1:04d}",
                'age': age,
                'gender': gender,
                'temperature': round(temp, 1),
                'heart_rate': max(40, min(160, hr)),
                'systolic': max(90, min(180, sys)),
                'diastolic': max(60, min(110, dia)),
                'symptoms': ','.join(symptoms),
                'disease': disease,
                'diagnosis_date': date.strftime('%Y-%m-%d')
            }
            records.append(record)
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # ─── CREATE BINARY SYMPTOM COLUMNS ───
        print(f"  Creating binary symptom columns...")
        
        # Initialize all symptom columns to 0
        for symptom in self.symptom_features:
            df[symptom] = 0
        
        # Populate symptom columns from the symptoms string
        for idx, row in df.iterrows():
            symptoms_str = row.get('symptoms', '')
            if pd.isna(symptoms_str) or symptoms_str == '':
                continue
            symptoms_list = [s.strip().lower().replace(' ', '_') for s in symptoms_str.split(',')]
            for symptom in symptoms_list:
                if symptom in self.symptom_features:
                    df.at[idx, symptom] = 1
        
        # ─── END BINARY SYMPTOM COLUMNS ───
        
        # Save the CSV
        filepath = self.data_dir / "patient_records.csv"
        df.to_csv(filepath, index=False)
        print(f"  ✅ Generated patient_records.csv ({len(df)} patients)")
        
        # Print distribution
        print(f"\n  Disease distribution:")
        for disease, count in df['disease'].value_counts().items():
            print(f"    {disease}: {count} ({count/len(df)*100:.1f}%)")
        
        # Print column info
        print(f"\n  Columns: {len(df.columns)} total")
        print(f"  Binary symptom columns: {len(self.symptom_features)}")
        
        return df


# ── Main entry point ──────────────────────────────────────
def generate_data(n_patients: int = 2000):
    """Main function to generate all data files"""
    generator = DataGenerator()
    generator.generate_all(n_patients=n_patients)


if __name__ == "__main__":
    # Generate 2000 patient records by default
    generate_data(n_patients=2000)