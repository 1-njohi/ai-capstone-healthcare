# ============================================================
# DATA LOADER
# Load and preprocess data for the AI healthcare system
# ============================================================

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


class DataLoader:
    """
    Load and preprocess data for the healthcare diagnostic system.
    Supports CSV, JSON, and synthetic data formats.
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the data directory
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Data files
        self.symptoms_file = self.data_dir / "symptoms.csv"
        self.diseases_file = self.data_dir / "diseases.csv"
        self.patients_file = self.data_dir / "patient_records.csv"
        self.symptom_matrix_file = self.data_dir / "disease_symptom_matrix.csv"
        
        # DataFrames
        self.symptoms_df = None
        self.diseases_df = None
        self.patients_df = None
        self.symptom_matrix_df = None
        
        # Load all data if files exist
        self._load_all()
        
        print(f"[DataLoader] Initialized with data directory: {self.data_dir}")

    def _load_all(self):
        """Load all data files if they exist"""
        if self.symptoms_file.exists():
            self.symptoms_df = pd.read_csv(self.symptoms_file)
            print(f"[DataLoader] Loaded {len(self.symptoms_df)} symptoms")
        
        if self.diseases_file.exists():
            self.diseases_df = pd.read_csv(self.diseases_file)
            print(f"[DataLoader] Loaded {len(self.diseases_df)} diseases")
        
        if self.patients_file.exists():
            self.patients_df = pd.read_csv(self.patients_file)
            print(f"[DataLoader] Loaded {len(self.patients_df)} patient records")
        
        if self.symptom_matrix_file.exists():
            self.symptom_matrix_df = pd.read_csv(self.symptom_matrix_file)
            print(f"[DataLoader] Loaded {len(self.symptom_matrix_df)} disease-symptom mappings")

    # ── Symptom Methods ──

    def get_symptom_list(self) -> List[str]:
        """Get list of all symptom names"""
        if self.symptoms_df is not None:
            return self.symptoms_df['symptom_name'].tolist()
        return []

    def get_symptom_by_id(self, symptom_id: str) -> Optional[Dict[str, Any]]:
        """Get symptom details by ID"""
        if self.symptoms_df is None:
            return None
        row = self.symptoms_df[self.symptoms_df['symptom_id'] == symptom_id]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    def get_symptom_id(self, symptom_name: str) -> Optional[str]:
        """Get symptom ID by name"""
        if self.symptoms_df is None:
            return None
        row = self.symptoms_df[self.symptoms_df['symptom_name'] == symptom_name]
        if row.empty:
            return None
        return row.iloc[0]['symptom_id']

    def get_all_symptom_data(self) -> Dict[str, Dict[str, Any]]:
        """Get all symptom data as a dictionary keyed by symptom name"""
        if self.symptoms_df is None:
            return {}
        
        result = {}
        for _, row in self.symptoms_df.iterrows():
            result[row['symptom_name']] = {
                'id': row['symptom_id'],
                'category': row['category'],
                'severity_weight': row['severity_weight']
            }
        return result

    # ── Disease Methods ──

    def get_disease_list(self) -> List[str]:
        """Get list of all disease names"""
        if self.diseases_df is not None:
            return self.diseases_df['disease_name'].tolist()
        return []

    def get_disease_by_id(self, disease_id: str) -> Optional[Dict[str, Any]]:
        """Get disease details by ID"""
        if self.diseases_df is None:
            return None
        row = self.diseases_df[self.diseases_df['disease_id'] == disease_id]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    def get_disease_id(self, disease_name: str) -> Optional[str]:
        """Get disease ID by name"""
        if self.diseases_df is None:
            return None
        row = self.diseases_df[self.diseases_df['disease_name'] == disease_name]
        if row.empty:
            return None
        return row.iloc[0]['disease_id']

    def get_disease_info(self, disease: str) -> Dict[str, Any]:
        """Get information about a specific disease"""
        if self.diseases_df is None:
            return {}
        
        row = self.diseases_df[self.diseases_df['disease_name'] == disease]
        if row.empty:
            return {}
        
        return row.iloc[0].to_dict()

    def get_disease_severity(self, disease: str) -> str:
        """Get severity level of a disease"""
        info = self.get_disease_info(disease)
        return info.get('severity', 'UNKNOWN')

    def get_all_disease_data(self) -> Dict[str, Dict[str, Any]]:
        """Get all disease data as a dictionary keyed by disease name"""
        if self.diseases_df is None:
            return {}
        
        result = {}
        for _, row in self.diseases_df.iterrows():
            result[row['disease_name']] = {
                'id': row['disease_id'],
                'category': row['category'],
                'severity': row['severity'],
                'contagious': row['contagious'],
                'treatable': row['treatable']
            }
        return result

    # ── Symptom-Disease Matrix Methods ──

    def get_symptom_probabilities(self, disease: str) -> Dict[str, float]:
        """Get symptom probabilities for a specific disease"""
        if self.symptom_matrix_df is None or self.diseases_df is None or self.symptoms_df is None:
            return {}
        
        # Get disease_id
        disease_row = self.diseases_df[self.diseases_df['disease_name'] == disease]
        if disease_row.empty:
            return {}
        
        disease_id = disease_row.iloc[0]['disease_id']
        
        # Get symptom probabilities
        matrix = self.symptom_matrix_df[self.symptom_matrix_df['disease_id'] == disease_id]
        
        # Merge with symptom names
        result = {}
        for _, row in matrix.iterrows():
            symptom_id = row['symptom_id']
            symptom_row = self.symptoms_df[self.symptoms_df['symptom_id'] == symptom_id]
            if not symptom_row.empty:
                symptom_name = symptom_row.iloc[0]['symptom_name']
                result[symptom_name] = row['probability']
        
        return result

    def get_disease_symptom_matrix(self) -> pd.DataFrame:
        """Get the full disease-symptom matrix"""
        return self.symptom_matrix_df

    # ── Patient Methods ──

    def get_patients(self) -> pd.DataFrame:
        """Get all patient records"""
        return self.patients_df

    def get_patient_by_id(self, patient_id: str) -> Dict[str, Any]:
        """Get a specific patient record"""
        if self.patients_df is None:
            return {}
        
        row = self.patients_df[self.patients_df['patient_id'] == patient_id]
        if row.empty:
            return {}
        
        return row.iloc[0].to_dict()

    def get_patients_by_disease(self, disease: str) -> pd.DataFrame:
        """Get all patients with a specific disease"""
        if self.patients_df is None:
            return pd.DataFrame()
        
        return self.patients_df[self.patients_df['disease'] == disease]

    def get_patients_by_age_range(self, min_age: int, max_age: int) -> pd.DataFrame:
        """Get patients within an age range"""
        if self.patients_df is None:
            return pd.DataFrame()
        
        return self.patients_df[(self.patients_df['age'] >= min_age) & 
                                (self.patients_df['age'] <= max_age)]

    def get_patient_symptoms(self, patient_id: str) -> List[str]:
        """Get symptoms for a specific patient"""
        patient = self.get_patient_by_id(patient_id)
        if not patient:
            return []
        return self.parse_symptoms(patient.get('symptoms', ''))

    def get_patient_vitals(self, patient_id: str) -> Dict[str, float]:
        """Get vitals for a specific patient"""
        patient = self.get_patient_by_id(patient_id)
        if not patient:
            return {}
        
        return {
            'temperature': patient.get('temperature', 37.0),
            'heart_rate': patient.get('heart_rate', 75),
            'systolic': patient.get('systolic', 120),
            'diastolic': patient.get('diastolic', 80)
        }

    # ── Utility Methods ──

    def parse_symptoms(self, symptoms_str: str) -> List[str]:
        """Parse a comma-separated symptoms string into a list"""
        if pd.isna(symptoms_str) or symptoms_str == '':
            return []
        return [s.strip() for s in symptoms_str.split(',')]

    def symptoms_to_vector(self, symptoms: List[str], symptom_list: Optional[List[str]] = None) -> np.ndarray:
        """
        Convert a list of symptoms to a binary feature vector.
        
        Args:
            symptoms: List of symptom names
            symptom_list: Optional list of all symptoms (uses loaded list if not provided)
            
        Returns:
            Binary numpy array
        """
        if symptom_list is None:
            symptom_list = self.get_symptom_list()
        
        vector = np.zeros(len(symptom_list))
        for i, symptom in enumerate(symptom_list):
            if symptom in symptoms:
                vector[i] = 1
        return vector

    def get_disease_distribution(self) -> Dict[str, int]:
        """Get distribution of diseases in patient records"""
        if self.patients_df is None:
            return {}
        return self.patients_df['disease'].value_counts().to_dict()

    def get_symptom_frequency(self) -> Dict[str, int]:
        """Get frequency of each symptom in patient records"""
        if self.patients_df is None:
            return {}
        
        symptom_counts = {symptom: 0 for symptom in self.get_symptom_list()}
        for _, row in self.patients_df.iterrows():
            symptoms = self.parse_symptoms(row.get('symptoms', ''))
            for symptom in symptoms:
                if symptom in symptom_counts:
                    symptom_counts[symptom] += 1
        
        return symptom_counts

    def get_data_summary(self) -> Dict[str, Any]:
        """Get a summary of all loaded data"""
        return {
            'symptoms': {
                'count': len(self.symptoms_df) if self.symptoms_df is not None else 0,
                'categories': self.symptoms_df['category'].unique().tolist() if self.symptoms_df is not None else []
            },
            'diseases': {
                'count': len(self.diseases_df) if self.diseases_df is not None else 0,
                'categories': self.diseases_df['category'].unique().tolist() if self.diseases_df is not None else []
            },
            'patients': {
                'count': len(self.patients_df) if self.patients_df is not None else 0,
                'diseases': self.get_disease_distribution()
            },
            'symptom_matrix': {
                'mappings': len(self.symptom_matrix_df) if self.symptom_matrix_df is not None else 0
            }
        }


# ── Test Data Loader ──────────────────────────────────────
def test_loader():
    """Test the DataLoader module"""
    print("\n" + "=" * 60)
    print("  TESTING DATA LOADER")
    print("=" * 60)
    
    loader = DataLoader()
    
    # Show summary
    summary = loader.get_data_summary()
    print(f"\n  Data Summary:")
    print(f"    Symptoms: {summary['symptoms']['count']}")
    print(f"    Diseases: {summary['diseases']['count']}")
    print(f"    Patients: {summary['patients']['count']}")
    print(f"    Disease-Symptom Mappings: {summary['symptom_matrix']['mappings']}")
    
    # Show disease distribution
    print(f"\n  Disease Distribution:")
    for disease, count in summary['patients']['diseases'].items():
        print(f"    {disease}: {count}")
    
    # Test getting symptom probabilities for a disease
    if loader.diseases_df is not None:
        disease = loader.diseases_df.iloc[0]['disease_name']
        probs = loader.get_symptom_probabilities(disease)
        print(f"\n  Top symptoms for {disease}:")
        top_symptoms = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
        for symptom, prob in top_symptoms:
            print(f"    {symptom}: {prob:.2%}")
    
    # Test getting a patient
    if loader.patients_df is not None and len(loader.patients_df) > 0:
        patient_id = loader.patients_df.iloc[0]['patient_id']
        patient = loader.get_patient_by_id(patient_id)
        print(f"\n  Sample Patient: {patient_id}")
        print(f"    Age: {patient.get('age')}, Gender: {patient.get('gender')}")
        print(f"    Disease: {patient.get('disease')}")
        print(f"    Symptoms: {patient.get('symptoms')}")
        print(f"    Temperature: {patient.get('temperature')}°C")
        print(f"    Heart Rate: {patient.get('heart_rate')} BPM")
    
    print("\n✅ Data Loader test passed!")


if __name__ == "__main__":
    test_loader()