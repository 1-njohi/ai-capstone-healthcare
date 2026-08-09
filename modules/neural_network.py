# ============================================================
# MODULE 5: Deep Neural Network Diagnostic Model
# Covers: Week 10 (Neural Networks & Deep Learning)
# ============================================================

import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import sys
import json
from typing import Dict, List, Tuple, Any, Optional

# Add parent directory to path for data module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


class NeuralDiagnosticModel:
    """
    Deep Neural Network for medical diagnosis.
    Multi-Layer Perceptron with batch normalization, dropout,
    and advanced regularization techniques.
    
    Loads data from CSV files in the data/ directory.
    
    Architecture:
    Input (22) → Dense(256) → BN → Dropout(0.3) 
              → Dense(128) → BN → Dropout(0.25)
              → Dense(64) → BN → Dropout(0.2)
              → Dense(32) → BN → Dropout(0.15)
              → Dense(13) → Softmax  # ← Updated to 13 diseases
    """

    SYMPTOM_FEATURES = [
        'fever', 'cough', 'fatigue', 'headache',
        'body_aches', 'loss_of_smell', 'chest_pain',
        'rash', 'joint_pain', 'shortness_of_breath',
        'sweating', 'frequent_urination', 'excessive_thirst',
        'blurred_vision', 'night_sweats', 'weight_loss',
        'stiff_neck', 'light_sensitivity', 'sore_throat',
        'runny_nose', 'sneezing', 'nausea'
    ]

    # Updated to include all 13 diseases from the data
    DISEASE_LABELS = [
        'flu', 'covid19', 'dengue', 'cardiac_event',
        'diabetes', 'common_cold', 'tuberculosis', 'meningitis',
        'pneumonia', 'bronchitis', 'allergies', 
        'gastroenteritis', 'strep_throat'
    ]

    def __init__(self, data_dir: str = "data", model_path: Optional[str] = None):
        """
        Initialize the neural network model.
        
        Args:
            data_dir: Path to the data directory containing CSV files
            model_path: Path to load a pre-trained model (optional)
        """
        self.data_dir = data_dir
        self.num_features = len(self.SYMPTOM_FEATURES)
        self.num_classes = len(self.DISEASE_LABELS)
        
        self.label_encoder = LabelEncoder()
        # Pre-fit the label encoder
        self.label_encoder.fit(self.DISEASE_LABELS)
        
        self.scaler = StandardScaler()
        self.model = None
        self.history = None
        self.is_trained = False
        self.best_val_accuracy = 0.0
        
        # Data
        self.X = None
        self.y = None
        self.df = None
        
        self._build_model()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        print(f"[NeuralNet] Initialized with {self.num_features} features, {self.num_classes} classes")
        print(f"[NeuralNet] Data directory: {self.data_dir}")
        print(f"[NeuralNet] Disease labels: {self.DISEASE_LABELS}")

    def _load_data(self) -> pd.DataFrame:
        """
        Load patient data from CSV file.
        
        Returns:
            DataFrame with patient records
        """
        # Try to load from data directory
        csv_path = os.path.join(self.data_dir, "patient_records.csv")
        
        if not os.path.exists(csv_path):
            print(f"[NeuralNet] Data file not found: {csv_path}")
            print("[NeuralNet] Attempting to load from parent directory...")
            csv_path = "../data/patient_records.csv"
            
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"Patient records file not found. Expected at: {self.data_dir}/patient_records.csv"
                )
        
        df = pd.read_csv(csv_path)
        print(f"[NeuralNet] Loaded {len(df)} patient records from {csv_path}")
        
        # Check if we have the expected columns
        missing_features = [f for f in self.SYMPTOM_FEATURES if f not in df.columns]
        if missing_features:
            print(f"[NeuralNet] Warning: Missing symptom columns: {missing_features}")
            # Add missing columns with default value 0
            for feature in missing_features:
                df[feature] = 0
        
        return df

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with patient records
            
        Returns:
            Tuple of (features, labels)
        """
        # Extract features - symptoms columns
        X = df[self.SYMPTOM_FEATURES].values.astype(np.float32)
        
        # Extract labels - disease column
        y = df['disease'].values
        
        # Print unique diseases for debugging
        unique_diseases = np.unique(y)
        print(f"[NeuralNet] Prepared {len(X)} samples with {len(self.SYMPTOM_FEATURES)} features")
        print(f"[NeuralNet] Unique diseases ({len(unique_diseases)}): {unique_diseases}")
        
        return X, y

    def _build_model(self):
        """Build the deep MLP architecture"""
        self.model = models.Sequential([
            layers.Input(shape=(self.num_features,)),
            
            layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            layers.Dense(64, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.15),
            
            layers.Dense(self.num_classes, activation='softmax')
        ], name='MedicalDNN_Enhanced')

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, epochs: int = 50, verbose: int = 1, validation_split: float = 0.2) -> Dict:
        """
        Train the neural network using data from CSV files.
        
        Args:
            epochs: Number of training epochs
            verbose: Verbosity level
            validation_split: Proportion of data for validation
            
        Returns:
            Dictionary with training history
        """
        print("\n" + "=" * 55)
        print("  Neural Network — Medical Diagnosis Training")
        print(f"  Architecture: {self.num_features} → 256 → 128 → 64 → 32 → {self.num_classes}")
        print("=" * 55)
        self.model.summary()

        # Load data from CSV
        self.df = self._load_data()
        
        # Prepare data
        X, y = self._prepare_data(self.df)
        
        # Encode labels - fit on the actual data
        y_encoded = self.label_encoder.fit_transform(y)
        self.X = X
        self.y = y_encoded
        
        # Verify label range
        unique_encoded = np.unique(y_encoded)
        print(f"[NeuralNet] Encoded labels range: {unique_encoded}")
        print(f"[NeuralNet] Expected range: 0 to {self.num_classes - 1}")
        
        if max(unique_encoded) >= self.num_classes:
            print(f"[NeuralNet] ERROR: Encoded label {max(unique_encoded)} >= num_classes {self.num_classes}")
            print(f"[NeuralNet] Unique diseases in data: {np.unique(y)}")
            print(f"[NeuralNet] Disease labels in model: {self.DISEASE_LABELS}")
            raise ValueError(f"Label encoder produced value {max(unique_encoded)} which is >= {self.num_classes}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=validation_split, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        print(f"[NeuralNet] Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                'best_neural_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=0
            )
        ]

        # Train the model
        self.history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=64,
            callbacks=callbacks_list,
            verbose=verbose
        )

        self.is_trained = True
        self.best_val_accuracy = max(self.history.history['val_accuracy'])
        print(f"\n✅ Best Validation Accuracy: {self.best_val_accuracy:.4f}")
        
        return {
            'history': self.history.history,
            'best_val_accuracy': self.best_val_accuracy,
            'epochs_trained': len(self.history.history['loss'])
        }

    def _safe_label_conversion(self, label) -> str:
        """Safely convert a label to a string, handling numpy types."""
        if hasattr(label, 'item'):
            label = label.item()
        if hasattr(label, 'decode'):
            label = label.decode('utf-8')
        if not isinstance(label, str):
            label = str(label)
        return label

    def predict(self, symptoms: List[str]) -> Dict:
        """Predict disease from symptom list"""
        if not self.is_trained:
            print("[NeuralNet] Model not trained. Training with default parameters...")
            self.train(epochs=30, verbose=0)

        features = self._symptoms_to_vector(symptoms)
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        predictions = self.model.predict(features_scaled, verbose=0)[0]
        
        pred_index = np.argmax(predictions)
        
        diagnosis_raw = self.label_encoder.inverse_transform([pred_index])[0]
        diagnosis = self._safe_label_conversion(diagnosis_raw)
        confidence = float(predictions[pred_index])

        # Get top 5 predictions
        top_5_indices = np.argsort(predictions)[::-1][:5]
        top_5 = []
        for i in top_5_indices:
            label_raw = self.label_encoder.inverse_transform([i])[0]
            label = self._safe_label_conversion(label_raw)
            top_5.append((label, float(predictions[i])))

        # Get all probabilities
        all_probs = {}
        for i in range(len(predictions)):
            label_raw = self.label_encoder.inverse_transform([i])[0]
            label = self._safe_label_conversion(label_raw)
            all_probs[label] = float(predictions[i])

        disease_info = self._get_disease_info(diagnosis)

        return {
            'diagnosis': diagnosis,
            'confidence': confidence,
            'top_5': top_5,
            'model_used': 'NeuralNetwork',
            'disease_info': disease_info,
            'all_probabilities': all_probs
        }

    def _symptoms_to_vector(self, symptoms: List[str]) -> np.ndarray:
        """Convert symptoms to binary feature vector"""
        normalized = [s.lower().strip().replace(' ', '_') for s in symptoms]
        
        vector = np.zeros(self.num_features, dtype=np.float32)
        for i, feature in enumerate(self.SYMPTOM_FEATURES):
            if feature in normalized:
                vector[i] = 1.0
        
        return vector

    def _get_disease_info(self, disease: str) -> Dict:
        """Get disease information"""
        disease_info = {
            'flu': {'severity': 'MODERATE', 'treatment': 'Rest, hydration, antiviral medication', 'common_symptoms': 'Fever, cough, fatigue, body aches', 'urgency': 'MEDIUM'},
            'covid19': {'severity': 'MODERATE', 'treatment': 'Isolation, supportive care, antiviral if severe', 'common_symptoms': 'Fever, cough, loss of smell, fatigue', 'urgency': 'HIGH'},
            'cardiac_event': {'severity': 'CRITICAL', 'treatment': 'Emergency care, aspirin, hospitalization', 'common_symptoms': 'Chest pain, shortness of breath, sweating', 'urgency': 'CRITICAL'},
            'diabetes': {'severity': 'MODERATE', 'treatment': 'Blood sugar monitoring, medication, diet', 'common_symptoms': 'Frequent urination, excessive thirst, fatigue', 'urgency': 'MEDIUM'},
            'dengue': {'severity': 'MODERATE', 'treatment': 'Hydration, pain relief, monitor for warning signs', 'common_symptoms': 'High fever, joint pain, rash, headache', 'urgency': 'HIGH'},
            'tuberculosis': {'severity': 'SEVERE', 'treatment': 'Antibiotics, isolation, regular monitoring', 'common_symptoms': 'Cough, weight loss, night sweats, fatigue', 'urgency': 'HIGH'},
            'meningitis': {'severity': 'CRITICAL', 'treatment': 'Emergency care, antibiotics, hospitalization', 'common_symptoms': 'Headache, stiff neck, high fever, light sensitivity', 'urgency': 'CRITICAL'},
            'pneumonia': {'severity': 'SEVERE', 'treatment': 'Antibiotics, rest, oxygen therapy if needed', 'common_symptoms': 'Fever, cough, shortness of breath, chest pain', 'urgency': 'HIGH'},
            'bronchitis': {'severity': 'MODERATE', 'treatment': 'Rest, hydration, cough medication', 'common_symptoms': 'Cough, mucus production, fatigue', 'urgency': 'MEDIUM'},
            'allergies': {'severity': 'MILD', 'treatment': 'Antihistamines, avoid allergens', 'common_symptoms': 'Runny nose, sneezing, itchy eyes', 'urgency': 'LOW'},
            'common_cold': {'severity': 'MILD', 'treatment': 'Rest, hydration, over-the-counter medication', 'common_symptoms': 'Runny nose, sneezing, sore throat, cough', 'urgency': 'LOW'},
            'gastroenteritis': {'severity': 'MODERATE', 'treatment': 'Hydration, rest, dietary management', 'common_symptoms': 'Nausea, vomiting, diarrhea, abdominal pain', 'urgency': 'MEDIUM'},
            'strep_throat': {'severity': 'MODERATE', 'treatment': 'Antibiotics, rest, pain relief', 'common_symptoms': 'Sore throat, fever, swollen lymph nodes', 'urgency': 'MEDIUM'}
        }
        return disease_info.get(disease, {'severity': 'UNKNOWN', 'treatment': 'Consult a healthcare provider', 'common_symptoms': 'Unknown', 'urgency': 'MEDIUM'})

    def analyze(self, percept) -> Dict:
        """Standard interface method for the Agent to call"""
        result = self.predict(percept.symptoms)
        return {
            'diagnosis': result['diagnosis'],
            'confidence': result['confidence'],
            'details': {
                'top_5': result['top_5'],
                'disease_info': result.get('disease_info', {}),
                'model_used': 'NeuralNetwork'
            }
        }

    def plot_training(self, save_path: Optional[str] = None):
        """Plot training history"""
        if not self.is_trained or self.history is None:
            print("[NeuralNet] Model not trained. Please call train() first.")
            return

        history = self.history.history
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1 = axes[0]
        ax1.plot(history['accuracy'], label='Training Accuracy', color='#3498db', linewidth=2)
        ax1.plot(history['val_accuracy'], label='Validation Accuracy', color='#e74c3c', linewidth=2, linestyle='--')
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        best_acc = max(history['val_accuracy'])
        best_idx = history['val_accuracy'].index(best_acc)
        ax1.annotate(f'Best: {best_acc:.3f}', 
                    xy=(best_idx, best_acc),
                    xytext=(best_idx + 2, best_acc - 0.05),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontweight='bold')
        
        ax2 = axes[1]
        ax2.plot(history['loss'], label='Training Loss', color='#2ecc71', linewidth=2)
        ax2.plot(history['val_loss'], label='Validation Loss', color='#e67e22', linewidth=2, linestyle='--')
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        best_loss = min(history['val_loss'])
        best_idx_loss = history['val_loss'].index(best_loss)
        ax2.annotate(f'Best: {best_loss:.3f}', 
                    xy=(best_idx_loss, best_loss),
                    xytext=(best_idx_loss + 2, best_loss + 0.5),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontweight='bold')

        plt.suptitle('Neural Network Training Curves', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✅ Saved: {save_path}")
        
        plt.show()

    def save_model(self, filepath: str = 'neural_model.keras'):
        """Save the trained model"""
        if not self.is_trained:
            print("[NeuralNet] Model not trained. Cannot save.")
            return
        
        self.model.save(filepath)
        
        metadata = {
            'num_features': self.num_features,
            'num_classes': self.num_classes,
            'best_val_accuracy': self.best_val_accuracy,
            'disease_labels': self.DISEASE_LABELS,
            'symptom_features': self.SYMPTOM_FEATURES
        }
        with open(filepath.replace('.keras', '_metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        print(f"\n✅ Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model"""
        try:
            self.model = models.load_model(filepath)
            self.is_trained = True
            
            metadata_path = filepath.replace('.keras', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.best_val_accuracy = metadata.get('best_val_accuracy', 0.0)
            
            print(f"\n✅ Model loaded from {filepath}")
            print(f"   Best Validation Accuracy: {self.best_val_accuracy:.4f}")
            
        except Exception as e:
            print(f"\n❌ Error loading model: {e}")
            self.is_trained = False


# ── Test the Neural Network ──────────────────────────────
def test_neural_network():
    """Test the NeuralDiagnosticModel module"""
    print("\n" + "=" * 60)
    print("  TESTING NEURAL NETWORK")
    print("=" * 60)
    
    nn = NeuralDiagnosticModel()
    
    # Check data
    import pandas as pd
    try:
        df = pd.read_csv("data/patient_records.csv")
        unique_diseases = df['disease'].unique()
        print(f"\n  Data contains {len(unique_diseases)} diseases: {unique_diseases}")
    except:
        pass
    
    nn.train(epochs=30, verbose=1)
    
    print("\n--- Predictions ---")
    test_cases = [
        ["fever", "cough", "loss_of_smell", "fatigue"],
        ["fever", "cough", "body_aches", "headache"],
        ["runny_nose", "sneezing", "sore_throat"],
        ["high_fever", "joint_pain", "rash"],
    ]
    
    for symptoms in test_cases:
        result = nn.predict(symptoms)
        print(f"\n  Symptoms: {symptoms}")
        print(f"    Diagnosis: {result['diagnosis']}")
        print(f"    Confidence: {result['confidence']:.2%}")
        print(f"    Top 5: {result['top_5'][:3]}")
    
    print("\n✅ Neural Network test passed!")


if __name__ == "__main__":
    test_neural_network()