
# 🏥 Intelligent Healthcare Diagnostic Assistant

*An AI-powered healthcare diagnostic system that integrates multiple AI techniques including intelligent agents, logical inference, probabilistic reasoning, machine learning, deep learning, fuzzy logic, and AI planning.*

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [AI Modules](#-ai-modules)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Evaluation](#-evaluation)
- [Data](#-data)
- [Testing](#-testing)
- [Requirements](#-requirements)
- [Team](#-team)
- [License](#-contact--details)

## 📖 Overview

This project implements an **Intelligent Healthcare Diagnostic Assistant** as a capstone project for the Introduction to Artificial Intelligence course at **Dedan Kimathi University of Technology**. The system integrates multiple AI paradigms to create a comprehensive medical diagnosis and treatment recommendation platform.

### 🎯 Mission

Build an end-to-end AI system that integrates intelligent agents, search, probabilistic reasoning, machine learning, NLP, fuzzy logic, and planning into a unified healthcare diagnostic and recommendation platform.

### ⚡ Key Capabilities

| Capability               | Description                                                 |
| ------------------------ | ----------------------------------------------------------- |
| Patient Symptom Analysis | Process patient symptoms and vitals                         |
| Multi-Model Diagnosis    | Combine multiple AI approaches for robust diagnosis         |
| Severity Assessment      | Evaluate urgency using fuzzy logic                          |
| Treatment Planning       | Generate step-by-step treatment plans using STRIPS planning |
| Performance Evaluation   | Comprehensive metrics and visualizations                    |

## ✨ Features

| Feature           | Description                                                            |
| ----------------- | ---------------------------------------------------------------------- |
| Intelligent Agent | Model-Based, Goal-Based agent coordinating all modules                 |
| Knowledge Base    | First-Order Logic with forward/backward chaining and certainty factors |
| Bayesian Network  | Probabilistic reasoning using Naive Bayes with log-space calculations  |
| ML Classifier     | Ensemble of Decision Tree, Random Forest, and Gradient Boosting        |
| Neural Network    | Deep Learning with TensorFlow/Keras (MLP with BatchNorm and Dropout)   |
| Fuzzy Logic       | Severity assessment using fuzzy inference                              |
| AI Planning       | STRIPS-based treatment plan generation                                 |
| Data Management   | Synthetic data generation and CSV-based data loading                   |

## 🧠 AI Modules

### 1. Intelligent Agent (`modules/agent.py`)

**PEAS Framework:**

- **Performance:** Diagnostic accuracy, patient safety, response time
- **Environment:** Patient symptoms, vitals, medical history
- **Actuators:** Diagnosis report, treatment plan, urgent alert
- **Sensors:** Symptom text input, temperature reading, heart rate

**Agent Type:** Model-Based + Goal-Based + Learning

**State Machine:**

```
IDLE → COLLECTING → DIAGNOSING → RECOMMENDING → PLANNING → DONE
```

**Key Methods:**

- `perceive()` - Store patient data and update agent state
- `think()` - Call all registered modules for analysis
- `act()` - Combine results, determine urgency, generate recommendations
- `run()` - Full agent cycle: Perceive → Think → Act

### 2. Knowledge Base (`modules/knowledge_base.py`)

**First-Order Logic** rules with certainty factors:

- **Forward Chaining** - Data-driven inference from symptoms to diagnosis
- **Backward Chaining** - Goal-driven inference to prove a diagnosis
- **Certainty Factors** - Confidence values for each rule (0.0 to 1.0)
- **Cycle Detection** - Prevents infinite loops in inference

**Example Rule:**

```
IF fever AND cough AND loss_of_smell AND fatigue → covid19_suspected (CF = 0.85)
```

**Supported Diseases:** 13 diseases with comprehensive symptom rules

### 3. Bayesian Network (`modules/bayesian_net.py`)

**Naive Bayes** classifier for probabilistic diagnosis.

**Core Formula:**

```
P(Disease | Symptoms) ∝ P(Disease) × P(Symptom₁|Disease) × P(Symptom₂|Disease) × ... × P(Symptomₙ|Disease)
```

- **Log-space calculations** - To prevent underflow
- **Prior probabilities** - Base rates in the population
- **Likelihood tables** - P(symptom|disease) for all symptom-disease pairs
- **Symptom impact analysis** - Likelihood ratio calculation
- **Odds ratio** - For clinical decision support

### 4. ML Classifier (`modules/ml_classifier.py`)

| Model             | Description                                |
| ----------------- | ------------------------------------------ |
| Decision Tree     | Entropy-based splitting with max_depth=8   |
| Random Forest     | 150 trees, max_depth=10, ensemble voting   |
| Gradient Boosting | 150 estimators, learning_rate=0.1          |
| Voting Classifier | Soft voting ensemble for improved accuracy |

**Key Features:**

- Cross-Validation with 5 folds (StratifiedKFold)
- Feature Importance analysis with visualization
- Confusion Matrices for all models
- Classification Report with precision, recall, f1-score

### 5. Neural Network (`modules/neural_network.py`)

**Deep MLP Architecture:**

```
Input (22) → Dense(256) → BN → Dropout(0.3)
           → Dense(128) → BN → Dropout(0.25)
           → Dense(64)  → BN → Dropout(0.2)
           → Dense(32)  → BN → Dropout(0.15)
           → Dense(13)  → Softmax
```

**Regularization Techniques:**

- **Batch Normalization** - Stabilizes training
- **Dropout** - Prevents overfitting
- **L2 Regularization** - Weight decay (0.001)
- **Early Stopping** - Patience 15
- **Learning Rate Reduction** - Factor 0.5, patience 8

### 6. Fuzzy Controller (`modules/fuzzy_controller.py`)

**Fuzzy Logic** severity assessment.

Inputs: Temperature (Celsius), Heart Rate (BPM), Symptom Count
Output: Severity Score (0-100) and Label

| Variable    | Fuzzy Sets                            |
| ----------- | ------------------------------------- |
| Temperature | Normal, Mild, High, Critical          |
| Heart Rate  | Low, Normal, Elevated, High, Critical |
| Symptoms    | Few, Several, Many, Extensive         |
| Severity    | Low, Mild, Moderate, High, Critical   |

**20 Fuzzy Rules with centroid defuzzification.**

### 7. Treatment Planner (`modules/planner.py`)

**STRIPS Planning** with BFS search.

**21 Medical Actions across 5 categories:**

- **Emergency** - CallEmergencyServices, TransferToICU, StartEmergencyOxygen
- **Diagnostic** - OrderBloodPanel, OrderPCRTest, OrderChestXray, ClinicalDiagnosis
- **Treatment** - PrescribeAntiviral, PrescribeAntibiotics, AdministerFluids
- **Monitoring** - MonitorVitals, MonitorBloodSugar
- **Follow-up** - ScheduleFollowUp, DischargePatient

## 📁 Project Structure

```
ai-capstone-healthcare/
├── modules/
│   ├── __init__.py
│   ├── agent.py                # Intelligent Agent
│   ├── knowledge_base.py       # FOL Knowledge Base
│   ├── bayesian_net.py         # Bayesian Network
│   ├── ml_classifier.py        # ML Classifier
│   ├── neural_network.py       # Neural Network
│   ├── fuzzy_controller.py     # Fuzzy Logic
│   └── planner.py              # Treatment Planner
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py              # Performance Metrics
│   ├── visualizations.py       # Charts and Plots
│   └── reports.py              # Report Generation
│
├── data/
│   ├── __init__.py
│   ├── generator.py            # Data Generator
│   ├── loader.py                # Data Loader
│   ├── symptoms.csv             # 30 Symptoms
│   ├── diseases.csv             # 13 Diseases
│   ├── disease_symptom_matrix.csv  # 89 Mappings
│   └── patient_records.csv      # 2,000+ Patient Records
│
├── evaluation_output/            # Generated evaluation artifacts
│   ├── all_confusion_matrices.png
│   ├── diagnosis_distribution.png
│   ├── module_comparison.png
│   ├── confidence_distribution.png
│   ├── metrics_report.json
│   ├── metrics_report.txt
│   └── patient_results.csv
│
├── app.py                       # Main Application
├── requirements.txt             # Dependencies
└── README.md                    # This File
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/1-njohi/ai-capstone-healthcare.git
cd ai-capstone-healthcare
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment

```bash
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Generate Data

```bash
python data/generator.py
```

### Step 6: Run the Application

```bash
python app.py
```

## 💻 Usage

### Run the Application

```bash
python app.py
```

### Main Menu Options

| Option                | Description                             |
| --------------------- | --------------------------------------- |
| 1. Run Test Patients  | Run 9 predefined test cases             |
| 2. Detailed Diagnosis | Single patient with detailed report     |
| 3. Interactive Mode   | Enter custom patient data               |
| 4. System Info        | View registered modules and performance |
| 5. Run Evaluation     | Generate metrics and visualizations     |
| 6. Exit               | Exit the application                    |

### Example Interactive Session

```
Patient ID: P001
Symptoms: fever,cough,loss_of_smell,fatigue
Age: 34
Temperature (°C): 38.5
Heart Rate (BPM): 98
Blood Pressure: 120/80

📋 DIAGNOSIS REPORT
════════════════════════════════════════════════════════════
Patient: P001
Diagnosis: covid19
Confidence: 72.3%
Urgency: HIGH

Recommendations:
1. ⚠️ URGENT: See a doctor within 24 hours
2. 🧪 Get PCR test for confirmation
3. 🛑 Isolate from others
4. 📊 Monitor oxygen levels with pulse oximeter
5. 🏥 Seek immediate care if breathing becomes difficult
```

## 📊 Evaluation

### Metrics Generated

| Metric           | Formula                       | Description                              |
| ---------------- | ----------------------------- | ---------------------------------------- |
| Accuracy         | Correct / Total               | Overall correctness                      |
| Precision        | TP / (TP + FP)                | How often positive predictions are right |
| Recall           | TP / (TP + FN)                | How often actual positives are caught    |
| F1-Score         | 2 × (P × R) / (P + R)       | Balance of precision and recall          |
| Confusion Matrix | Grid of predictions vs actual | Which diseases get confused              |
| ECE              | Expected Calibration Error    | Confidence calibration quality           |

### Visualization Outputs

| File                        | Description                           |
| --------------------------- | ------------------------------------- |
| all_confusion_matrices.png  | Confusion matrices for all ML models  |
| diagnosis_distribution.png  | Distribution of diagnoses             |
| urgency_distribution.png    | Distribution of urgency levels        |
| module_comparison.png       | Performance comparison across modules |
| confidence_distribution.png | Confidence score distribution         |

## 📂 Data

### Data Files

| File                       | Records | Description                                              |
| -------------------------- | ------- | -------------------------------------------------------- |
| symptoms.csv               | 30      | Symptoms with categories and severity weights            |
| diseases.csv               | 13      | Diseases with metadata (severity, contagious, treatable) |
| disease_symptom_matrix.csv | 89      | Probability mappings between diseases and symptoms       |
| patient_records.csv        | 2,000+  | Synthetic patient records with binary symptom columns    |

### Load Data Programmatically

```python
from data.loader import DataLoader

loader = DataLoader()

# Get all symptoms
symptoms = loader.get_symptom_list()

# Get all diseases
diseases = loader.get_disease_list()

# Get all patients
patients = loader.get_patients()

# Get symptom probabilities for a disease
probs = loader.get_symptom_probabilities('covid19')

# Get patients by disease
flu_patients = loader.get_patients_by_disease('flu')

# Get a specific patient
patient = loader.get_patient_by_id('P0001')
```

## 🧪 Testing

### Test Individual Modules

```bash
# Test Knowledge Base
python -c "from modules.knowledge_base import test_knowledge_base; test_knowledge_base()"

# Test Bayesian Network
python -c "from modules.bayesian_net import test_bayesian_net; test_bayesian_net()"

# Test ML Classifier
python -c "from modules.ml_classifier import test_ml_classifier; test_ml_classifier()"

# Test Neural Network
python -c "from modules.neural_network import test_neural_network; test_neural_network()"

# Test Fuzzy Controller
python -c "from modules.fuzzy_controller import test_fuzzy_controller; test_fuzzy_controller()"

# Test Treatment Planner
python -c "from modules.planner import test_planner; test_planner()"
```

## 📝 Requirements

```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
tensorflow>=2.13.0
pgmpy>=0.1.23
scikit-fuzzy>=0.4.2
networkx>=3.1
scipy>=1.11.0
```

## 👥 Team

| Role      | Reg Number        | Name          | Module                                                  |
| --------- | ----------------- | ------------- | ------------------------------------------------------- |
| Team Lead | C026-01-2472/2024 | Denis Wanjohi | Agent, Integration, Knowledge Base & Bayesian Network |
| Member 2  |                   | Wayne Omumia  | ML Classifier & Neural Network                          |
| Member 3  |                   | Ryan         | Fuzzy Logic & Planner                                   |

## 📞 Contact & Details

| Contact    | Details                                                    |
| ---------- | ---------------------------------------------------------- |
| University | Dedan Kimathi University of Technology                     |
| Course     | CCS 3101 - Introduction to Artificial Intelligence         |
| Semester   | 13-Week Capstone Project                                   |
| License    | Academic Use Only - Dedan Kimathi University of Technology |
