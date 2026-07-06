import numpy as np

# Age groups: (probability, priority, factor)
AGE_GROUPS = {
    "0-18": {"probability": 0.15, "priority": 3, "factor": "pediatric"},
    "19-35": {"probability": 0.25, "priority": 2, "factor": "young_adult"},
    "36-55": {"probability": 0.35, "priority": 1, "factor": "adult"},
    "56-75": {"probability": 0.15, "priority": 4, "factor": "elderly, comorbidities"},
    "76+": {"probability": 0.10, "priority": 5, "factor": "geriatric, high_risk"}
}

# Diseases: Base severity, Treatment range (hours), Subtypes
DISEASES = {
    "Broken Arm": {
        "base_severity": 2,
        "treatment_range": (12, 24),
        "subtypes": ["Hairline Fracture", "Simple Fracture", "Compound Fracture"]
    },
    "Concussion": {
        "base_severity": 4,
        "treatment_range": (18, 24),
        "subtypes": ["Mild", "Moderate", "Severe"]
    },
    "Simple Fracture": {
        "base_severity": 3,
        "treatment_range": (12, 18),
        "subtypes": ["Non-displaced", "Displaced", "Comminuted"]
    },
    "Appendicitis": {
        "base_severity": 5,
        "treatment_range": (12, 24),
        "subtypes": ["Early Stage", "Acute", "Perforated"]
    },
    "Pneumonia": {
        "base_severity": 4,
        "treatment_range": (18, 24),
        "subtypes": ["Mild", "Moderate", "Severe"]
    }
}


class Patient:
    """
    A blueprint for every patient entering the simulation.
    Holds all demographic, clinical, and temporal data required for the final outputs.
    """
    def __init__(self, patient_id, arrival_time):
        # --- Identifiers and Arrival ---
        self.patient_id = patient_id
        self.arrival_time = arrival_time

        # --- Demographics & Clinical Attributes ---
        self.age_group = None
        self.disease = None
        self.subtype = None
        self.vital_signs = {
            "temperature": None,
            "heart_rate": None,
            "systolic_bp": None
        }

        # --- Triage Calculations ---
        self.severity_score = None
        self.priority_level = None
        self.triage_time = None

        # --- Routing Outcomes & Timestamps ---
        self.status = "Generated"  # E.g., 'Admitted', 'Queued', 'Rejected'
        self.bed_number = None
        self.queue_entry_time = None
        self.admission_time = None
        self.discharge_time = None
        self.wait_to_bed = None
        self.total_time_in_system = None
        self.rejection_reason = None

    def assign_demographic_and_illness(self):
        """
        Randomly assigns age group, disease, and subtype to the patient
        based on the global configuration constants.
        """
        # 1. Assign Age Group (Weighted Probabilities)
        age_categories = list(AGE_GROUPS.keys())
        age_probabilities = [AGE_GROUPS[age]["probability"] for age in age_categories]
        self.age_group = np.random.choice(age_categories, p=age_probabilities)

        # 2. Assign Disease (Equal Probabilities)
        disease_categories = list(DISEASES.keys())
        self.disease = np.random.choice(disease_categories)

        # 3. Assign Subtype (Equal Probabilities)
        subtypes = DISEASES[self.disease]["subtypes"]
        self.subtype = np.random.choice(subtypes)


def generate_interarrival_time():
    """
    Generates the time (in hours) until the next patient arrives.
    Based on the baseline requirement: uniformly distributed between 1 and 3 hours.
    """
    return np.random.uniform(1.0, 3.0)
