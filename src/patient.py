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

    def generate_vital_signs(self):
        """
        Generates physical symptoms (vital signs) adjusted for the patient's age group.
        Uses Normal (Gaussian) distributions to represent human biometrics.
        """
        # 1. Temperature (°C)
        # Justification: Core temperature is relatively stable across all ages.
        # Mean 37.0, SD 0.8 allows for standard variance and fever spikes.
        self.vital_signs["temperature"] = np.random.normal(loc=37.0, scale=0.8)

        # 2. Heart Rate (BPM) & 3. Systolic Blood Pressure (mmHg)
        if self.age_group == "0-18":
            # Justification: Children have naturally faster resting heart rates and
            # naturally lower blood pressure than adults.
            self.vital_signs["heart_rate"] = np.random.normal(loc=95.0, scale=15.0)
            self.vital_signs["systolic_bp"] = np.random.normal(loc=105.0, scale=15.0)

        elif self.age_group in ["19-35", "36-55"]:
            # Justification: Standard healthy adult baselines.
            self.vital_signs["heart_rate"] = np.random.normal(loc=80.0, scale=15.0)
            self.vital_signs["systolic_bp"] = np.random.normal(loc=120.0, scale=15.0)

        else:  # "56-75" and "76+"
            # Justification: Older patients have a much higher statistical probability
            # of hypertension (high BP) and a wider variance in heart health.
            self.vital_signs["heart_rate"] = np.random.normal(loc=75.0, scale=20.0)
            self.vital_signs["systolic_bp"] = np.random.normal(loc=135.0, scale=25.0)

    def calculate_triage_scores(self):
        """
        Computes the severity score, priority level, and evaluation time
        based on the patient's clinical and demographic data.
        """
        # 1. Base Disease Severity
        base_sev = DISEASES[self.disease]["base_severity"]

        # 2. Subtype Severity Modifier (Addressing the nuance!)
        # Justification: A 'Mild' or 'Early' subtype should reduce the overall severity,
        # while 'Severe' or 'Compound' subtypes require a higher urgency bump.
        subtype_modifiers = {
            "Mild": -1, "Hairline Fracture": -1, "Early Stage": -1, "Non-displaced": -1,
            "Moderate": 0, "Simple Fracture": 0, "Displaced": 0, "Acute": 0,
            "Severe": 2, "Compound Fracture": 2, "Comminuted": 2, "Perforated": 2
        }
        # Fetch the modifier for the patient's specific subtype (defaults to 0 if not found)
        subtype_sev = subtype_modifiers.get(self.subtype, 0)

        # 3. Vital Sign Penalty
        vital_penalty = 0
        if self.vital_signs["temperature"] > 38.0 or self.vital_signs["temperature"] < 36.0:
            vital_penalty += 1  # Fever or hypothermia

        if self.vital_signs["systolic_bp"] > 140.0 or self.vital_signs["systolic_bp"] < 90.0:
            vital_penalty += 1  # Hyper/hypotension

        # 4. Age-Related Risk Factor
        age_risk = 1 if self.age_group in ["0-18", "76+"] else 0

        # --- Final Severity Score Calculation ---
        self.severity_score = base_sev + subtype_sev + vital_penalty + age_risk

        # --- Priority Level Calculation ---
        # 60% importance to the immediate clinical severity, 40% to the demographic age-priority.
        age_prio = AGE_GROUPS[self.age_group]["priority"]
        self.priority_level = (self.severity_score * 0.6) + (age_prio * 0.4)

        # --- Triage Time Generation ---
        # Standard triage evaluation: 5 to 15 minutes (converted to hours).
        self.triage_time = np.random.uniform(5 / 60, 15 / 60)


def generate_interarrival_time():
    """
    Generates the time (in hours) until the next patient arrives.
    Based on the baseline requirement: uniformly distributed between 1 and 3 hours.
    """
    return np.random.uniform(1.0, 3.0)
