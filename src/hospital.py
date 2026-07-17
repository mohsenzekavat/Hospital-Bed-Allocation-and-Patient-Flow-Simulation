import numpy as np
from patient import AGE_GROUPS, DISEASES


class Hospital:
    """
    Manages the global state of the hospital simulation, including the clock,
    resources (beds), and the waiting queue.
    """
    def __init__(self, bed_capacity=10, queue_capacity=5):
        # --- Simulation Clock & Horizon ---
        self.clock = 0.0
        # The simulation horizon is 30 days, measured in hours (30 * 24 = 720 hours)
        self.horizon = 720.0

        # --- Capacity Constraints ---
        # The hospital now uses whatever numbers are passed in, defaulting to Baseline!
        self.bed_capacity = bed_capacity
        self.queue_capacity = queue_capacity

        # We build the exact right number of empty beds immediately based on the parameter
        self.beds = [None] * self.bed_capacity

        self.queue = []

        # --- Data Collection Tracking ---
        self.patient_log = []
        self.hourly_log = []

    def handle_patient_arrival(self, patient):
        """
        Handles the routing of a patient immediately after their triage evaluation.
        Evaluates bed availability first, then queue availability, and rejects if both are full.
        """
        # Check for an available bed
        if None in self.beds:
            # Find the index of the first available bed
            bed_index = self.beds.index(None)

            # Admit the patient to this bed
            self.beds[bed_index] = patient
            patient.bed_number = bed_index
            patient.status = "Admitted"

            # Record timestamps for the patient log
            patient.admission_time = self.clock
            patient.wait_to_bed = 0.0  # They got a bed immediately

            treatment_time = self.generate_treatment_duration(patient)
            patient.discharge_time = self.clock + treatment_time

        # If beds are full, check if there is room in the queue
        elif len(self.queue) < self.queue_capacity:
            # Add the patient to the back of the queue (appending to the list)
            self.queue.append(patient)
            patient.status = "Queued"

            # Record timestamp for when they entered the queue
            patient.queue_entry_time = self.clock

        # If both beds and the queue are full, reject the patient
        else:
            patient.status = "Rejected"
            # Explicitly log the required rejection reason
            patient.rejection_reason = "queue_full"

            # The patient leaves immediately, so their total time is just their triage time
            patient.total_time_in_system = self.clock - patient.arrival_time

    def check_queue_timeouts(self):
        """
        Scans the queue to enforce the First-In-First-Out (FIFO) discipline
        and removes patients who have exceeded their medically safe waiting time.
        Timeouts are rounded to strict hourly increments.
        """
        surviving_queue = []

        for patient in self.queue:
            # 1. Safe Waiting Math (Strict Hourly)
            # Calculate the raw inverse priority
            raw_safe_wait = 12.0 / max(1.0, patient.priority_level)

            # Round to the nearest whole hour (e.g., 2.4 becomes 2.0, 1.6 becomes 2.0)
            # We also use max(1.0, ...) to guarantee even the most critical patient gets at least 1 hour
            safe_wait_time = max(1.0, round(raw_safe_wait))

            time_waited = self.clock - patient.queue_entry_time

            # 2. Timeout Enforcement
            if time_waited >= safe_wait_time:
                patient.status = "Rejected"
                patient.rejection_reason = "timeout"
                patient.total_time_in_system = self.clock - patient.arrival_time

            else:
                surviving_queue.append(patient)

        # 3. FIFO Logic
        self.queue = surviving_queue

    @staticmethod
    def generate_treatment_duration(patient):
        """
        Calculates bed occupation time by mapping the severity score directly
        to the disease's treatment range, adding small random noise.
        """
        min_hours, max_hours = DISEASES[patient.disease]["treatment_range"]

        # 1. Normalize the Severity (Assume a max realistic severity of 10)
        # This converts the severity into a percentage (e.g., a score of 5 becomes 0.50)
        effective_severity = min(patient.severity_score, 10.0)
        severity_ratio = effective_severity / 10.0

        # 2. Map to the Treatment Range
        # A low severity ratio anchors near min_hours; a high ratio anchors near max_hours
        target_duration = min_hours + (severity_ratio * (max_hours - min_hours))

        # 3. Add Random Noise (Stochastic Variance)
        # Justification: Medical treatment isn't perfectly deterministic.
        # We add a +/- 10% random variance to the target duration.
        variance = target_duration * 0.10
        randomized_duration = np.random.uniform(target_duration - variance, target_duration + variance)

        # 4. Enforce limits and round to sync with the hourly clock
        # Guarantee it never drops below the absolute disease minimum
        final_duration = max(min_hours, randomized_duration)

        return max(1.0, round(final_duration))

    def process_discharges_and_admissions(self, use_priority_queue=False):
        """
        Checks beds for patients who have finished treatment, discharges them,
        and immediately pulls the next eligible patient from the front of the queue.
        """
        for i in range(self.bed_capacity):
            patient = self.beds[i]

            # --- 1. Discharge Logic ---
            # If the bed has a patient AND the clock has reached their discharge time
            if patient is not None and self.clock >= patient.discharge_time:
                # Log final stats and mark as officially complete
                patient.status = "Discharged"
                patient.total_time_in_system = patient.discharge_time - patient.arrival_time

                # Free the physical resource (the bed)
                self.beds[i] = None

            # --- 2. Pull from Queue Logic ---
            # If the bed is currently empty AND there are people waiting
            if self.beds[i] is None and len(self.queue) > 0:

                # --- NEW: SCENARIO 2 PRIORITY QUEUE LOGIC ---
                if use_priority_queue:
                    # Sort the queue by priority level (Highest number moves to index 0)
                    self.queue.sort(key=lambda p: p.priority_level, reverse=True)
                # --------------------------------------------

                # FIFO (or Priority-First): .pop(0) removes and returns the first person in the line
                next_patient = self.queue.pop(0)

                # Admit them to this newly freed bed
                self.beds[i] = next_patient
                next_patient.bed_number = i
                next_patient.status = "Admitted"

                # Record their admission timestamps
                next_patient.admission_time = self.clock
                next_patient.wait_to_bed = self.clock - next_patient.queue_entry_time

                # Generate their specific treatment time and stamp their future discharge
                treatment_time = self.generate_treatment_duration(next_patient)
                next_patient.discharge_time = self.clock + treatment_time

    def enforce_horizon(self):
        """
        Executes at exactly the simulation horizon (hour 720).
        Marks all patients remaining in the queue or in beds as 'Censored'
        to ensure accurate statistical outputs.
        """
        # 1. Censor patients still waiting in the queue
        for patient in self.queue:
            patient.status = "Censored"
            patient.rejection_reason = "horizon_reached"
            # Calculate how long they were in the system before the cutoff
            patient.total_time_in_system = self.horizon - patient.arrival_time

            # 2. Censor patients still occupying beds
        for i in range(self.bed_capacity):
            patient = self.beds[i]
            if patient is not None:
                patient.status = "Censored"
                # They were admitted, but the month ended before they were discharged
                patient.total_time_in_system = self.horizon - patient.arrival_time

        # 3. Clean up the facility (Optional but good practice)
        self.queue = []
        self.beds = [None] * self.bed_capacity

    def log_hourly_snapshot(self, scenario_name, replication_id):
        """
        Calculates and logs the current system pressure, utilization, and queue length.
        Designed to be called exactly at the top of every simulation hour.
        """
        # Count how many beds are currently occupied
        beds_used = sum(1 for bed in self.beds if bed is not None)
        available_beds = self.bed_capacity - beds_used
        queue_len = len(self.queue)

        # Calculate specific rubric metrics
        queue_full = 1 if queue_len >= self.queue_capacity else 0
        bed_utilization = beds_used / self.bed_capacity

        # System Pressure: Total patients in system / Total capacity (beds + queue max)
        total_capacity = self.bed_capacity + self.queue_capacity
        system_pressure = (beds_used + queue_len) / total_capacity

        # Over Capacity Pressure: Measures how strained the waiting room is specifically
        over_cap_pressure = queue_len / self.queue_capacity if self.queue_capacity > 0 else 0

        # Build the snapshot dictionary
        snapshot = {
            "scenario": scenario_name,
            "replication": replication_id,
            "time": self.clock,
            "beds_used": beds_used,
            "available_beds": available_beds,
            "queue_length": queue_len,
            "queue_full": queue_full,
            "bed_utilization": bed_utilization,
            "system_pressure": system_pressure,
            "over_capacity_pressure": over_cap_pressure
        }

        # Save it to our hospital's internal tracker
        self.hourly_log.append(snapshot)
