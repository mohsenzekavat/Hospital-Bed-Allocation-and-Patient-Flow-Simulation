class Hospital:
    """
    Manages the global state of the hospital simulation, including the clock,
    resources (beds), and the waiting queue.
    """
    def __init__(self):
        # --- Simulation Clock & Horizon ---
        self.clock = 0.0
        # The simulation horizon is 30 days, measured in hours (30 * 24 = 720 hours)
        self.horizon = 720.0

        # --- Capacity Constraints ---
        # The hospital has exactly 10 inpatient beds
        self.bed_capacity = 10

        # We use a list of 10 slots to represent the physical beds.
        # 'None' means the bed is empty. When occupied, it will hold a Patient object.
        # This makes it easy to assign a specific 'bed_number' (index 0 to 9).
        self.beds = [None] * self.bed_capacity

        # The queue can hold a maximum of 5 patients
        self.queue_capacity = 5

        # We use a standard list for the queue. Using a list's .append() and .pop(0)
        # methods will perfectly mimic the required First-In-First-Out (FIFO) logic.
        self.queue = []

        # --- Data Collection Tracking ---
        # These lists will store the dictionaries for every generated patient and hourly snapshot
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
