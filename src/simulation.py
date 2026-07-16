from patient import Patient, generate_interarrival_time
from hospital import Hospital
import numpy as np
import os
import pandas as pd


def run_simulation(scenario_name="Baseline", replication_id=1):
    """
    Runs a single full 30-day (720-hour) replication of the hospital simulation.
    Acts as the master event engine advancing the clock.
    """
    # 1. Initialize the Environment
    hospital = Hospital()

    # Set up our first future events
    next_arrival_time = generate_interarrival_time()
    next_monitor_time = 1.0
    patient_counter = 1

    # 2. The Master Event Loop
    # The simulation runs as long as the clock is strictly less than 720.0
    while hospital.clock < hospital.horizon:

        # Determine what happens next: an arrival or the hourly monitor tick?
        next_event_time = min(next_arrival_time, next_monitor_time)

        # Fast-forward the master clock to the exact time of the next event
        hospital.clock = next_event_time

        # --- EVENT A: HOURLY TICK ---
        if hospital.clock == next_monitor_time:
            # Because treatment and safe wait times were rounded to whole hours in Phase 2,
            # they will perfectly align with this hourly tick!

            # 1. Check for patients whose wait times expired
            hospital.check_queue_timeouts()

            # 2. Check for patients ready to leave beds (and pull from the queue)
            hospital.process_discharges_and_admissions()

            # 3. Take our data snapshot for the Hourly Monitoring Table
            hospital.log_hourly_snapshot(scenario_name, replication_id)

            # Schedule the next hourly tick
            next_monitor_time += 1.0

            # --- EVENT B: PATIENT ARRIVAL ---
        if hospital.clock == next_arrival_time:
            # 1. Create the new patient (we set arrival_time to 0 temporarily)
            new_patient = Patient(patient_id=f"P{patient_counter}", arrival_time=0)
            new_patient.assign_demographic_and_illness()
            new_patient.generate_vital_signs()
            new_patient.calculate_triage_scores()

            triage_duration = np.random.uniform(0.25, 0.50)

            # We record that they entered the building in the past,
            # and are requesting a bed exactly NOW at the current hospital.clock
            new_patient.arrival_time = hospital.clock - triage_duration

            # Store the patient object in the master log so we can export them later
            hospital.patient_log.append(new_patient)

            # 2. Route the patient through the hospital doors (Phase 2)
            hospital.handle_patient_arrival(new_patient)

            # 3. Schedule the next patient arrival
            next_arrival_time = hospital.clock + generate_interarrival_time()
            patient_counter += 1

    # 3. Horizon Handling (The 720-Hour Mark)
    # The while loop finishes exactly when the clock hits 720.0.
    # Now we trigger the shutdown sequence to censor remaining patients.
    hospital.clock = hospital.horizon
    hospital.enforce_horizon()

    # Return the hospital object, so we can extract its data in the next step
    return hospital


def run_experiment(num_replications=30, scenario_name="Baseline"):
    """
    Runs multiple independent replications of the simulation.
    Aggregates all patient and hourly data into master lists.
    """
    master_patient_log = []
    master_hourly_log = []

    print(f"--- Starting Experiment: {scenario_name} ({num_replications} Replications) ---")

    for rep in range(1, num_replications + 1):
        # 1. Enforce Independence (Strict Requirement)
        # We set a new, unique seed at the start of every replication.
        # Using a base number (like 42) + the replication number ensures the runs
        # are totally independent, but perfectly reproducible if you run the script again!
        np.random.seed(42 + rep)

        # 2. Run the Engine
        # This calls the function we wrote in Step 3.3
        hospital_result = run_simulation(scenario_name=scenario_name, replication_id=rep)

        # 3. Aggregate Patient Data
        # We loop through the generated patients and use the to_dict() method
        # we wrote in Step 3.1 to turn them into clean data rows.
        for patient in hospital_result.patient_log:
            master_patient_log.append(patient.to_dict(scenario_name, rep))

        # 4. Aggregate Hourly Data
        # .extend() takes the list of dictionaries from the hospital and unpacks
        # them directly into our master list.
        master_hourly_log.extend(hospital_result.hourly_log)

        print(f"Replication {rep} complete. Processed {len(hospital_result.patient_log)} patients.")

    print("--- Experiment Complete! ---")

    # Return the two massive datasets containing all 30 months of data
    return master_patient_log, master_hourly_log


if __name__ == "__main__":
    # 1. Run the 30-Replication Experiment
    patients_data, hourly_data = run_experiment(num_replications=30, scenario_name="Baseline")

    # 2. Convert the master lists into Pandas DataFrames
    print("Converting data to Pandas DataFrames...")
    df_patients = pd.DataFrame(patients_data)
    df_hourly = pd.DataFrame(hourly_data)

    # 3. Ensure the output directory exists
    # exist_ok=True means it won't crash if the folder is already there
    output_dir = os.path.join("..", "outputs", "tables")
    os.makedirs(output_dir, exist_ok=True)

    # 4. Define the file paths
    patient_file = os.path.join(output_dir, "patient_level_table_baseline.csv")
    hourly_file = os.path.join(output_dir, "hourly_monitoring_table_baseline.csv")

    # 5. Export to CSV (index=False prevents pandas from writing row numbers to the file)
    print("Exporting to CSV...")
    df_patients.to_csv(patient_file, index=False)
    df_hourly.to_csv(hourly_file, index=False)

    print("--- Simulation Complete & Data Saved! ---")
    print(f"Total Patients Logged: {len(df_patients)}")
    print(f"Total Hourly Snapshots Logged: {len(df_hourly)}")
    print(f"Files saved in: {output_dir}")
