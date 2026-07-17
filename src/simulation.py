from patient import Patient, generate_interarrival_time
from hospital import Hospital
import numpy as np
import os
import pandas as pd


def run_simulation(scenario_name="Baseline", replication_id=1, beds=10, arr_min=1.0, arr_max=3.0, use_priority=False):
    """
    Runs a single full 30-day (720-hour) replication of the hospital simulation.
    Acts as the master event engine advancing the clock.
    """
    # 1. Initialize the Environment (Cleanly parameterized!)
    hospital = Hospital(bed_capacity=beds)

    # Set up our first future events (using parameterized arrival times)
    next_arrival_time = np.random.uniform(arr_min, arr_max)
    next_monitor_time = 1.0
    patient_counter = 1

    # 2. The Master Event Loop
    while hospital.clock < hospital.horizon:
        next_event_time = min(next_arrival_time, next_monitor_time)
        hospital.clock = next_event_time

        # --- EVENT A: HOURLY TICK ---
        if hospital.clock == next_monitor_time:
            hospital.check_queue_timeouts()

            # --- SCENARIO UPDATE: Pass the priority flag to the admission logic ---
            hospital.process_discharges_and_admissions(use_priority_queue=use_priority)

            hospital.log_hourly_snapshot(scenario_name, replication_id)
            next_monitor_time += 1.0

        # --- EVENT B: PATIENT ARRIVAL ---
        if hospital.clock == next_arrival_time:
            new_patient = Patient(patient_id=f"P{patient_counter}", arrival_time=0)
            new_patient.assign_demographic_and_illness()
            new_patient.generate_vital_signs()
            new_patient.calculate_triage_scores()

            triage_duration = np.random.uniform(0.25, 0.50)
            new_patient.arrival_time = hospital.clock - triage_duration

            hospital.patient_log.append(new_patient)
            hospital.handle_patient_arrival(new_patient)

            # --- SCENARIO UPDATE: Schedule next arrival using parameters ---
            next_arrival_time = hospital.clock + np.random.uniform(arr_min, arr_max)
            patient_counter += 1

    # 3. Horizon Handling (The 720-Hour Mark)
    hospital.clock = hospital.horizon
    hospital.enforce_horizon()

    return hospital


def run_experiment(num_replications=30, scenario_name="Baseline", beds=10, arr_min=1.0, arr_max=3.0,
                   use_priority=False):
    """
    Runs multiple independent replications of the simulation.
    Aggregates all patient and hourly data into master lists.
    """
    master_patient_log = []
    master_hourly_log = []

    print(f"--- Starting Experiment: {scenario_name} ({num_replications} Replications) ---")

    for rep in range(1, num_replications + 1):
        np.random.seed(42 + rep)

        # --- SCENARIO UPDATE: Pass all parameters down to the simulation engine ---
        hospital_result = run_simulation(scenario_name, rep, beds, arr_min, arr_max, use_priority)

        for patient in hospital_result.patient_log:
            master_patient_log.append(patient.to_dict(scenario_name, rep))

        master_hourly_log.extend(hospital_result.hourly_log)

        print(f"Replication {rep} complete. Processed {len(hospital_result.patient_log)} patients.")

    print(f"--- {scenario_name} Complete! ---")
    return master_patient_log, master_hourly_log


# ==========================================
# EXPLICIT SCENARIO RUNNERS
# ==========================================

def run_baseline_experiment():
    """Runs the strictly required baseline model."""
    print("\nExecuting Baseline Experiment...")
    return run_experiment(
        scenario_name="Baseline",
        beds=10, arr_min=1.0, arr_max=3.0, use_priority=False
    )


def run_scenario_1_capacity():
    """SCENARIO 1: Increase beds from 10 to 12."""
    print("\nExecuting Scenario 1: Capacity Expansion...")
    return run_experiment(
        scenario_name="Scenario_1_Capacity",
        beds=12, arr_min=1.0, arr_max=3.0, use_priority=False
    )


def run_scenario_2_priority():
    """SCENARIO 2: Change queue from FIFO to Priority-Based."""
    print("\nExecuting Scenario 2: Priority Queue...")
    return run_experiment(
        scenario_name="Scenario_2_Priority",
        beds=10, arr_min=1.0, arr_max=3.0, use_priority=True  # Turning the flag ON
    )


def run_scenario_3_surge():
    """SCENARIO 3: Winter Surge (Faster Arrivals)."""
    print("\nExecuting Scenario 3: Winter Surge...")
    return run_experiment(
        scenario_name="Scenario_3_Surge",
        beds=10, arr_min=0.5, arr_max=2.5, use_priority=False  # Shrinking the arrival time
    )


# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":

    # Ensure the output directory exists
    output_dir = os.path.join("..", "outputs", "tables")
    os.makedirs(output_dir, exist_ok=True)

    # Put all our runner functions in a list, so,we can loop through them
    experiments_to_run = [
        run_baseline_experiment,
        run_scenario_1_capacity,
        run_scenario_2_priority,
        run_scenario_3_surge
    ]

    print("--- INITIATING FULL PROJECT SIMULATION SUITE ---")

    # Loop through each function and save its data
    for run_func in experiments_to_run:
        patients_data, hourly_data = run_func()

        df_patients = pd.DataFrame(patients_data)
        df_hourly = pd.DataFrame(hourly_data)

        # Grab the scenario name directly from the data to name our files safely
        scenario_name_safe = df_patients.iloc[0]['scenario']

        patient_file = os.path.join(output_dir, f"patient_level_table_{scenario_name_safe}.csv")
        hourly_file = os.path.join(output_dir, f"hourly_monitoring_table_{scenario_name_safe}.csv")

        print(f"Exporting {scenario_name_safe} to CSV...")
        df_patients.to_csv(patient_file, index=False)
        df_hourly.to_csv(hourly_file, index=False)

    print("\n--- ALL SCENARIOS COMPLETE & DATA SAVED ---")
    print(f"Check your '{output_dir}' folder for the 8 CSV files.")
