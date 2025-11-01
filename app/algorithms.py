import numpy as np
from collections import deque
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import os
import io
import base64
import matplotlib.pyplot as plt

# --- Configuration ---
# NOTE: In a real Django app, you'd handle the model file path carefully.
MODEL_PATH = 'predictive_model_gbr.pkl'

# --- ML Model & Utility Functions ---

def train_predictive_model():
    data = []
    for _ in range(250):
        cpu = np.random.randint(10, 100)
        mem = np.random.randint(50, 500)
        io_op = np.random.randint(5, 50)
        prev_burst = np.random.randint(1, 25)
        burst = int(prev_burst * (0.7 + np.random.rand() * 0.6)) + np.random.randint(0, 4)
        data.append([cpu, mem, io_op, prev_burst, burst])
        
    # Renamed 'io' column to 'io_op' to avoid conflict if any
    df = pd.DataFrame(data, columns=['cpu', 'mem', 'io_op', 'prev_burst', 'burst'])
    X = df[['cpu', 'mem', 'io_op', 'prev_burst']]
    y = df['burst']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, MODEL_PATH)
    return model

def load_predictive_model():
    """Loads or trains the predictive model."""
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return train_predictive_model()
    else:
        return train_predictive_model()

def predict_burst(model, features):
    """Predicts the next burst time. Features order: [cpu, mem, io_op, prev_burst]"""
    # Note: features must match training features: 'cpu', 'mem', 'io_op', 'prev_burst'
    return model.predict([features])[0]

# --- Scheduling Algorithms ---

def recalculate_metrics(schedule_order, processes_dict):
    """Calculates final metrics for a given NON-PREEMPTIVE process order."""
    time = 0
    # Create a list for the final results to avoid modifying the input list
    final_results = []
    
    for p in schedule_order:
        process_data = processes_dict[p['id']]
        # Use a copy of the process dictionary to store results
        result_p = p.copy() 
        
        actual_burst = process_data.get('actual_burst', process_data['burst'])
        arrival_time = process_data['arrival']
        
        if time < arrival_time:
            time = arrival_time
            
        result_p['start'] = time
        result_p['completion'] = time + actual_burst
        result_p['waiting'] = result_p['start'] - arrival_time
        result_p['turnaround'] = result_p['completion'] - arrival_time
        
        time += actual_burst
        final_results.append(result_p)
        
    return final_results

def fcfs(processes):
    processes_dict = {p['id']: p for p in processes}
    # Create a copy before sorting to avoid modifying the original list
    schedule_order = sorted(processes.copy(), key=lambda x: x['arrival'])
    return recalculate_metrics(schedule_order, processes_dict)

def sjf(processes):
    """Non-preemptive Shortest Job First with robust simulation."""
    processes_dict = {p['id']: p for p in processes}
    # Use a copy for the process queue
    proc_queue = sorted(processes.copy(), key=lambda x: x['arrival'])
    completed_order = []
    ready_queue = []
    time = 0

    while proc_queue or ready_queue:
        while proc_queue and proc_queue[0]['arrival'] <= time:
            ready_queue.append(proc_queue.pop(0))

        if ready_queue:
            # Sort by predicted burst time
            ready_queue.sort(key=lambda x: x['burst'])
            p = ready_queue.pop(0)
            completed_order.append(p)
            actual_burst = processes_dict[p['id']].get('actual_burst', p['burst'])
            time += actual_burst
        else:
            if proc_queue:
                time = proc_queue[0]['arrival']
                
    return recalculate_metrics(completed_order, processes_dict)


def priority(processes):
    """Non-preemptive Priority Scheduling with robust simulation."""
    processes_dict = {p['id']: p for p in processes}
    proc_queue = sorted(processes.copy(), key=lambda x: x['arrival'])
    completed_order = []
    ready_queue = []
    time = 0

    while proc_queue or ready_queue:
        while proc_queue and proc_queue[0]['arrival'] <= time:
            ready_queue.append(proc_queue.pop(0))

        if ready_queue:
            # Sort by priority (higher number = higher priority based on Tkinter logic)
            ready_queue.sort(key=lambda x: x['priority'], reverse=True) 
            p = ready_queue.pop(0)
            completed_order.append(p)
            actual_burst = processes_dict[p['id']].get('actual_burst', p['burst'])
            time += actual_burst
        else:
            if proc_queue:
                time = proc_queue[0]['arrival']
                
    return recalculate_metrics(completed_order, processes_dict)


def round_robin(processes, quantum):
    """Preemptive Round Robin with correct, internal metric calculation."""
    time = 0
    ready_queue = deque()
    # Use a copy for proc_queue
    proc_queue = deque(sorted(processes.copy(), key=lambda p: p['arrival']))
    
    # Use actual_burst if available, otherwise use the provided burst
    remaining_burst = {p['id']: p.get('actual_burst', p['burst']) for p in processes}
    start_times = {}
    completion_times = {}
    
    # The scheduling order itself isn't needed for non-Gantt metrics in RR
    
    while proc_queue or ready_queue:
        # Add processes that have arrived to the ready queue
        while proc_queue and proc_queue[0]['arrival'] <= time:
            arrived_proc = proc_queue.popleft()
            ready_queue.append(arrived_proc)

        if ready_queue:
            p = ready_queue.popleft()
            pid = p['id']

            if pid not in start_times:
                start_times[pid] = time
            
            exec_time = min(quantum, remaining_burst[pid])
            time += exec_time
            remaining_burst[pid] -= exec_time

            # Add new arrivals during the execution time slice
            while proc_queue and proc_queue[0]['arrival'] <= time:
                ready_queue.append(proc_queue.popleft())

            if remaining_burst[pid] == 0:
                completion_times[pid] = time
            else:
                ready_queue.append(p)
        else:
            # CPU is idle, advance time to the next arrival
            if proc_queue:
                time = proc_queue[0]['arrival']

    results = []
    for p in processes:
        pid = p['id']
        actual_burst = p.get('actual_burst', p['burst'])
        completion = completion_times.get(pid, 0)
        arrival = p['arrival']
        start = start_times.get(pid, 0)
        
        turnaround = completion - arrival
        waiting = turnaround - actual_burst

        results.append({**p, 'start': start, 'completion': completion, 'waiting': waiting, 'turnaround': turnaround})
    
    return results

# --- Plotting Functions (Base64 for Django) ---

def plot_bar_charts_base64(all_results):
    """Generates a bar chart and returns it as a base64 encoded PNG string."""
    fig = plt.Figure(figsize=(10, 5), dpi=100)
    ax = fig.subplots()
    labels = list(all_results.keys())
    
    # Calculate means, handle empty lists gracefully
    def safe_mean(data):
        return np.mean(data) if data else 0

    waits = [safe_mean([p['waiting'] for p in r]) for r in all_results.values()]
    turns = [safe_mean([p['turnaround'] for p in r]) for r in all_results.values()]
    
    x = np.arange(len(labels))
    width = 0.35

    r1 = ax.bar(x - width/2, waits, width, label='Avg Waiting Time', color='skyblue')
    r2 = ax.bar(x + width/2, turns, width, label='Avg Turnaround Time', color='salmon')
    
    ax.set_ylabel('Time (ms)')
    ax.set_title('Algorithm Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()

    # Add labels to the bars
    ax.bar_label(r1, padding=3, fmt='%.2f')
    ax.bar_label(r2, padding=3, fmt='%.2f')
    
    fig.tight_layout()

    # Convert plot to PNG image in memory
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return image_base64


def generate_gantt_chart_base64(user_result, rec_result, user_name, rec_name):
    """
    Generates a combined Gantt chart comparison and returns its Base64 encoding.

    The structure is translated from the user's provided Tkinter logic using 
    matplotlib's Figure and broken_barh function.
    """
    if not user_result and not rec_result:
        return ""
    
    # 1. Identify all unique PIDs to set Y-axis consistently
    all_pids = set()
    if user_result:
        all_pids.update(p['id'] for p in user_result)
    if rec_result:
        all_pids.update(p['id'] for p in rec_result)
        
    pids = sorted(list(all_pids))
    y_labels = [f"P{pid}" for pid in pids]
    y_pos = np.arange(len(y_labels))
    pid_to_index = {pid: i for i, pid in enumerate(pids)}

    # --- Plot Setup ---
    # Use Figure and subplots for Django compatibility
    fig = plt.Figure(figsize=(10, 5), dpi=100)
    # Create two subplots stacked vertically, sharing the X-axis
    axs = fig.subplots(2, 1, sharex=True)
    fig.suptitle("Gantt Chart Comparison", fontsize=16)

    # Clean the recommended name for display
    clean_rec_name = rec_name.replace('ML: ', '')
    
    # -------------------- Plot 1: User's Choice --------------------
    axs[0].set_title(f"Your Choice: {user_name}")
    
    for p in user_result: 
        # Check for valid PID before plotting
        if p['id'] not in pid_to_index: continue
        
        p_index = pid_to_index[p['id']]
        start = p['start']
        # Use 'completion' and 'start' keys from the result dictionary
        duration = p['completion'] - p['start']
        
        # Plot the bar: (x_start, x_duration), y_position
        # y_position is slightly offset to center the bar (0.1 start, 0.8 height)
        axs[0].broken_barh([(start, duration)], (p_index + 0.1, 0.8), 
                           facecolors='tab:blue', edgecolor='black')
        
        # Add text label (P1, P2, etc.)
        text_x = start + duration / 2
        text_y = p_index + 0.5  # Center of the bar (y_pos + 0.1 + 0.8/2)
        axs[0].text(text_x, text_y, f"P{p['id']}", 
                    ha='center', va='center', color='white', fontweight='bold', fontsize=9)
    
    # ------------------ Plot 2: ML Recommended -------------------
    axs[1].set_title(f"Recommended: {clean_rec_name}")

    for p in rec_result: 
        if p['id'] not in pid_to_index: continue
        
        p_index = pid_to_index[p['id']]
        start = p['start']
        duration = p['completion'] - p['start']
        
        # Plot the bar
        axs[1].broken_barh([(start, duration)], (p_index + 0.1, 0.8), 
                           facecolors='tab:orange', edgecolor='black')
        
        # Add text label (P1, P2, etc.)
        text_x = start + duration / 2
        text_y = p_index + 0.5
        axs[1].text(text_x, text_y, f"P{p['id']}", 
                    ha='center', va='center', color='white', fontweight='bold', fontsize=9)

    # -------------------- Axis Configuration --------------------
    for ax in axs: 
        # Set Y-axis labels and ticks to the middle of the bar
        ax.set_yticks(y_pos + 0.5) 
        ax.set_yticklabels(y_labels)
        ax.set_ylabel('Process')
        ax.grid(True, axis='x', ls='--', alpha=0.6)
        # Ensure y-axis limits cover all processes
        ax.set_ylim(0, len(y_labels)) 
        
    axs[1].set_xlabel("Time (ms)")
    
    # Final cleanup and save
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save chart to a Base64 string
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig) # Close figure to free memory
    
    return image_base64