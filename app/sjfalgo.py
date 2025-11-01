import matplotlib
# IMPORTANT: Use 'Agg' backend for non-GUI environments like Django
matplotlib.use('Agg') 

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# --- 1. Data Generation (Simulates the environment) ---
def generate_processes(arrivals):
    """Generates processes with simulated past bursts (features) and an 'Actual_Burst' time (target)."""
    n = len(arrivals)
    np.random.seed(42) 

    b1 = np.random.randint(2, 10, n) 
    b2 = np.random.randint(2, 10, n)
    b3 = np.random.randint(2, 10, n)

    # CPU load depends slightly on arrival time
    cpu_load = 0.3 + 0.05 * np.array(arrivals) + np.random.uniform(-0.05, 0.05, n)
    cpu_load = np.clip(cpu_load, 0.2, 0.9)

    df = pd.DataFrame({
        'PID': [f'P{i+1}' for i in range(n)],
        'Arrival': arrivals,
        'Burst1': b1, 'Burst2': b2, 'Burst3': b3, 
        'CPU_Load': cpu_load
    })

    # True burst calculation (ML target variable)
    df['Actual_Burst'] = (
        1.8 + 3.5 * df['CPU_Load'] + 0.4 * (b1 + b2 + b3) / 3 + 0.1 * np.array(arrivals) + np.random.normal(0, 0.3, n)
    )
    df['Actual_Burst'] = np.maximum(1.0, df['Actual_Burst']).round(2)
    return df.round(2)

# --- 2. Prediction Methods (Heuristics) ---
def simple_average(df):
    df['Simple_Avg'] = df[['Burst1', 'Burst2', 'Burst3']].mean(axis=1).round(2)
    return df

def exponential_average(df, alpha=0.6):
    preds = []
    for i in range(len(df)):
        b = [df.loc[i, 'Burst1'], df.loc[i, 'Burst2'], df.loc[i, 'Burst3']]
        pred = b[0] 
        for j in range(1, len(b)):
            pred = alpha * b[j] + (1 - alpha) * pred
        preds.append(round(pred, 2))
    df['Exp_Avg'] = preds
    return df

# --- 3. ML Model Prediction ---
def ml_prediction(df):
    X = df[['CPU_Load', 'Burst1', 'Burst2', 'Burst3']]
    y = df['Actual_Burst']
    model = LinearRegression()
    model.fit(X, y)
    
    predictions = model.predict(X)
    df['ML_Pred'] = np.maximum(1.0, predictions).round(2)
    return df

# --- 4. SJF Scheduling ---
def sjf(df, use_col):
    """Performs dynamic, non-preemptive SJF scheduling using a specified prediction column."""
    processes = df.copy()
    processes['Executed'] = False
    current_time = 0.0
    metrics_list = []
    
    while not processes['Executed'].all():
        ready_queue = processes[
            (processes['Arrival'] <= current_time) & (processes['Executed'] == False)
        ]
        
        if ready_queue.empty:
            unexecuted = processes[processes['Executed'] == False]
            if unexecuted.empty: break
            
            next_arrival_time = unexecuted['Arrival'].min()
            
            if next_arrival_time > current_time:
                metrics_list.append({'PID': 'IDLE', 'Start': current_time, 'End': next_arrival_time})
            current_time = next_arrival_time
            continue
        
        next_job = ready_queue.sort_values(by=[use_col, 'Arrival']).iloc[0]
        pid = next_job['PID']
        burst = next_job[use_col]
        
        start_time = current_time
        end_time = start_time + burst
        current_time = end_time
        
        completion = round(end_time, 2)
        turnaround = round(completion - next_job.Arrival, 2)
        waiting = round(turnaround - burst, 2)
        
        metrics_list.append({
            'PID': pid, 'Arrival': next_job['Arrival'], 'ML_Pred': burst,
            'Completion': completion, 'Turnaround': turnaround, 'Waiting': waiting,
            'Start': start_time, 'End': end_time
        })
        processes.loc[processes['PID'] == pid, 'Executed'] = True

    final_schedule_df = pd.DataFrame([m for m in metrics_list if m['PID'] != 'IDLE'])

    if final_schedule_df.empty:
        return 0.0, 0.0, []

    avg_turnaround = final_schedule_df['Turnaround'].mean()
    avg_waiting = final_schedule_df['Waiting'].mean()
    
    return avg_turnaround, avg_waiting, metrics_list

# --- 5. Plotting Functions ---

def plot_gantt_chart(timeline_list):
    """Generates the Gantt chart for the ML-predicted SJF schedule."""
    pids = [p['PID'] for p in timeline_list if p['PID'] != 'IDLE']
    unique_pids = sorted(list(set(pids)))
    fig, ax = plt.subplots(figsize=(10, 2.5))
    
    colors_map = plt.cm.get_cmap('Spectral', len(unique_pids) + 1)
    color_map = {pid: colors_map(i) for i, pid in enumerate(unique_pids)}
    color_map['IDLE'] = 'lightgray'

    for segment in timeline_list:
        duration = segment['End'] - segment['Start']
        if duration > 0:
            pid = segment['PID']
            ax.barh(0, duration, left=segment['Start'], 
                            color=color_map[pid], edgecolor='k', height=0.5)
            
            if pid != 'IDLE':
                ax.text((segment['Start'] + segment['End']) / 2, 0, pid, 
                                va='center', ha='center', color='black', fontweight='bold')

    ax.set_yticks([])
    ax.set_xlabel("Time (ms)")
    ax.set_title("3. SJF Gantt Chart (ML Prediction)", fontsize=12)
    ax.set_xlim(left=0)
    
    if timeline_list:
        max_time = timeline_list[-1]['End']
        ax.set_xticks(np.arange(0, max_time + 1, max(1, int(max_time / 10)))) 
    
    fig.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_burst_prediction_comparison(df):
    """
    Generates a grouped bar chart comparing Actual Burst against 
    Simple, Exponential, and ML predictions for each process. (Chart 1)
    """
    pids = df['PID']
    bar_width = 0.18
    x = np.arange(len(pids))
    
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the four values for each process
    rects1 = ax.bar(x - 1.5*bar_width, df['Actual_Burst'], bar_width, label='Actual Burst', color='#003366')
    rects2 = ax.bar(x - 0.5*bar_width, df['ML_Pred'], bar_width, label='ML Predicted', color='#8a10c5')
    rects3 = ax.bar(x + 0.5*bar_width, df['Exp_Avg'], bar_width, label='Exp. Avg', color='#FF9900')
    rects4 = ax.bar(x + 1.5*bar_width, df['Simple_Avg'], bar_width, label='Simple Avg', color='#00AA00')

    ax.set_ylabel('Burst Time (ms)')
    ax.set_title('1. Burst Prediction Comparison (Actual vs. Heuristics/ML)')
    ax.set_xticks(x)
    ax.set_xticklabels(pids)
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)
    
    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_actual_burst_vs_arrival(df):
    """
    Generates a scatter plot showing the relationship between Actual Burst Time and Arrival Time. (Chart 2)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(df['Arrival'], df['Actual_Burst'], color='#007ACC', s=100, alpha=0.8, edgecolors='w', linewidth=0.5, zorder=3)

    # Annotate points with PID
    for i in range(len(df)):
        ax.annotate(df['PID'].iloc[i], 
                    (df['Arrival'].iloc[i], df['Actual_Burst'].iloc[i]),
                    textcoords="offset points", xytext=(5,5), ha='left', fontsize=9)

    ax.set_xlabel("Arrival Time (ms)")
    ax.set_ylabel("Actual Burst Time (ms)")
    ax.set_title("2. Actual Burst Time vs. Arrival Time")
    ax.grid(True, linestyle=':', alpha=0.6)

    fig.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# --- 6. Main Runner Function ---
def run_all_simulations(arrivals):
    """Runs all prediction methods and SJF simulations, returning data and charts."""
    df = generate_processes(arrivals) 
    df = simple_average(df)
    df = exponential_average(df)
    df = ml_prediction(df)
    
    simulation_results = {}
    
    # Run SJF simulations for all prediction methods to get avg metrics
    # 1. Simple Avg SJF
    avg_tat_simple, avg_wt_simple, _ = sjf(df, use_col='Simple_Avg')
    simulation_results['Simple_Avg'] = {'avg_tat': avg_tat_simple, 'avg_wt': avg_wt_simple}

    # 2. Exponential Avg SJF
    avg_tat_exp, avg_wt_exp, _ = sjf(df, use_col='Exp_Avg')
    simulation_results['Exp_Avg'] = {'avg_tat': avg_tat_exp, 'avg_wt': avg_wt_exp}

    # 3. ML Predicted SJF (Main schedule data)
    avg_tat_ml, avg_wt_ml, timeline_list_ml = sjf(df, use_col='ML_Pred')
    final_schedule_df_ml = pd.DataFrame([m for m in timeline_list_ml if m['PID'] != 'IDLE'])

    simulation_results['ML_Pred'] = {'avg_tat': avg_tat_ml, 'avg_wt': avg_wt_ml, 'schedule_df': final_schedule_df_ml}
    
    # Generate the three requested charts
    burst_comparison_base64 = plot_burst_prediction_comparison(df) 
    burst_arrival_base64 = plot_actual_burst_vs_arrival(df) 
    gantt_base64 = plot_gantt_chart(timeline_list_ml)

    # Returning the full DF (for tables) and the three specific charts
    return df, simulation_results, burst_comparison_base64, burst_arrival_base64, gantt_base64