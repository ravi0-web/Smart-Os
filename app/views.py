from django.shortcuts import render
from django.http import HttpResponseBadRequest

from . import algorithms
import numpy as np
import json # You likely have this already, but ensure it's imported
from . import sjfalgo
from .sjfalgo import run_all_simulations 

from .page_replacement_logic import run_page_replacement_simulation, make_patterned_sequence
import random

# Load the ML model once when the server starts
try:
    ML_MODEL = algorithms.load_predictive_model()
except Exception as e:
    print(f"Error loading ML model: {e}. Predictive features will be disabled.")
    ML_MODEL = None

def cpu(request):
    """Handles the main form submission and simulation logic."""
    # Initialize context with submitted_data set to a JSON string of an empty dict 
    # for the template's JavaScript to safely parse on initial GET.
    context = {'all_results': None, 'submitted_data': json.dumps({})}

    if request.method == 'POST':
        # --- NEW: Dictionary to store submitted form data for template reload ---
        # Initialize with static form fields
        submitted_data = {
            'num_processes': request.POST.get('num_processes', '3'),
            'user_algo': request.POST.get('user_algo', 'FCFS'),
            'quantum': request.POST.get('quantum', '4'),
        }
        # ---------------------------------------------------------------------

        try:
            # 1. Parse Input Data from POST request
            num_processes = int(submitted_data['num_processes'])
            user_algo = submitted_data['user_algo']
            quantum = int(submitted_data['quantum'])
            
            processes = []
            for i in range(1, num_processes + 1):
                arrival_str = request.POST.get(f'P{i}_arrival', '0')
                burst_str = request.POST.get(f'P{i}_burst', '1') 
                priority_str = request.POST.get(f'P{i}_priority', '1')

                # Store process input data for template reload
                submitted_data[f'P{i}_arrival'] = arrival_str
                submitted_data[f'P{i}_burst'] = burst_str
                submitted_data[f'P{i}_priority'] = priority_str
                
                # Safely convert to int
                arrival = int(arrival_str)
                burst = int(burst_str)
                priority_val = int(priority_str)
                
                # Ensure burst time is positive before proceeding
                if burst <= 0:
                    continue

                # 2. ML Feature Generation (Heuristic)
                pid = i
                cpu_feat = (pid * 17 + arrival * 5) % 80 + 20
                mem_feat = (burst * 23 + pid * 7) % 450 + 50
                io_feat = (arrival * 9 + burst * 3) % 45 + 5

                # 3. ML Prediction
                if 'ML_MODEL' in globals() and ML_MODEL:
                    # Pass features in the correct order: [cpu, mem, io_op, prev_burst]
                    predicted_burst = max(1, round(algorithms.predict_burst(ML_MODEL, [cpu_feat, mem_feat, io_feat, burst])))
                else:
                    predicted_burst = burst # Fallback
                    
                processes.append({
                    'id': pid, 'arrival': arrival, 'priority': priority_val,
                    'actual_burst': burst,
                    'predicted_burst': predicted_burst 
                })

            if not processes:
                # If all processes were skipped, render with submitted data but no results
                context['submitted_data'] = json.dumps(submitted_data)
                return render(request, 'cpu.html', context)

            # 4. Run Simulations (Actual vs. ML-Predicted)
            all_results = {}
            # Prepare process lists for actual and predicted burst times
            procs_actual = [{'id':p['id'], 'arrival':p['arrival'], 'priority':p['priority'], 'burst':p['actual_burst']} for p in processes]
            procs_predicted = [{'id':p['id'], 'arrival':p['arrival'], 'priority':p['priority'], 'burst':p['predicted_burst'], 'actual_burst':p['actual_burst']} for p in processes]

            # --- User's choice simulation (Actual Burst) ---
            if user_algo == "Round Robin":
                all_results[f"Your Choice: {user_algo} (Q={quantum})"] = algorithms.round_robin([p.copy() for p in procs_actual], quantum)
            else:
                algo_func = getattr(algorithms, user_algo.lower())
                all_results[f"Your Choice: {user_algo}"] = algo_func([p.copy() for p in procs_actual])
            
            user_result_key = [k for k in all_results.keys() if k.startswith("Your Choice:")][0]

            # --- ML-Predicted simulations (Predicted Burst, validated against Actual) ---
            all_results["ML: FCFS"] = algorithms.fcfs([p.copy() for p in procs_predicted])
            all_results["ML: SJF"] = algorithms.sjf([p.copy() for p in procs_predicted])
            all_results["ML: Priority"] = algorithms.priority([p.copy() for p in procs_predicted])
            all_results["ML: Round Robin (Q=4)"] = algorithms.round_robin([p.copy() for p in procs_predicted], quantum=4)
            
            # 5. Determine Best ML Algorithm & Metrics
            ml_results = {k:v for k,v in all_results.items() if "ML:" in k}
            best_ml_algo = min(ml_results, key=lambda k: np.mean([p['waiting'] for p in ml_results[k]]))
            
            user_avg_wait = np.mean([p['waiting'] for p in all_results[user_result_key]])
            best_avg_wait = np.mean([p['waiting'] for p in all_results[best_ml_algo]])

            # 6. Generate Plots
            # CRITICAL FIX 1: Use the correct function name: plot_bar_charts_base64
            plot_base64 = algorithms.plot_bar_charts_base64(all_results) 
            
            # CRITICAL FIX 2: Call the Gantt chart function
            user_results = all_results[user_result_key]
            best_ml_results = all_results[best_ml_algo]
            
            user_name_only = user_algo
            best_ml_algo_name = best_ml_algo.replace('ML: ', '')
            
            gantt_base64 = algorithms.generate_gantt_chart_base64(
                user_results,
                best_ml_results,
                user_result_key, # Pass the full name for the user (includes Q for RR)
                best_ml_algo      # Pass the full name for ML (includes ML: prefix)
            )

            # 7. Prepare Context for Template
            context.update({
                'all_results': all_results,
                'user_algo': user_algo,
                'user_avg_wait': f"{user_avg_wait:.2f}",
                'best_ml_algo_name': best_ml_algo_name,
                'best_avg_wait': f"{best_avg_wait:.2f}",
                'plot_base64': plot_base64,
                'gantt_base64': gantt_base64, # <-- NEW: Pass the Gantt chart
                'submitted_data': json.dumps(submitted_data), # <-- JSON encode for JS parsing
            })
            
        except (ValueError, KeyError, AttributeError, TypeError) as e:
            print(f"Simulation Error: {e}")
            return HttpResponseBadRequest(f"Invalid input data or simulation error: {e}")
        
    return render(request, 'cpu.html', context)

def paging(request):
    context = {}
    
    if request.method == 'POST':
        form_data = request.POST
        page_seq_str = form_data.get('page_sequence', '').strip()
        capacity_str = form_data.get('capacity', '3').strip()
        
        # Handle "Generate Random" button action
        if 'generate_random' in form_data:
            pattern = random.choice(["sequential","loop","random","localized","bursty"])
            pages = make_patterned_sequence(pattern)
            context['initial_pages'] = ",".join(map(str, pages))
            context['initial_capacity'] = capacity_str
            context['message'] = f"Generated {pattern} pattern."
            return render(request, 'paging.html', context)

        try:
            # Parse inputs
            pages = [int(x.strip()) for x in page_seq_str.split(",") if x.strip()]
            capacity = int(capacity_str)
            if not pages or capacity <= 0:
                 raise ValueError("Invalid page sequence or frame capacity.")

            # Run the simulation
            results = run_page_replacement_simulation(pages, capacity)
            
            # Prepare context
            context.update({
                'results': results,
                'initial_pages': page_seq_str,
                'initial_capacity': capacity_str,
            })

        except ValueError as e:
            context['error'] = f"Invalid Input: {e}. Please ensure pages are comma-separated integers and capacity is a positive integer."
            context['initial_pages'] = page_seq_str
            context['initial_capacity'] = capacity_str

    # Render template (use default values if not POST)
    if 'initial_pages' not in context:
        context['initial_pages'] = "0, 1, 2, 3, 0, 1, 4, 0, 1, 2, 3, 4"
        context['initial_capacity'] = "3"

    return render(request, 'paging.html', context)









def sjf(request):
    """
    Handles the ML-Predicted SJF submission, runs the simulation, and renders the results.
    """
    context = {}
    
    if request.method == 'POST':
        form_data = request.POST
        num_processes = int(form_data.get('num_processes', 0))
        
        arrivals = []
        
        try:
            for i in range(1, num_processes + 1):
                arrival_key = f'P{i}_arrival'
                arrivals.append(float(form_data.get(arrival_key, 0.0)))
            
            if not arrivals:
                 raise ValueError("Incomplete process data.")

        except ValueError:
            context['error'] = "Invalid input: Please ensure all Arrival Times are valid numbers."
            context['submitted_data_sjf'] = json.dumps(dict(form_data))
            return render(request, 'sjf_ml_predictor.html', context)

        # 2. Run Simulation
        try:
            # Capturing the three specific charts
            df_full, simulation_results, burst_comparison_base64, burst_arrival_base64, gantt_base64_sjf = run_all_simulations(arrivals)
            
            # 3. Format Output for Template
            initial_predictions = df_full[[
                'PID', 'Arrival', 'CPU_Load', 'Actual_Burst', 'Simple_Avg', 'Exp_Avg', 'ML_Pred'
            ]].to_dict('records')
            
            ml_schedule_df = simulation_results['ML_Pred']['schedule_df']
            sjf_schedule = ml_schedule_df[['PID', 'Arrival', 'ML_Pred', 'Completion', 'Turnaround', 'Waiting']].to_dict('records')
            
            sjf_results = {
                'avg_turnaround': simulation_results['ML_Pred']['avg_tat'],
                'avg_waiting': simulation_results['ML_Pred']['avg_wt'],
                'schedule': sjf_schedule
            }

            # 4. Prepare Context for Template
            context.update({
                'sjf_results': sjf_results,
                'initial_predictions': initial_predictions,
                'burst_comparison_base64': burst_comparison_base64, # Chart 1
                'burst_arrival_base64': burst_arrival_base64,       # Chart 2
                'gantt_base64_sjf': gantt_base64_sjf,               # Chart 3
                'submitted_data_sjf': json.dumps(dict(form_data)),
            })
            
        except Exception as e:
            context['error'] = f"A simulation error occurred: {e}"
            context['submitted_data_sjf'] = json.dumps(dict(form_data))

    # 5. Render Template
    return render(request, 'sjf.html', context)

def index(request):
    return render(request, "index.html")
# ==============================================
# SMART OS → RULE-BASED CHATBOT (No OpenAI API)
# ==============================================

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
import re

@csrf_exempt
def chatbot_api(request):
    """Smart OS → Learning-based (Rule-based) Chatbot"""
    if request.method != "POST":
        return JsonResponse({"response": "Invalid request"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
        user_msg = data.get("message", "").lower().strip()

        if not user_msg:
            return JsonResponse({"response": "Please ask a question about Operating Systems!"})

        # --- RULE-BASED KNOWLEDGE BASE ---
        responses={
    # Basics
    "  operating system": "An OS manages hardware, executes programs, and acts as an interface between user and machine.",
    "os": "An OS manages hardware, executes programs, and acts as an interface between user and machine.",
    "functions of operating system": "Key functions: process management, memory management, file system management, device management, I/O control, and security.",
    "goals of os": "Primary goals are convenience, efficiency, and ability to evolve.",
    "types of operating system": "Types: Batch OS, Time-Sharing OS, Distributed OS, Real-Time OS, Network OS, Mobile OS, and Embedded OS.",
    "os examples": "Windows, Linux, macOS, Android, iOS are popular OS examples.",
    
    # CPU Scheduling
    "cpu scheduling": "CPU scheduling decides which process runs next to optimize CPU usage.",
    "scheduling algorithms": "FCFS, SJF, Round Robin, Priority, Multilevel Queue, and Multilevel Feedback Queue.",
    "fcfs": "FCFS executes tasks in arrival order. Simple but causes long wait times.",
    "sjf": "SJF chooses the shortest burst time first. Minimizes waiting time but hard to predict burst time.",
    "preemptive sjf": "Also called SRTF — process with shortest remaining time gets CPU.",
    "round robin": "RR gives each process equal time slices. Great for time-sharing systems.",
    "priority scheduling": "CPU goes to highest-priority task. Lower priority tasks may starve.",
    "aging in os": "Aging prevents starvation by gradually increasing process priority.",
    "mlfq": "Multilevel Feedback Queue combines multiple queues and moves processes between them based on behavior.",
    
    # Process & Threads
    "process": "A program in execution with its own memory and resources.",
    "thread": "Lightweight process sharing memory with parent process.",
    "difference between process and thread": "Process has its own memory; thread shares memory with others in same process.",
    "pcb": "PCB (Process Control Block) stores process information like registers, memory, and states.",
    "process states": "New, Ready, Running, Waiting, Terminated.",
    "context switching": "Saving CPU state of a running process and loading another. Adds overhead.",
    
    # Synchronization
    "critical section": "Section of code accessing shared resources — must avoid race conditions.",
    "race condition": "Occurs when multiple processes access shared data simultaneously leading to inconsistency.",
    "semaphore": "Semaphore controls access to resources via signaling (wait, signal).",
    "mutex": "A mutex is a lock allowing one thread to access a resource at a time.",
    "difference between mutex and semaphore": "Mutex = single lock; Semaphore = signaling mechanism, allows multiple access levels.",
    "deadlock": "Processes wait forever due to circular resource holding.",
    "deadlock conditions": "Mutual exclusion, hold & wait, no preemption, circular wait.",
    "deadlock prevention": "Break one of the 4 deadlock conditions.",
    "deadlock avoidance": "Banker's Algorithm checks for safe state.",
    
    # Memory Management
    "memory management": "Manages RAM allocation, deallocation, and protection.",
    "paging": "Memory divided into fixed-size pages and frames.",
    "page table": "Maps logical pages to physical frames.",
    "segmentation": "Memory division based on program structure (variable size segments).",
    "virtual memory": "Lets processes run even if not fully in RAM using disk space.",
    "tlb": "Translation Lookaside Buffer stores recent page table entries for fast lookup.",
    "fragmentation": "Memory waste. Internal fragmentation = inside allocated block; External = scattered small free blocks.",
    "thrashing": "Too much paging → low CPU usage and slow system.",
    
    # Page Replacement
    "page replacement algorithms": "FIFO, LRU, Optimal, Clock, LFU.",
    "fifo": "Replaces oldest loaded page. Simple but suffers Belady's anomaly.",
    "lru": "Least Recently Used — predicts future based on past usage.",
    "optimal replacement": "Replaces page not needed for longest future time (theoretically best).",
    "clock algorithm": "Second-chance FIFO using circular pointer.",
    
    # File System
    "file system": "Manages how data is stored and accessed on disk.",
    "file allocation methods": "Contiguous, Linked, Indexed Allocation.",
    "directory structure": "Single-level, Two-level, Tree-structured, Acyclic, General graph.",
    "inode": "Linux structure storing file metadata, not filename.",
    
    # I/O & Devices
    "device driver": "Software that helps OS communicate with hardware.",
    "spooling": "Buffering data for devices like printers while CPU continues work.",
    "dma": "Direct Memory Access lets devices transfer data to memory without CPU intervention.",
    
    # Security
    "os security": "Protects system from unauthorized access and attacks.",
    "authentication vs authorization": "Authentication verifies identity; authorization grants access rights.",
    
    # Linux / Commands
    "linux": "Open-source UNIX-like OS widely used for servers and programming.",
    "kernel vs shell": "Kernel manages hardware; Shell is command interface.",
    "common linux commands": "ls, cd, mkdir, rm, ps, grep, chmod, top",
    
    # Misc
    "system call": "Interface between user programs and OS kernel.",
    "bootstrap": "Bootstrapping loads OS from disk to RAM at startup.",
    "interrupt": "Signal that temporarily stops CPU to handle events."
}

        # --- MATCH USER INPUT ---
        response = None
        for key, value in responses.items():
            if re.search(key, user_msg):
                response = value
                break

        if not response:
            response = (
                "I'm still learning! Please ask something related to Operating System concepts "
                "(like CPU scheduling, memory management, deadlock, or paging)."
            )

        return JsonResponse({"response": response})

    except Exception as e:
        print("Chatbot error:", e)
        return JsonResponse({"response": "⚠️ Something went wrong on the server."})
    
