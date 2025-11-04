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
        responses = {
            "what is operating system": "An Operating System (OS) is system software that manages computer hardware, software resources, and provides services for computer programs.",
            "functions of operating system": "Main functions of an OS include process management, memory management, file management, device management, and security.",
            "types of operating system": "Common types: Batch OS, Time-Sharing OS, Distributed OS, Real-Time OS, and Embedded OS.",
            "cpu scheduling": "CPU Scheduling is the process of selecting which process gets to use the CPU next to improve CPU utilization and system response.",
            "scheduling algorithms": "Common CPU scheduling algorithms include FCFS (First Come First Serve), SJF (Shortest Job First), Priority Scheduling, and Round Robin.",
            "fcfs": "FCFS (First Come First Serve) is the simplest scheduling algorithm where the process that arrives first is executed first.",
            "sjf": "SJF (Shortest Job First) schedules the process with the smallest burst time next. It can be preemptive or non-preemptive.",
            "priority scheduling": "In Priority Scheduling, each process is assigned a priority. The CPU is allocated to the process with the highest priority.",
            "round robin": "Round Robin scheduling gives each process a fixed time slot (quantum) in a cyclic order, ensuring fairness.",
            "deadlock": "A deadlock occurs when a set of processes are blocked because each process is holding a resource and waiting for another.",
            "deadlock conditions": "Four necessary conditions for deadlock: Mutual Exclusion, Hold and Wait, No Preemption, and Circular Wait.",
            "memory management": "Memory management handles allocation and deallocation of main memory for processes.",
            "paging": "Paging divides memory into fixed-size blocks called pages (logical memory) and frames (physical memory). It helps in efficient memory utilization.",
            "segmentation": "Segmentation divides memory into variable-sized segments based on program logic like functions, arrays, etc.",
            "virtual memory": "Virtual Memory allows execution of processes not completely in main memory by using secondary storage as an extension of RAM.",
            "page replacement algorithms": "Examples: FIFO, LRU, Optimal Page Replacement — used to decide which page to remove when new pages are needed.",
            "banker's algorithm": "Banker's Algorithm is used to avoid deadlock by checking the safe state before allocating resources.",
            "thrashing": "Thrashing occurs when excessive paging reduces CPU performance due to constant swapping of pages.",
            "semaphore": "A semaphore is a synchronization tool used to solve critical section problems and avoid race conditions.",
            "critical section": "A critical section is a code segment where shared resources are accessed. Synchronization is used to prevent conflicts.",
            "file system": "The file system manages how data is stored and retrieved on disks. Examples: FAT32, NTFS, EXT4.",
            "os examples": "Examples of operating systems: Windows, Linux, macOS, Android, and iOS."
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
    
