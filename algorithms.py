import numpy as np
from scipy.optimize import minimize
import math
import random

# Number of vehicles
r = 200  # RSU radius
e = 160 #distance to road
w_uu = 2  # Cellular mode's channel bandwidth (e.g., MHz)
uplink_channel = 5  # Number of uplink channels available
sigma_uu = np.random.uniform(0.9, 1.1)  # Noise factor (varies slightly around 1)
gamma = 2  # Path loss exponent (common value for urban areas)
num_vehicles = 10  # Number of vehicles in the network

# Utility function weights
energy_weight = 1  # Weight given to energy consumption in the utility function
time_weight = 0.8  # Weight given to time (latency) in the utility function
# Function to calculate local computation time
def local_computation_time(C, f_loc):
    return C / f_loc

# Function to calculate local energy consumption
def local_energy_consumption(C, vehicle_idx):
    return 0.5 * C  # Energy consumption per unit

# Function to calculate VEC computation time
def VEC_computation_time(C, f_vec, data_size, transmission_rate, latency, t_ptd):
    return C / f_vec + data_size / transmission_rate + latency * t_ptd

# Function to calculate VEC energy consumption
def VEC_energy_consumption(data_size, transmission_rate, transmission_power=0.5):
    return transmission_power * data_size / transmission_rate

# Function to calculate local utility
def utility_local(C, vehicle_idx, f_loc):
    return np.log(1 + (energy_weight * local_energy_consumption(C, vehicle_idx) + time_weight * local_computation_time(C, f_loc)))

# Function to calculate VEC utility
def utility_VEC(C, vehicle_idx, f_vec, data_size, transmission_rate, latency, t_ptd):
    return np.log(1 + (energy_weight * VEC_energy_consumption(data_size, transmission_rate) + time_weight * VEC_computation_time(C, f_vec, data_size, transmission_rate, latency, t_ptd))) - 0.5 * f_vec



# Function to run MACTER algorithm
def MACTER_algorithm(num_tests):
    results_summary = {
        "MACTER Offloading Decisions": [],
        "MACTER VEC Resources Allocation": [],
        "Computation Efficiency": [],
        "Computation Time": [],
        "Energy Consumption": [],
        "Resource Utilization": []
    }

    for _ in range(num_tests):
        # Adjusting parameters for each test case
        task_data_size = np.random.uniform(150, 300, num_vehicles)  # More computationally intensive tasks
        vehicle_computation_capacity = np.random.uniform(0.5, 2, num_vehicles)  # Further reduced local computation capacity
        service_coefficient = vehicle_computation_capacity / task_data_size  # Recompute service coefficient
        position = [i * 10 for i in range(num_vehicles)]
        transition_power = [random.randint(30, 50) for i in range(num_vehicles)]
        distance_traveled = [(r * math.ceil(position[i] / 3)) - position[i] for i in range(num_vehicles)]
        transmission_rate = []  # Variable transmission rates
        for i in range(num_vehicles):
            if distance_traveled[i] != 0:
                a = math.pow(distance_traveled[i], -gamma)
            else:
                a = 1
            b = ((transition_power[i] * a * uplink_channel) / sigma_uu)
            transmission_rate.append(w_uu * math.log2(1 + b))

        VEC_computation_capacity = np.random.uniform(20, 30)  # Increased VEC capacity
        network_latency = np.random.uniform(0.1, 0.5, num_vehicles)  # Simulate higher network latencies
        t_max = np.random.uniform(10, 20, num_vehicles)
        speed = np.random.uniform(30, 60, num_vehicles)  # Simulate realistic traffic patterns
        t_stay = [2 * (math.sqrt((r * r) - (e * e))) / speed[i] for i in range(num_vehicles)]
        t_ptd = [min(t_stay[i], t_max[i]) for i in range(num_vehicles)]

        # MACTER Algorithm Implementation
        offloading_decisions = np.zeros(num_vehicles)
        VEC_resources = np.zeros(num_vehicles)
        total_computation_time = 0
        total_energy_consumption = 0

        for iteration in range(10):  # Max iterations
            # Step 1: Resource Allocation
            def objective(f_vec):
                total_utility = 0
                for i in range(num_vehicles):
                    if offloading_decisions[i] == 1:
                        C = service_coefficient[i] * task_data_size[i]
                        data_size = task_data_size[i]
                        total_utility += utility_VEC(C, i, f_vec[i], data_size, transmission_rate[i], network_latency[i], t_ptd[i])
                return -total_utility

            constraints = ({'type': 'eq', 'fun': lambda f_vec: np.sum(f_vec) - VEC_computation_capacity})
            bounds = [(0, VEC_computation_capacity) for _ in range(num_vehicles)]
            result = minimize(objective, VEC_resources, method='SLSQP', bounds=bounds, constraints=constraints)
            VEC_resources = result.x

            # Step 2: Offloading Decision
            for i in range(num_vehicles):
                C = service_coefficient[i] * task_data_size[i]
                f_loc = vehicle_computation_capacity[i]
                data_size = task_data_size[i]

                utility_loc = utility_local(C, i, f_loc)
                utility_vec = utility_VEC(C, i, VEC_resources[i], data_size, transmission_rate[i], network_latency[i], t_ptd[i])
                if utility_vec > utility_loc:
                    offloading_decisions[i] = 1
                    total_computation_time += VEC_computation_time(C, VEC_resources[i], data_size, transmission_rate[i], network_latency[i], t_ptd[i])
                    total_energy_consumption += VEC_energy_consumption(data_size, transmission_rate[i])
                else:
                    offloading_decisions[i] = 0
                    total_computation_time += local_computation_time(C, f_loc)
                    total_energy_consumption += local_energy_consumption(C, i)

        computation_efficiency = total_computation_time / total_energy_consumption if total_energy_consumption != 0 else 0
        results_summary["MACTER Offloading Decisions"].append(offloading_decisions)
        results_summary["MACTER VEC Resources Allocation"].append(VEC_resources)
        results_summary["Computation Efficiency"].append(computation_efficiency)
        results_summary["Computation Time"].append(total_computation_time / num_vehicles)
        results_summary["Energy Consumption"].append(total_energy_consumption / num_vehicles)
        results_summary["Resource Utilization"].append(np.sum(VEC_resources) / VEC_computation_capacity)

    return results_summary

# Function to run CTORA algorithm
def CTORA_algorithm(num_tests):
    results_summary = {
        "CTORA Offloading Decisions": [],
        "CTORA VEC Resources Allocation": [],
        "Computation Efficiency": [],
        "Computation Time": [],
        "Energy Consumption": [],
        "Resource Utilization": []
    }

    for _ in range(num_tests):
        # Adjusting parameters for each test case
        task_data_size = np.random.uniform(150, 300, num_vehicles)  # More computationally intensive tasks
        vehicle_computation_capacity = np.random.uniform(0.5, 2, num_vehicles)  # Further reduced local computation capacity
        service_coefficient = vehicle_computation_capacity / task_data_size  # Recompute service coefficient
        transmission_rate = np.random.uniform(25, 150)  # Variable transmission rates
        VEC_computation_capacity = np.random.uniform(100, 200)  # Increased VEC capacity
        network_latency = np.random.uniform(0.1, 0.5, num_vehicles)  # Simulate higher network latencies
        traffic_pattern = np.random.uniform(0.1, 1.0, num_vehicles)  # Simulate realistic traffic patterns

        # CTORA Algorithm Implementation
        offloading_decisions = np.zeros(num_vehicles)
        VEC_resources = np.full(num_vehicles, VEC_computation_capacity / num_vehicles)  # Equal allocation for simplicity
        total_computation_time = 0
        total_energy_consumption = 0

        for i in range(num_vehicles):
            C = service_coefficient[i] * task_data_size[i]
            f_loc = vehicle_computation_capacity[i]
            data_size = task_data_size[i]

            utility_loc = utility_local(C, i, f_loc)
            utility_vec = utility_VEC(C, i, VEC_resources[i], data_size, transmission_rate, network_latency[i], traffic_pattern[i])
            if utility_vec > utility_loc:
                offloading_decisions[i] = 1
                total_computation_time += VEC_computation_time(C, VEC_resources[i], data_size, transmission_rate, network_latency[i], traffic_pattern[i])
                total_energy_consumption += VEC_energy_consumption(data_size, transmission_rate)
            else:
                offloading_decisions[i] = 0
                total_computation_time += local_computation_time(C, f_loc)
                total_energy_consumption += local_energy_consumption(C, i)

        computation_efficiency = total_computation_time / total_energy_consumption if total_energy_consumption != 0 else 0
        results_summary["CTORA Offloading Decisions"].append(offloading_decisions)
        results_summary["CTORA VEC Resources Allocation"].append(VEC_resources)
        results_summary["Computation Efficiency"].append(computation_efficiency)
        results_summary["Computation Time"].append(total_computation_time / num_vehicles)
        results_summary["Energy Consumption"].append(total_energy_consumption / num_vehicles)
        results_summary["Resource Utilization"].append(np.sum(VEC_resources) / VEC_computation_capacity)

    return results_summary



# Function to run CODO algorithm
def CODO_algorithm(num_tests):
    results_summary = {
        "CODO Offloading Decisions": [],
        "CODO VEC Resources Allocation": []
    }

    for _ in range(num_tests):
        # Adjusting parameters for each test case
        task_data_size = np.random.uniform(150, 300, num_vehicles)
        vehicle_computation_capacity = np.random.uniform(0.5, 2, num_vehicles)
        service_coefficient = vehicle_computation_capacity / task_data_size
        transmission_rate = np.random.uniform(25, 150)
        VEC_computation_capacity = np.random.uniform(300, 400)
        network_latency = np.random.uniform(0.1, 0.5, num_vehicles)
        traffic_pattern = np.random.uniform(0.1, 1.0, num_vehicles)

        # CODO Algorithm Implementation
        offloading_decisions = np.zeros(num_vehicles)
        VEC_resources = np.full(num_vehicles, VEC_computation_capacity / num_vehicles)  # Initial equal allocation

        for iteration in range(10):  # Max iterations
            # Collaborative Step: Adjusting VEC resources based on offloading decisions
            for i in range(num_vehicles):
                if offloading_decisions[i] == 1:
                    C = service_coefficient[i] * task_data_size[i]
                    data_size = task_data_size[i]
                    utility_vec = utility_VEC(C, i, VEC_resources[i], data_size, transmission_rate, network_latency[i], traffic_pattern[i])
                    # Collaborative adjustment based on utility difference
                    if utility_vec > np.mean(VEC_resources):
                        VEC_resources[i] += 0.1 * (VEC_computation_capacity - VEC_resources[i])
                    else:
                        VEC_resources[i] -= 0.1 * VEC_resources[i]

            # Step 2: Offloading Decision
            for i in range(num_vehicles):
                C = service_coefficient[i] * task_data_size[i]
                f_loc = vehicle_computation_capacity[i]
                data_size = task_data_size[i]

                utility_loc = utility_local(C, i, f_loc)
                utility_vec = utility_VEC(C, i, VEC_resources[i], data_size, transmission_rate, network_latency[i], traffic_pattern[i])
                if utility_vec > utility_loc:
                    offloading_decisions[i] = 1
                else:
                    offloading_decisions[i] = 0

        results_summary["CODO Offloading Decisions"].append(offloading_decisions)
        results_summary["CODO VEC Resources Allocation"].append(VEC_resources)

    return results_summary


# Number of tests to run for each algorithm
num_tests = 5

# Running the MACTER algorithm
macter_results = MACTER_algorithm(num_tests)

# Running the CTORA algorithm
ctora_results = CTORA_algorithm(num_tests)

codo_results = CODO_algorithm(num_tests)

# Printing the results
print("MACTER Algorithm Results:")
print(macter_results)
print("\nCTORA Algorithm Results:")
print(ctora_results)
print("\nCODO Algorithm Results:")
print(codo_results)
