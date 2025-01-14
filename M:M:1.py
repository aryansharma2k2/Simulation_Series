import math
import random
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare

# Random Number Generator 
class RandomNumberGenerator:
    def __init__(self, seed):
        self.idum = seed
        self.MASK = 123459876
        self.IA = 16807
        self.IM = 2147483647
        self.AM = 1.0 / self.IM
        self.IQ = 127773
        self.IR = 2836

    def rand(self):
        self.idum ^= self.MASK
        k = self.idum // self.IQ
        self.idum = self.IA * (self.idum - k * self.IQ) - self.IR * k
        if self.idum < 0:
            self.idum += self.IM
        return self.AM * self.idum

    def expdev(self, rate):
        while True:
            dummy = self.rand()
            if dummy != 0.0:
                return -math.log(dummy) / rate

# Represents an event (arrival or departure) in the queue
class Event:
    def __init__(self, event_type, time, customer_id):
        self.event_type = event_type
        self.time = time
        self.customer_id = customer_id

# Simulates an MM1 queue (single server)
class MM1Queue:
    def __init__(self, lambda_rate, K, rng):
        self.lambda_rate = lambda_rate
        self.K = K
        self.mu = 1
        self.current_time = 0
        self.events = []
        self.queue = []
        self.served_customers = []
        self.total_waiting_time = 0
        self.total_service_time = 0
        self.customer_count = 0
        self.rng = rng

        self.schedule_event('arrival', self.rng.expdev(self.lambda_rate))

    def schedule_event(self, event_type, time):
        event = Event(event_type, time, self.customer_count)
        self.events.append(event)

    def run(self):
        while len(self.served_customers) < self.K:
            self.events.sort(key=lambda x: x.time)
            next_event = self.events.pop(0)
            self.current_time = next_event.time

            if next_event.event_type == 'arrival':
                self.process_arrival(next_event.customer_id)
            elif next_event.event_type == 'departure':
                self.process_departure(next_event.customer_id)

    def process_arrival(self, customer_id):
        self.customer_count += 1
        self.queue.append(self.current_time)
        self.schedule_event('arrival', self.current_time + self.rng.expdev(self.lambda_rate))

        if len(self.queue) == 1:
            service_time = self.rng.expdev(self.mu)
            self.total_service_time += service_time
            self.schedule_event('departure', self.current_time + service_time)

    def process_departure(self, customer_id):
        arrival_time = self.queue.pop(0)
        service_time = self.rng.expdev(self.mu)
        
        waiting_time = self.current_time - arrival_time
        self.total_waiting_time += waiting_time
        self.total_service_time += service_time
        self.served_customers.append((arrival_time, service_time, self.current_time))

        if self.queue:
            service_time = self.rng.expdev(self.mu)
            self.schedule_event('departure', self.current_time + service_time)

    def print_results(self, L):
        avg_service_time = self.total_service_time / self.K
        avg_waiting_time = self.total_waiting_time / self.K
        print(f"\nSimulation Results for lambda = {self.lambda_rate}, K = {self.K}")
        print(f"Final clock time = {self.current_time}")
        print(f"Average service time = {avg_service_time}")
        print(f"Average waiting time = {avg_waiting_time}")

        for idx in [L, L + 1, L + 10, L + 11]:
            if idx < len(self.served_customers):
                arrival_time, service_time, departure_time = self.served_customers[idx]
                print(f"Customer {idx}: Arrival = {arrival_time}, Service = {service_time}, Departure = {departure_time}")

# Chi-square test to assess RNG 
def chi_square_test(samples, bins=10):
    observed_freq, _ = np.histogram(samples, bins=bins, range=(0, 1))
    expected_freq = len(samples) / bins
    chi2_stat, p_value = chisquare(observed_freq, f_exp=[expected_freq] * bins)
    
    print(f"\nChi-square statistic: {chi2_stat}")
    print(f"P-value: {p_value}")

    if p_value > 0.05:
        print("The RNG appears to be uniform.")
    else:
        print("The RNG does not appear to be uniform.")

# Plotting average waiting and system times
def plot_average_times(rho_values, avg_waiting_times, avg_system_times, K):
    plt.figure(figsize=(8, 6))
    plt.plot(rho_values, avg_waiting_times, label='Average Waiting Time', marker='o')
    plt.plot(rho_values, avg_system_times, label='Average System Time', marker='x')
    plt.xlabel('rho (Traffic Intensity)')
    plt.ylabel('Time')
    plt.title(f'Average Waiting and System Times for K = {K}')
    plt.grid()
    plt.legend()
    plt.show()

# Compare simulated and analytical results in plots
def plot_comparison(rho_values, sim_times, analytical_times, title, ylabel):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(rho_values, sim_times[0], label='Simulated Waiting Time', marker='o')
    plt.plot(rho_values, analytical_times[0], label='Analytical Waiting Time', linestyle='--', color='red')
    plt.xlabel('rho (Traffic Intensity)')
    plt.ylabel(ylabel)
    plt.title(title + ' - Waiting Time')
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(rho_values, sim_times[1], label='Simulated System Time', marker='o')
    plt.plot(rho_values, analytical_times[1], label='Analytical System Time', linestyle='--', color='red')
    plt.xlabel('rho (Traffic Intensity)')
    plt.ylabel(ylabel)
    plt.title(title + ' - System Time')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

# Measure running time for simulations
def measure_running_time(rho_values, K):
    running_times = []
    for rho in rho_values:
        start_time = time.time()
        rng = RandomNumberGenerator(int(time.time()))
        mm1_queue = MM1Queue(rho, K, rng)
        mm1_queue.run()
        end_time = time.time()
        running_times.append(end_time - start_time)
    return running_times

def main():
    if len(sys.argv) != 4:
        print("Usage: python proj1.py <lambda> <K> <L>")
        return

    lambda_rate = float(sys.argv[1])
    K = int(sys.argv[2])
    L = int(sys.argv[3])

    rng = RandomNumberGenerator(int(time.time()))
    mm1_queue = MM1Queue(lambda_rate, K, rng)
    mm1_queue.run()
    mm1_queue.print_results(L)

#FOR TASKS 
    # # Average waiting and system times for different rho values
    # rho_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # sim_waiting_times, sim_system_times = [], []
    # K1=1000
    # K2=100000
    # for rho in rho_values:
    #     mm1_queue = MM1Queue(rho, K1, rng)
    #     mm1_queue.run()
    #     avg_waiting = mm1_queue.total_waiting_time / K1
    #     avg_service = mm1_queue.total_service_time / K1
    #     sim_waiting_times.append(avg_waiting)
    #     sim_system_times.append(avg_waiting + avg_service)

    # plot_average_times(rho_values, sim_waiting_times, sim_system_times, K1)
    # sim_waiting_times, sim_system_times = [], []
    # for rho in rho_values:
    #     mm1_queue = MM1Queue(rho, K2, rng)
    #     mm1_queue.run()
    #     avg_waiting = mm1_queue.total_waiting_time / K2
    #     avg_service = mm1_queue.total_service_time / K2
    #     sim_waiting_times.append(avg_waiting)
    #     sim_system_times.append(avg_waiting + avg_service)

    # plot_average_times(rho_values, sim_waiting_times, sim_system_times, K2)

    # analytical_waiting_times = [rho / (1 - rho) for rho in rho_values]
    # analytical_system_times = [1 / (1 - rho) for rho in rho_values]

    # plot_comparison(rho_values, (sim_waiting_times, sim_system_times), (analytical_waiting_times, analytical_system_times), 'Comparison', 'Time')

    # running_times = measure_running_time(rho_values, K)
    # plt.figure(figsize=(8, 6))
    # plt.plot(rho_values, running_times, marker='o')
    # plt.xlabel('rho (Traffic Intensity)')
    # plt.ylabel('Running Time (seconds)')
    # plt.title('Running Time vs rho')
    # plt.grid()
    # plt.show()

    # rng_samples = [rng.rand() for _ in range(1000)]
    # chi_square_test(rng_samples)

if __name__ == '__main__':
    main()
