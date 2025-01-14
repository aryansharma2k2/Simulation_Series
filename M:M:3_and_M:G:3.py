import math
import random
import sys
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt


class RandomNumberGenerator:
    def __init__(self, seed):
        self.state = seed
        self.MASK = 123459876
        self.A = 16807
        self.M = 2147483647
        self.AM = 1.0 / self.M
        self.Q = 127773
        self.R = 2836

    def rand(self):
        self.state ^= self.MASK
        quotient = self.state // self.Q
        self.state = self.A * (self.state - quotient * self.Q) - self.R * quotient
        if self.state < 0:
            self.state += self.M
        return self.AM * self.state

    def expdev(self, rate):
        while True:
            random_value = self.rand()
            if random_value != 0.0:
                return -math.log(random_value) / rate

    def pareto(self, k, alpha):
        while True:
            u = self.rand()
            if u != 0.0:
                return k * (1 - u) ** (-1 / alpha)


class Customer:
    def __init__(self, arrival_time):
        self.arrival_time = arrival_time
        self.service_time = 0
        self.waiting_time = 0
        self.finish_time = 0


class Event:
    def __init__(self, event_type, time, customer_id):
        self.event_type = event_type
        self.time = time
        self.customer_id = customer_id


class QueueSimulation:
    def __init__(self, arrival_rate, total_customers, service_rate, system_type, k=None, alpha=None, discipline=0):
        self.arrival_rate = arrival_rate
        self.total_customers = total_customers
        self.service_rate = service_rate
        self.system_type = system_type
        self.k = k
        self.alpha = alpha
        self.rng = RandomNumberGenerator(seed=12345)
        self.discipline = discipline
        self.current_time = 0
        self.queue = []
        self.served_customers = []
        self.events = []
        self.servers = [None, None, None]  # 3 servers
        self.total_waiting_time = 0
        self.total_service_time = 0
        self.customer_count = 0
        self.schedule_event('arrival', self.rng.expdev(self.arrival_rate))  # First arrival event

    def run_simulation(self):
        """Run the simulation until K customers are served."""
        while len(self.served_customers) < self.total_customers:
            self.events.sort(key=lambda x: x.time)  # Sort events by time
            next_event = self.events.pop(0)
            self.current_time = next_event.time

            if next_event.event_type == 'arrival':
                self.process_arrival(next_event.customer_id)
            elif next_event.event_type == 'departure':
                self.process_departure()

        self.display_results()

    def process_arrival(self, customer_id):
        customer = Customer(self.current_time)
        self.customer_count += 1
        self.queue.append(customer)
        self.schedule_event('arrival', self.current_time + self.rng.expdev(self.arrival_rate))  # Schedule next arrival
        self.attempt_service_start()

    def process_departure(self):
        for i in range(3):
            if self.servers[i] is not None:
                customer = self.servers[i]
                service_time = self.get_service_time()
                waiting_time = self.current_time - customer.arrival_time
                self.total_waiting_time += waiting_time
                self.total_service_time += service_time
                customer.finish_time = self.current_time + service_time
                self.served_customers.append(customer)

                self.servers[i] = None

                if customer in self.queue:
                    self.queue.remove(customer)

                self.attempt_service_start()
                return

    def get_service_time(self):
        if self.system_type == 0:  # M/M/3 system
            return self.rng.expdev(self.service_rate)
        else:  # M/G/3 system (Pareto distributed service time)
            return self.rng.pareto(self.k, self.alpha)

    def attempt_service_start(self):
        if self.discipline == 1:  # Sort by service time for Shortest Job First 
            self.queue.sort(key=lambda x: x.service_time)

        for i in range(3):
            if self.servers[i] is None and self.queue:
                customer = self.queue.pop(0)
                self.servers[i] = customer
                service_time = self.get_service_time()
                customer.service_time = service_time
                self.schedule_event('departure', self.current_time + service_time)
                return

    def schedule_event(self, event_type, time):
        event = Event(event_type, time, self.customer_count)
        self.events.append(event)

    def display_results(self):
        served_count = len(self.served_customers)
        if served_count == 0:
            print("No customers were served.")
            return

        avg_waiting_time = self.total_waiting_time / served_count
        avg_service_time = self.total_service_time / served_count
        avg_system_time = avg_waiting_time + avg_service_time

        waiting_times = [cust.waiting_time for cust in self.served_customers]
        service_times = [cust.service_time for cust in self.served_customers]
        confidence_interval = self.calculate_confidence_interval(waiting_times, service_times)

        # Fixing np.float64 issue by converting to float
        waiting_interval = (float(confidence_interval[0]), float(confidence_interval[1]))
        service_interval = (float(confidence_interval[2]), float(confidence_interval[3]))

        print(f"Lambda (λ): {self.arrival_rate}")
        print(f"K: {self.total_customers}")
        print(f"Final clock time: {self.current_time:.2f}")
        print(f"Average system time: {avg_system_time:.2f}")
        print(f"Average waiting time: {avg_waiting_time:.2f}")
        print(f"Confidence Intervals (for waiting time): {waiting_interval}")
        print(f"Average service time: {avg_service_time:.2f}")
        print(f"Confidence Intervals (for service time): {service_interval}")


    def calculate_confidence_interval(self, waiting_times, service_times):
        avg_waiting = np.mean(waiting_times)
        avg_service = np.mean(service_times)

        std_waiting = np.std(waiting_times, ddof=1)
        std_service = np.std(service_times, ddof=1)

        t_value = t.ppf(0.975, len(waiting_times) - 1)

        ci_waiting = t_value * (std_waiting / np.sqrt(len(waiting_times)))
        ci_service = t_value * (std_service / np.sqrt(len(service_times)))

        return (avg_waiting - ci_waiting, avg_waiting + ci_waiting, avg_service - ci_service, avg_service + ci_service)


def main():
    # Simulation parameters
    lambda_rate = 0.0001  # Arrival rate (λ)
    total_customers = 50000  # Number of customers to process
    discipline = 0  # 0 for FCFS, 1 for SJF
    system_type = 1  # 0 for M/M/3, 1 for M/G/3
    k = 332  # Minimum for Pareto distribution (if M/G/3)
    alpha = 1.1  # Shape parameter for Pareto distribution (if M/G/3)
    
    rng = RandomNumberGenerator(seed=12345)
    simulation = QueueSimulation(lambda_rate, total_customers, 1 / 3000, system_type, k, alpha, discipline)
    simulation.run_simulation()


if __name__ == "__main__":
    main()