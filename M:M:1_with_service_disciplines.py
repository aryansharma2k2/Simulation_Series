import math
import random
import sys
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

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

class Event:
    def __init__(self, event_type, time, customer_id, disk_id=None):
        self.event_type = event_type
        self.time = time
        self.customer_id = customer_id
        self.disk_id = disk_id  # For I/O disk events

class Customer:
    def __init__(self, arrival_time, priority):
        self.arrival_time = arrival_time
        self.priority = priority
        self.service_time = 0
        self.queue_times = []  # Track waiting times for each queue visited

class MM1Queue:
    def __init__(self, lambda_rate, K, rng, discipline):
        self.lambda_rate = lambda_rate
        self.K = K
        self.mu = 1
        self.current_time = 0
        self.events = []
        self.queues = {1: [], 2: [], 3: [], 4: []}  # Queues for each priority
        self.served_customers = []
        self.total_waiting_time = 0
        self.total_service_time = 0
        self.customer_count = 0
        self.rng = rng
        self.discipline = discipline
        self.schedule_event('arrival', self.rng.expdev(self.lambda_rate))

    def run(self):
        while len(self.served_customers) < self.K:
            self.events.sort(key=lambda x: x.time)
            next_event = self.events.pop(0)
            self.current_time = next_event.time

            if next_event.event_type == 'arrival':
                self.process_arrival(next_event.customer_id)
            elif next_event.event_type == 'departure':
                self.process_departure()

    def process_arrival(self, customer_id):
        priority = self.assign_priority()
        customer = Customer(self.current_time, priority)
        self.customer_count += 1
        self.queues[priority].append(customer)
        self.schedule_event('arrival', self.current_time + self.rng.expdev(self.lambda_rate))

        # Attempt to start service if there's at least one customer in any queue
        self.start_service()

    def process_departure(self):
        customer = None

        # Determine which customer to serve based on the discipline
        if self.discipline == 1:  # FCFS
            for p in range(1, 5):
                if self.queues[p]:  # Check for non-empty queue
                    customer = self.queues[p].pop(0)  # Serve the first customer
                    break
        elif self.discipline == 2:  # LCFS
            for p in range(1, 5):
                if self.queues[p]:  # Check for non-empty queue
                    customer = self.queues[p].pop()  # Serve the last customer (LIFO)
                    break
        elif self.discipline == 3:  # SJF
            shortest_job = None
            for p in range(1, 5):
                if self.queues[p]:  # Check for non-empty queue
                    # Find the shortest job in the queue
                    for candidate in self.queues[p]:
                        if (shortest_job is None or 
                            candidate.service_time < shortest_job.service_time):
                            shortest_job = candidate
            if shortest_job:
                # Remove the selected shortest job from its queue
                self.queues[shortest_job.priority].remove(shortest_job)
                customer = shortest_job
        elif self.discipline in [4, 5]:  # Priority queues (non-preemptive and preemptive)
            for p in range(1, 5):
                if self.queues[p]:  # Check for non-empty queue
                    customer = self.queues[p].pop(0)  # Serve the highest priority
                    break
        else:
            raise ValueError("Unknown service discipline")

        if customer is not None:  # Check if customer was assigned
            service_time = self.rng.expdev(self.mu)  # Calculate service time
            waiting_time = self.current_time - customer.arrival_time
            self.total_waiting_time += waiting_time
            self.total_service_time += service_time
            self.served_customers.append((customer.arrival_time, service_time, self.current_time))

            # Schedule the next departure if there are still customers waiting
            if any(self.queues[p] for p in range(1, 5)):
                self.schedule_event('departure', self.current_time + service_time)
            # Start service for the next customer
            self.start_service()
        else:
            pass

    def assign_priority(self):
        rand_value = random.random()
        if rand_value < 0.25:
            return 1
        elif rand_value < 0.5:
            return 2
        elif rand_value < 0.75:
            return 3
        else:
            return 4

    def start_service(self):
        # Start service for the next customer in the highest priority queue
        for p in range(1, 5):
            if self.queues[p]:  # Check if there's a customer to serve
                customer = self.queues[p][0]  # Get the first customer in the highest priority queue
                service_time = self.rng.expdev(self.mu)
                self.schedule_event('departure', self.current_time + service_time)
                return  # Exit once service has started

    def schedule_event(self, event_type, time):
        event = Event(event_type, time, self.customer_count)
        self.events.append(event)

    def print_results(self, L):
        if not self.served_customers:
            print("No customers were served.")
            return

        avg_waiting_time = self.total_waiting_time / len(self.served_customers)
        avg_service_time = self.total_service_time / len(self.served_customers)
        avg_system_time = avg_waiting_time + avg_service_time

        # Collect service times and waiting times for confidence interval calculations
        waiting_times = [self.current_time - arrival_time for arrival_time, _, _ in self.served_customers]
        service_times = [service_time for _, service_time, _ in self.served_customers]

        # Function to calculate confidence interval
        def calculate_ci(data):
            n = len(data)
            if n < 2:  # Require at least two data points for CI
                return 0, 0  # Return placeholder values
            mean = np.mean(data)
            stddev = np.std(data, ddof=1)  # Sample standard deviation
            t_value = t.ppf(0.975, n - 1)  # 95% CI
            margin_of_error = t_value * (stddev / np.sqrt(n))
            return mean - margin_of_error, mean + margin_of_error

        # Calculate confidence intervals for waiting time, service time, and system time
        ci_waiting_time = calculate_ci(waiting_times)
        ci_service_time = calculate_ci(service_times)
        ci_system_time = calculate_ci([wt + st for wt, st in zip(waiting_times, service_times)])

        # Print only the required results
        print(f"Lambda (λ): {self.lambda_rate}")
        print(f"K: {self.K}")
        print(f"Final clock time: {self.current_time:.2f}")
        print(f"Average system time: {avg_system_time:.2f} (95% CI: {ci_system_time})")
        print(f"Average waiting time: {avg_waiting_time:.2f} (95% CI: {ci_waiting_time})")

class WebServerIO:
    def __init__(self, lambda_rate, K, rng):
        self.lambda_rate = lambda_rate
        self.K = K
        self.mu_CPU = 1.0
        self.mu_io = 5.0  
        self.current_time = 0
        self.events = []
        self.cpu_queue = []
        self.io_queues = {1: [], 2: [], 3: []}  # I/O disk queues
        self.served_customers = []
        self.total_waiting_time = 0
        self.total_service_time = 0
        self.customer_count = 0
        self.rng = rng
        self.schedule_event('arrival', self.rng.expdev(self.lambda_rate))

    def run(self):
        while len(self.served_customers) < self.K:
            self.events.sort(key=lambda x: x.time)
            next_event = self.events.pop(0)
            self.current_time = next_event.time

            if next_event.event_type == 'arrival':
                self.process_arrival(next_event.customer_id)
            elif next_event.event_type == 'departure_cpu':
                self.process_departure_cpu()
            elif next_event.event_type == 'departure_io':
                self.process_departure_io(next_event.disk_id)

    def process_arrival(self, customer_id):
        priority = self.assign_priority()
        customer = Customer(self.current_time, priority)
        self.customer_count += 1
        self.cpu_queue.append(customer)
        self.schedule_event('arrival', self.current_time + self.rng.expdev(self.lambda_rate))

        # Attempt to start service if there's at least one customer in the CPU queue
        self.start_service_cpu()

    def process_departure_cpu(self):
        if self.cpu_queue:
            customer = self.cpu_queue.pop(0)  # Serve the first customer (FCFS)
            service_time = self.rng.expdev(self.mu_CPU)
            waiting_time = self.current_time - customer.arrival_time
            customer.queue_times.append(waiting_time)
            self.total_waiting_time += waiting_time
            self.total_service_time += service_time
            self.served_customers.append((customer.arrival_time, service_time, self.current_time))
            
            # Determine where the customer goes next
            if random.random() < 0.1:  # Probability to leave the system
                pass  # Customer leaves, do not schedule next event
            else:
                disk_choice = self.assign_io_disk()
                self.io_queues[disk_choice].append(customer)
                self.schedule_event('departure_io', self.current_time + self.rng.expdev(1 / self.mu_io), disk_id=disk_choice)

            # Schedule next CPU departure if the CPU queue is not empty
            self.start_service_cpu()

    def process_departure_io(self, disk_id):
        if self.io_queues[disk_id]:
            customer = self.io_queues[disk_id].pop(0)  # Serve the first customer in the selected I/O disk queue
            service_time = self.rng.expdev(1 / self.mu_io)  # I/O service time
            self.total_service_time += service_time

            # After finishing with I/O, send customer back to CPU
            self.cpu_queue.append(customer)
            # Attempt to start service for the next customer in the CPU queue
            self.start_service_cpu()

    def assign_io_disk(self):
        # Randomly assign to one of the I/O disks based on defined probabilities
        rand_value = random.random()
        if rand_value < 0.3:
            return 1
        elif rand_value < 0.6:
            return 2
        else:
            return 3

    def start_service_cpu(self):
        if self.cpu_queue:  # Check if there's a customer to serve
            customer = self.cpu_queue[0]  # Get the first customer in the CPU queue
            service_time = self.rng.expdev(self.mu_CPU)
            self.schedule_event('departure_cpu', self.current_time + service_time)

    def schedule_event(self, event_type, time, disk_id=None):
        event = Event(event_type, time, self.customer_count, disk_id)
        self.events.append(event)

    def print_results(self, L):
        if not self.served_customers:
            print("No customers were served.")
            return

        avg_waiting_time = self.total_waiting_time / len(self.served_customers)
        avg_service_time = self.total_service_time / len(self.served_customers)
        avg_system_time = avg_waiting_time + avg_service_time

        # Calculate waiting times for confidence interval
        waiting_times = [self.current_time - arrival_time for arrival_time, _, _ in self.served_customers]
        
        # Calculate system times for confidence interval
        system_times = [waiting_time + service_time for _, service_time, waiting_time in self.served_customers]

        # Function to calculate confidence interval
        def calculate_ci(data):
            n = len(data)
            if n < 2:  # Require at least two data points for CI
                return 0, 0  # Return placeholder values
            mean = np.mean(data)
            stddev = np.std(data, ddof=1)  # Sample standard deviation
            t_value = t.ppf(0.975, n - 1)  # 95% CI
            margin_of_error = t_value * (stddev / np.sqrt(n))
            return mean - margin_of_error, mean + margin_of_error

        ci_waiting_time = calculate_ci(waiting_times)
        ci_system_time = calculate_ci(system_times)

        print(f"Lambda (λ): {self.lambda_rate}")
        print(f"K: {self.K}")
        print(f"Final clock time: {self.current_time:.2f}")
        print(f"Average waiting time: {avg_waiting_time:.2f} (95% CI: {ci_waiting_time})")
        print(f"Average system time: {avg_system_time:.2f} (95% CI: {ci_system_time})")


    def assign_priority(self):
        # Define priority assignment logic
        return random.randint(1, 4)  # Assign a random priority from 1 to 4

def main():
    if len(sys.argv) != 5:
        print("Usage: python proj2.py <lambda> <K> <L> <discipline>")
        return

    lambda_rate = float(sys.argv[1])
    K = int(sys.argv[2])
    L = int(sys.argv[3])
    discipline = int(sys.argv[4])
    if L==1:
        rng = RandomNumberGenerator(seed=12345)
        queue = WebServerIO(lambda_rate, K, rng)
        queue.run()
        queue.print_results(L)
        
    else:
        rng = RandomNumberGenerator(seed=12345)
        queue = MM1Queue(lambda_rate, K, rng, discipline)
        queue.run()
        queue.print_results(L)


if __name__ == "__main__":
    main()