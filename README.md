# Queue Simulation Series: Modeling Queueing Systems and Service Disciplines

## Overview

The **Queue Simulation Series** consists of three related projects designed to model different types of queueing systems using **event-driven simulation**. These projects focus on various queuing models, from basic **M/M/1** queues to more complex systems involving **multiple servers** and **priority-based scheduling**. The goal of these simulations is to assess system performance under different conditions, service disciplines, and statistical methods.

The core approach in each project involves:
- **Event-driven simulation** to manage the sequence of arrivals, service times, and departures.
- **Statistical analysis**, including the calculation of confidence intervals for performance metrics like **waiting time**, **service time**, and **system time**.
- **Random number generation** for event timing and service time distribution using various methods such as **exponential** and **Pareto distributions**.
- **Queue disciplines** like **FCFS** (First Come, First Serve), **LCFS** (Last Come, First Serve), **SJF** (Shortest Job First), and **priority queues** to model different service behaviors.

---

## Projects

### **Project 1: Basic Queue Simulation (M/M/1 System)**

In this first project, we focused on simulating a **basic M/M/1 queue system**. The simulation modeled a single server handling customer requests with a **Poisson arrival process** (with rate \( \lambda \)) and **exponential service times** (with rate \( \mu \)).

#### Key Features:
- **Random number generation** to model arrival times and service times.
- **Event-driven simulation** to manage the sequence of arrivals, service times, and departures.
- **Statistical analysis** to calculate **average waiting times**, **average service times**, and **confidence intervals** for the performance metrics.

This simulation serves as a foundation for building more complex models, such as systems with multiple servers or priority queues.

---

### **Project 2: Priority Queues and Service Disciplines (M/M/1 Queue with Multiple Disciplines)**

Building on the basic M/M/1 model, the second project introduces **priority-based queues** and different **service disciplines**. The system allows customers to be assigned random priorities, and service can be performed based on different queueing disciplines:
- **FCFS (First Come, First Serve)**
- **LCFS (Last Come, First Serve)**
- **SJF (Shortest Job First)**
- **Priority Queues (non-preemptive and preemptive)**

#### Key Features:
- **Customer priority assignment** based on a random process.
- **Multiple service disciplines** to evaluate the effect on average waiting time, service time, and system time.
- **Confidence interval calculations** for waiting time and system time based on the served customers.
- **Random number generation** using a custom **RandomNumberGenerator** class to model exponential and priority queue behavior.

This project demonstrated the impact of service disciplines on system performance and provided insights into how different strategies affect customer wait times and overall system efficiency.

---

### **Project 3: Multi-Server Queuing System with Pareto Service Times (M/M/3 and M/G/3 Systems)**

The third project introduces a more complex queuing system with **multiple servers** and the option to simulate **Pareto-distributed service times** (M/G/3 systems). The system includes three servers and uses **Poisson arrivals** with either **exponentially distributed service times** (M/M/3) or **Pareto-distributed service times** (M/G/3). The simulation explores two key disciplines: 
- **First Come, First Serve (FCFS)**
- **Shortest Job First (SJF)**

#### Key Features:
- **Multi-server system** with three servers processing requests.
- **Pareto service time distribution** in the M/G/3 model, providing a more realistic representation of service times that are not exponentially distributed.
- **Shortest Job First (SJF)** scheduling discipline, which prioritizes jobs with the shortest service times.
- **Comprehensive statistical analysis**, including **confidence intervals** for waiting times, service times, and system times using the **t-distribution**.

The goal of this simulation was to evaluate the effects of **service time distributions** and **scheduling disciplines** on system performance. By implementing both **M/M/3** and **M/G/3** models, the project provided insights into how service time variability impacts queue dynamics and overall system efficiency.

---

## Simulation Tools and Techniques

Each of the simulations in the series used the following tools and techniques:
- **Random Number Generation:** A custom `RandomNumberGenerator` class was implemented to simulate **exponential** and **Pareto** distributions for event timings and service times.
- **Event-Driven Simulation:** Events (arrivals, departures) were scheduled and processed based on time, allowing for accurate modeling of the queueing process.
- **Queue Disciplines:** Different queue disciplines (FCFS, LCFS, SJF, Priority Queues) were implemented to assess their effects on system performance.
- **Statistical Analysis:** The simulations used **confidence intervals** to provide a measure of uncertainty and reliability in the performance metrics, calculated using the **t-distribution** for more accurate estimations.

---

## Results and Insights

In each project, the following performance metrics were analyzed:
- **Average waiting time:** The time customers spend in the queue before being served.
- **Average service time:** The time customers spend receiving service.
- **Average system time:** The total time spent in the system (waiting time + service time).
- **Confidence intervals:** 95% confidence intervals were calculated for each of the metrics to account for sampling variability.

Some insights from the series include:
- **Service discipline matters:** Different queueing disciplines (e.g., SJF vs. FCFS) lead to different customer experiences, especially in systems with heavy loads or varying service times.
- **Multiple servers reduce waiting time:** In multi-server systems (like the M/M/3 model), the waiting time decreases significantly as compared to a single-server system.
- **Pareto distributions are realistic:** The M/G/3 system with Pareto-distributed service times demonstrated that service times often follow a heavy-tailed distribution, which is more reflective of real-world scenarios.

---

## Conclusion

This **Simulation Series** provides a comprehensive exploration of **queueing systems**, focusing on the impact of service disciplines, server configurations, and service time distributions on system performance. From simple M/M/1 models to more complex multi-server and priority systems, these simulations highlight key insights into how queuing theory can be applied to real-world problems.

Each project builds upon the last, offering a progression from basic to advanced simulations and introducing more sophisticated techniques, such as **confidence interval estimation** and the use of **Pareto distributions** for service times. This series serves as a valuable tool for understanding and analyzing the behavior of queueing systems in various environments.

---

## Running the Simulations

### Prerequisites
To run these simulations, you will need the following Python libraries:
- `numpy`
- `scipy`
- `matplotlib`

You can install them using `pip`:

```bash
pip install numpy scipy matplotlib
