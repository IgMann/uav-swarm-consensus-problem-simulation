# UAV Swarm Vertical Alignment Consensus Simulation

This project explores the vertical alignment of Unmanned Aerial Vehicles (UAVs) using various consensus protocols and interaction topologies. The primary objective is to study how different strategies impact the convergence of UAV heights to a common value over time. Project is made as part of a master's studies in the AI ​​& ML master's program at the University of Novi Sad.

## Table of Contents
  - [Introduction](#introduction)
  - [Theoretical Background](#theoretical-background)
    - [Consensus Algorithms](#consensus-algorithms)
    - [Protocol Design](#protocol-design)
    - [Game Theory and Mechanism Design](#game-theory-and-mechanism-design)
  - [Simulation Design](#simulation-design)
    - [Initialization](#initialization)
    - [Iterative Update](#iterative-update)
  - [Protocols](#protocols)
  - [Topologies](#topologies)
  - [Installation](#installation)
  - [Contributors](#contributors)
  - [References](#references)
  - [License](#license)

## Introduction

Vertical alignment of UAVs is a critical task in formations where coordination among multiple agents is necessary. This project models and simulates the behavior of UAVs adjusting their heights according to specific rules (protocols) and interaction patterns (topologies). The ultimate goal is to achieve a consensus where all UAVs reach the same altitude, demonstrating the effectiveness of the chosen protocols and topologies.

## Theoretical Background

### Consensus Algorithms

Consensus algorithms are fundamental in distributed systems, ensuring all agents (UAVs) agree on a certain state (height) through local interactions. These algorithms are crucial in various applications, including UAV formations, autonomous vehicles, and multi-agent systems. The primary goal of a consensus algorithm is to reach a common decision value among all agents, despite each agent having only local information about the system state.

### Protocol Design

The design of consensus protocols involves developing rules that dictate how agents update their states based on local information. The protocols can be linear or non-linear, and their effectiveness depends on the underlying interaction topology and the specific application requirements. For instance, in UAV vertical alignment, protocols like arithmetic mean, geometric mean, harmonic mean, and mean of order 2 are used to adjust the UAV heights.

According to Bauso et al. (2006), optimal consensus protocols can be derived using non-linear control policies. These protocols ensure that agents reach consensus on a group decision value, which is a function of all agents' initial states. The design of these protocols can be interpreted through the lens of mechanism design in game theory, where agents optimize individual objectives to achieve a collective goal.

### Game Theory and Mechanism Design

Mechanism design, a subfield of game theory, focuses on designing rules or incentives that lead to desired outcomes in strategic settings. In the context of consensus problems, mechanism design ensures that rational agents, each optimizing their own objective, collectively reach a consensus on the desired group decision value. This approach distributes intelligence and reduces monitoring costs, making it suitable for decentralized control systems.

Bauso et al. (2006) provide a game-theoretic interpretation of consensus protocols, viewing them as solutions to mechanism design problems. By imposing individual objectives through convex penalty functions, a supervisor can guide agents towards a unique optimal protocol, achieving consensus as a byproduct of local optimizations.

## Simulation Design

The simulation involves initializing UAVs with random heights and positions, then iteratively updating their heights based on the chosen protocol and topology. The process continues for a specified number of iterations or until the UAVs' heights converge to a common value.

### Initialization

UAVs are initialized with random (x, y) positions within specified ranges. This random initialization sets up a diverse starting point for the simulation, ensuring varied interaction patterns.

### Iterative Update

In each iteration, UAVs update their heights according to the specified protocol. The interaction partners are determined based on the selected topology. The simulation tracks these updates to analyze the convergence behavior over time.

## Protocols

1. **Arithmetic Mean**: Simple average of neighboring UAVs' heights.
2. **Geometric Mean**: Multiplicative average, suitable for proportional growth or decay.
3. **Harmonic Mean**: Useful in scenarios where lower values are more significant.
4. **Mean of Order 2**: A special case of power mean, providing different convergence characteristics.

## Topologies

1. **All-to-All**: Provides interactions between all UAVs in swarm without any restrictions.
2. **N-Nearest Neighbors**: Provides interactions between given number (n) of the closest UAVs in swarm to designated UAV. This localized interaction promoting regional consensus.
3. **Random**: Provides interactions between random or given number (n) of UAVs in swarm to designated UAV. Random interactions introducing variability and robustness against network changes.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/IgMann/uav-swarm-consensus-problem-simulation.git
    cd uav-swarm-consensus-problem-simulation
    ```

2. Install the required dependencies:
    ```bash
    conda env create -f environment.yaml
    conda activate uav-swarm-consensus
    ```

## Contributors

- Igor Mandarić
- Stefan Dragičević

## References

- Bauso, D., Giarré, L., & Pesenti, R. (2006). Non-linear protocols for optimal distributed consensus in networks of dynamic agents. Systems & Control Letters, 55(11), 918-928. [DOI: 10.1016/j.sysconle.2006.06.005](https://doi.org/10.1016/j.sysconle.2006.06.005)
- Chapter 13.5 from "Game Theory with Engineering Applications" by Dario Bauso

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.
