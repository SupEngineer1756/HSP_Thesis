
# Hoist Scheduling Optimization

This project implements the methodologies presented in Emna Laajili's research on the **Cyclic Multi-Hoist Scheduling Problem**, specifically the work detailed in:

- Laajili, E. (2021). *Modélisation et algorithmes pour le dimensionnement et l'ordonnancement cyclique d'atelier de traitement de surface*. Université de Technologie de Belfort-Montbéliard. [Link](https://theses.hal.science/tel-03551785v1/file/These_LAAJILI_UTBM.pdf)
- Laajili, E., Lamrous, S., Manier, M.-A., & Nicod, J.-M. (2019). *Collision-Free Based Model for the Cyclic Multi-Hoist Scheduling Problem*. International Conference on Systems, Man, and Cybernetics. [Link](https://hal.science/hal-02399243v1)

This work addresses **hoist scheduling in surface treatment workshops**, optimizing both the **cycle time**.

## Problem Definition

The **Cyclic Hoist Scheduling Problem (CHSP)** involves scheduling automated hoists that transport parts between processing tanks in a surface treatment line. The goal is to **minimize the cycle time** while ensuring:
- **No-wait constraints**: Parts must not exceed the maximum soaking time.
- **No-storage constraints**: There are no intermediate buffers.

## Mixed-Integer Linear Programming (MILP) Model

The **MILP model** is used to determine the **minimum feasible cycle time** while enforcing all technological constraints. The objective function is:

  $\min T$

where **T** is the cycle time.

### Constraints:
1. **Tank and Hoist Capacity Constraints**  
   Each tank and hoist can process only one part at a time.

2. **Soaking Time Constraints**  
   Each part must stay in a tank for a duration within a fixed range \([m_i, M_i]\).

3. **No-Wait and No-Storage Constraints**  
   No intermediate storage is allowed, and all operations must proceed without interruption.

4. **Hoist Motion Constraints**  
   Hoists must have enough time to complete empty moves before the next transport task.

The MILP model ensures a **feasible cyclic schedule** for multiple hoists while minimizing \(T\). However, solving this exactly for large instances can be computationally expensive, which is why a **Genetic Algorithm (GA)** is used as an alternative.

## Genetic Algorithm (GA) for Optimization

A **Genetic Algorithm** is used to find an optimal hoist schedule after the MILP determines the minimum cycle time.

### Representation:
Each **chromosome** represents a **hoist movement sequence**, encoded as an ordered list of empty and loaded moves.

### Evolutionary Process:
1. **Initialization** – Generate an initial population of feasible hoist schedules.
2. **Selection** – Choose the best-performing schedules based on a fitness function.
3. **Crossover** – Combine elements of two schedules to create new ones.
4. **Mutation** – Introduce slight variations to explore different scheduling possibilities.
5. **Evaluation** – Use the MILP model to verify feasibility and compute the cycle time.
6. **Termination** – Repeat until a stopping condition (e.g., number of generations) is met.

### Hybrid Approach:
The GA **first generates partially feasible solutions**, and then the MILP model **validates feasibility** by checking collision-free constraints. This hybrid approach balances **exploration (GA)** with **precision (MILP)**.

## References

This implementation is based on the methodologies described in:

- Laajili, E. (2021). *Modélisation et algorithmes pour le dimensionnement et l'ordonnancement cyclique d'atelier de traitement de surface*. Université de Technologie de Belfort-Montbéliard. [Link](https://theses.hal.science/tel-03551785v1/file/These_LAAJILI_UTBM.pdf)
- Laajili, E., Lamrous, S., Manier, M.-A., & Nicod, J.-M. (2019). *Collision-Free Based Model for the Cyclic Multi-Hoist Scheduling Problem*. International Conference on Systems, Man, and Cybernetics. [Link](https://hal.science/hal-02399243v1)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
