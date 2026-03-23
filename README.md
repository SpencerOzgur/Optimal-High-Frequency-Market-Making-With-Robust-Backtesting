# Optimal High-Frequency Trading in an Informed Limit Order Book

Columbia University — IEOR 4733 Project

## Overview

This project implements and extends the Avellaneda-Stoikov (2008) market making framework for high-frequency trading in limit order books.

We begin by reproducing the classical model and a Stanford (2018) extension, and then introduce more realistic execution dynamics, including queue position modeling and data-driven fill probabilities.

The goal is to bridge the gap between theoretical optimal market making and practical microstructure-aware trading systems.

---

## Repository Structure

### `src/`
Core implementation of models and simulation components.

- Market making models (Avellaneda-Stoikov, inventory control)
- Price processes and simulation environment
- Execution models (Poisson baseline → extended models)
- Utility functions and shared components

---

### `scripts/`
Executable scripts for running simulations and pipelines.

- Reproduce baseline results
- Run extended models
- End-to-end experiment execution

---

### `experiments/`
Reproducible research experiments.

- Avellaneda-Stoikov (2008) replication
- Stanford (2018) extension
- Queue position and execution model experiments (planned)

Each experiment includes:
- Configuration
- Execution script
- Output results and visualizations

---

### `docs/`
Supporting documentation and research references.

- Original papers (Avellaneda-Stoikov, Ho-Stoll)
- Notes and derivations
- Literature review and design decisions

---

## Planned Extensions

- Queue-position-aware execution modeling
- Machine learning-based fill probability estimation
- Integration with high-frequency market data (TAQ / ITCH)
- Realistic backtesting and evaluation framework

---

## Setup

```bash
git clone https://github.com/SpencerOzgur/Optimal-High-Frequency-Trading-in-Informed-Limit-Order-Book
cd Optimal-High-Frequency-Trading-in-Informed-Limit-Order-Book
