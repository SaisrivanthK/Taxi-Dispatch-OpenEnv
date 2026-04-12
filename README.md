---
title: Taxi Dispatch OpenEnv
emoji: 🚖
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---
# Taxi Dispatch OpenEnv — Real-World Fleet Optimization Environment

## 1. Overview

This project implements a **real-world task simulation** for taxi dispatch and fleet optimization using the OpenEnv specification.

The environment models how a ride-hailing platform dynamically assigns drivers and vehicles to rider requests under realistic constraints such as:

- Rider priority (normal, premium, VIP)
- Vehicle compatibility (capacity, type)
- Traffic conditions
- Dynamic demand (including surge scenarios)
- Driver availability and utilization

The goal is to evaluate and improve decision-making agents that operate in complex, real-time logistics systems.

---

## 2. Real-World Relevance

This environment simulates a real operational problem faced by companies such as Uber, Ola, and Lyft:

- Matching supply (drivers) with demand (riders)
- Minimizing wait times and cancellations
- Maximizing fleet utilization
- Handling demand spikes and geographic imbalance

This is not a toy problem — it is a simplified abstraction of **urban mobility optimization systems**.

---

## 3. OpenEnv Specification Compliance

The environment fully implements the OpenEnv interface:

### Core API

- `reset(task_id, seed)` → returns initial observation  
- `step(action)` → returns `(observation, reward, done, info)`  
- `state()` → returns full internal state  
- `score()` → returns normalized score (0.0–1.0)

### Typed Models

All inputs and outputs are defined using Pydantic models:

- `Observation`
- `Action`
- `StepResult`

### Environment Spec

The environment is described in `openenv.yaml`, including:

- observation space
- action space
- reward structure
- task definitions

---

## 4. Environment Design

### 4.1 Entities

#### Riders
- Tier: normal / premium / VIP
- Group size
- Pickup and drop zones
- Wait time
- Sharing preference

#### Vehicles
- Types: basic, sedan, XUV, armored
- Capacity constraints
- Zone location

#### Drivers
- Availability
- Zone location
- Utilization tracking

---

### 4.2 Action Space

Agents can take one of the following actions:

- `assign` — assign a driver and vehicle to a rider
- `pool` — combine two compatible riders into one trip
- `reposition` — move a driver to a different zone
- `wait` — delay action
- `no_op` — do nothing

---

### 4.3 Observation Space

Each step returns:

- Current simulation time
- Waiting riders
- Available vehicles
- Driver states
- Traffic factor
- System metrics (match rate, wait time, etc.)
- Recent events

---

## 5. Tasks and Objectives

The environment defines three tasks with increasing difficulty.

### Task 1: Ride Matching (Easy)

Objective:
- Maximize match rate
- Ensure correct vehicle assignment
- Minimize unsafe matches

Characteristics:
- Low demand
- No traffic
- No pooling or repositioning

---

### Task 2: Dispatch Allocation (Medium)

Objective:
- Maximize completion rate
- Reduce cancellations
- Balance efficiency and utilization

Characteristics:
- Moderate demand
- Traffic enabled
- Repositioning allowed

---

### Task 3: Surge Mobility (Hard)

Objective:
- Maintain system stability under high demand
- Use pooling effectively
- Optimize driver distribution

Characteristics:
- High demand
- Dynamic arrivals
- Pooling + repositioning required

---

## 6. Reward Function

The reward function provides **dense feedback throughout the episode**, not just terminal rewards.

### Positive Signals
- Successful ride completion
- Correct vehicle–rider matching
- Reduced wait time
- Effective pooling
- Successful driver repositioning

### Negative Signals
- Long wait times
- Rider cancellations
- Unsafe or incorrect matches
- Inefficient assignments
- Idle system behavior

The final score is normalized between **0.0 and 1.0** using task-specific graders.

---

## 7. Agent Design

### Baseline Agent

A deterministic rule-based agent is provided:

- Assigns nearest feasible vehicle
- Uses simple heuristics for matching
- Limited global optimization

---

### Smart Agent (Improved)

The improved agent uses a **multi-objective scoring function**:

Factors considered:
- Rider priority (VIP > premium > normal)
- Distance between driver and pickup
- Wait time penalty
- Vehicle suitability (capacity and type)

The agent evaluates multiple candidate assignments and selects the highest-scoring action.

---

## 8. Evaluation

### Baseline Performance

| Task | Score |
|------|------|
| Ride Matching | 0.58 |
| Dispatch Allocation | 0.78 |
| Surge Mobility | 0.78 |

---

### Improved Agent Performance

| Task | Score |
|------|------|
| Ride Matching | 0.68 – 0.72 |
| Dispatch Allocation | 0.80+ |
| Surge Mobility | 0.80+ |

The improved agent demonstrates better prioritization, reduced wait times, and more efficient dispatching.

---

## 9. Running the Project

### 9.1 Install Dependencies

```bash
pip install -r requirements.txt