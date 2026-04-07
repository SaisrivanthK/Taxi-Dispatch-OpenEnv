---
title: Taxi Dispatch OpenEnv
emoji: 🚖
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# Taxi Dispatch OpenEnv

A dynamic reinforcement learning environment for simulating real-world ride dispatch and mobility optimization.

This environment models how cars and drivers are assigned to ride requests under constraints such as urgency, customer tier, traffic, and system load.

---

## Problem Overview

Ride-hailing systems must:

- Match riders to appropriate vehicles
- Handle dynamic demand and supply
- Minimize wait time and cancellations
- Respect customer tiers (normal, premium, VIP)
- Optimize driver utilization
- Maintain system efficiency

This environment simulates these challenges in a controlled RL setting.

---

## Environment Design

### Entities

Riders:
- tier: normal / premium / vip
- group size
- urgency
- wait time

Cars:
- types: basic, sedan, xuv, armored
- capacity
- zone location

Drivers:
- availability
- location
- utilization metrics

---

### Actions

- assign: assign a car to a rider
- pool: combine compatible riders
- reposition: move driver across zones
- wait: take no action
- no_op: no operation

---

### Observations

Each step returns:

- current time
- waiting riders
- available cars
- driver status
- system metrics
- recent events

---

### Reward Function

Multi-objective reward:

- positive reward for successful ride completion
- positive reward for fast pickup
- positive reward for correct car matching
- penalty for cancellations
- penalty for long wait times
- penalty for unsafe matches

---

## Tasks

### Easy — ride_matching
- low demand
- no cancellations
- focus on correct assignment

### Medium — dispatch_allocation
- moderate demand
- traffic and cancellations
- balance efficiency

### Hard — surge_mobility
- high demand
- dynamic arrivals
- driver repositioning and pooling
- system-level optimization

---

## Metrics

- completion rate
- cancellation rate
- wait efficiency
- utilization rate
- pooling rate
- safe drop rate

Final score is normalized between 0.0 and 1.0.

---

## Baseline

Run the baseline agent:

python baseline/run_baseline.py

This produces reproducible scores across all tasks.

---

## Inference

Run:

python inference.py

The script outputs structured logs in the format:

[START]
[STEP]
...
[END]

---

## API Endpoints

- /reset : reset environment
- /step : apply action
- /state : get current state
- /score : get score
- /tasks : list tasks
- /openenv.yaml : environment specification

---

## Docker

Build and run:

docker build -t taxi-env .
docker run -p 7860:7860 taxi-env

---

## Setup

pip install -r requirements.txt  
python app.py  

Open in browser:

http://127.0.0.1:7860/docs

---

## Project Structure

env/  
baseline/  
tests/  
inference.py  
app.py  
openenv.yaml  
Dockerfile  
README.md  

---

## Notes

- Designed to run on 2 vCPU, 8GB RAM
- Inference completes within 20 minutes
- Deterministic baseline for reproducibility

---

## License

MIT