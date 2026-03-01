# type: ignore
import numpy as np

states = ["Engaged", "Casual", "AtRisk", "Churned"]
observations = ["High", "Medium", "Low", "None"]

state_idx = {s: i for i, s in enumerate(states)}
obs_idx = {o: i for i, o in enumerate(observations)}

A = np.array([
    [0.6, 0.3, 0.1, 0.0],
    [0.2, 0.4, 0.3, 0.1],
    [0.1, 0.2, 0.4, 0.3],
    [0.0, 0.0, 0.1, 0.9]
])

B = np.array([
    [0.7, 0.2, 0.1, 0.0],
    [0.3, 0.4, 0.2, 0.1],
    [0.1, 0.3, 0.4, 0.2],
    [0.0, 0.0, 0.1, 0.9]
])

pi = np.array([0.5, 0.3, 0.2, 0.0])