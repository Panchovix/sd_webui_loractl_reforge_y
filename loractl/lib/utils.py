import numpy as np
import re

# Given a string like x@y,z@a, returns [[x, z], [y, a]] sorted for consumption by np.interp


def normalise_steps(step, n_steps):
    if step >= 1:
        return float(step)
    if step <= 0:
        return 0.0
    return n_steps * step

def sorted_positions(raw_steps, n_steps):
    steps = [[float(s.strip()) for s in re.split("[@~]", x)]
             for x in re.split("[,;]", str(raw_steps))]
    # If we just got a single number, just return it
    step_triggers = {}
    if len(steps[0]) == 1:
        step_triggers[0] = steps[0][0]
    else:
        for s in sorted(steps, key=lambda s: normalise_steps(s[1] if len(s) == 2 else 1, n_steps)):
            step_triggers[int(normalise_steps(s[1] if len(s) == 2 else 1, n_steps))] = s[0]
    return step_triggers


def calculate_weight(m, step, max_steps, step_offset=2):
    if isinstance(m, list):
        if m[1][-1] <= 1.0:
            if max_steps > 0:
                step = (step) / (max_steps - step_offset)
            else:
                step = 1.0
        else:
            step = step
        v = np.interp(step, m[1], m[0])
        return v
    else:
        return m


def params_to_weights(params, steps):
    weights = sorted_positions(params.positional[1], steps)
    weights_return = {}
    for (step, weight) in weights.items():
        if step not in weights_return:
            weights_return[step] = {}
        weights_return[step] = weight
    return weights_return


hires = False
loractl_active = True

def is_hires():
    return hires


def set_hires(value):
    global hires
    hires = value


def set_active(value):
    global loractl_active
    loractl_active = value

def is_active():
    global loractl_active
    return loractl_active
