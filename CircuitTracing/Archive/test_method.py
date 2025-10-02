#!/usr/bin/env python3
"""
Test the method signature issue.
"""

import sys
sys.path.append('/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/CircuitTracing')

from Tracing_app import FeatureActivationTracker

# Test method signature
tracker = FeatureActivationTracker()
print("Method signature:", tracker.get_activations.__code__.co_varnames)
print("Number of arguments:", tracker.get_activations.__code__.co_argcount)

# Test method call
try:
    result = tracker.get_activations("test", [4], "max")
    print("Method call successful")
except Exception as e:
    print(f"Method call failed: {e}")
    import traceback
    traceback.print_exc()
