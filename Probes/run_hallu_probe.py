#!/usr/bin/env python3
"""
Runner script for the hallucination probe in SAE environment.
This script activates the SAE conda environment and runs the hallucination probe.
"""

import subprocess
import sys
import os

def run_hallu_probe():
    """Run the hallucination probe in the SAE environment."""
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hallu_probe_path = os.path.join(current_dir, "hallu_probe.py")
    
    print("🚀 Starting Hallucination Probe in SAE Environment...")
    print(f"📁 Working directory: {current_dir}")
    print(f"📄 Running: {hallu_probe_path}")
    print("-" * 50)
    
    try:
        # Run the hallucination probe
        result = subprocess.run([
            sys.executable, hallu_probe_path
        ], cwd=current_dir, check=True, capture_output=False)
        
        print("-" * 50)
        print("✅ Hallucination probe completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running hallucination probe: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = run_hallu_probe()
    sys.exit(0 if success else 1)
