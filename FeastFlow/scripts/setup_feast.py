import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, UTC

# --- Configuration ---
# Set the path to the Feast feature repository root
FEATURE_REPO_PATH = Path("feature_repo") 
# ---------------------

def setup_redis():
    """Start Redis server for Feast online store"""
    try:
        # Try to start Redis using docker-compose
        # We ensure it's up, running, and not already running 
        subprocess.run(["docker-compose", "up", "-d", "redis"], check=True, capture_output=True, text=True)
        print("Redis server started successfully (or was already running).")
        time.sleep(5)  # Wait for Redis to be ready
        return True
    except Exception as e:
        print(f"Failed to start Redis with docker-compose: {e}")
        print("Please make sure Docker is running and Redis is accessible on localhost:6379")
        return False

# --- FIX 1: Corrected Feast CLI execution ---
def apply_feast():
    """Apply Feast feature definitions and clearly display output/errors."""
    print(f"Applying Feast definitions in: {FEATURE_REPO_PATH}")
    
    try:
        # Run feast apply
        result = subprocess.run(
            ["feast", "apply"], 
            cwd=FEATURE_REPO_PATH,
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            print("--- Feast Apply Output (STDOUT) ---")
            print(result.stdout)
            
        if result.stderr:
            print("--- Feast Apply Warnings/Errors (STDERR) ---")
            print(result.stderr)
        # ------------------------------------------------------------------
        
        if result.returncode == 0:
            print("\nFeast feature store applied successfully! ✅")
            return True
        else:
            print("\nError applying Feast feature store: ❌")
            print(f"Feast command failed with return code: {result.returncode}")
            # The error message is already in result.stderr, so we just return False
            return False
            
    except FileNotFoundError:
        print("\nError: The 'feast' command was not found. ⚠️")
        print("Ensure Feast is installed and the correct environment is activated.")
        return False
    except Exception as e:
        print(f"\nCritical Error during subprocess execution: {e}")
        return False

def materialize_features():
    """Materialize features to online store"""
    print("Materializing features to online store...")
    try:
        now = datetime.now(UTC).isoformat(timespec='seconds')
        result = subprocess.run(
            ["feast", "materialize", "2024-01-01", now],
            cwd=FEATURE_REPO_PATH,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("Features materialized successfully! 💾")
            print("--- Feast Materialize Output ---")
            print(result.stdout)
            return True
        else:
            print("Error materializing features: ❌")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Error materializing features: {e}")
        return False

if __name__ == "__main__":
    print("Setting up Feast feature store pipeline...")

    if setup_redis():
        if apply_feast():
            materialize_features()