#!/usr/bin/env python3
"""
Easy WandB Sweep Runner

This script makes it easy to start a WandB sweep without manual commands.
"""

import wandb
import subprocess
import sys
import os
import yaml
from dotenv import load_dotenv
from pathlib import Path

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def main():
    """Initialize and run the WandB sweep."""
    
    # Load environment variables
    load_dotenv()
    
    # Login to WandB
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    
    print("🚀 Starting WandB Sweep for Two-Tower ML Retrieval")
    print("=" * 60)
    
    # Load the sweep configuration
    print("📋 Loading sweep configuration...")
    sweep_config_path = "backend/sweep/sweep_config.yaml"
    if not os.path.exists(sweep_config_path):
        sweep_config_path = "sweep_config.yaml"  # fallback for running from sweep directory
    
    with open(sweep_config_path, "r") as f:
        sweep_config = yaml.safe_load(f)
    
    # Initialize the sweep
    print("📋 Initializing sweep...")
    sweep_id = wandb.sweep(
        sweep=sweep_config, 
        project="two-tower-ml-retrieval"
    )
    
    print(f"✅ Sweep initialized with ID: {sweep_id}")
    print(f"🌐 View your sweep at: https://wandb.ai/your-username/two-tower-ml-retrieval/sweeps/{sweep_id}")
    
    # Ask user how many agents to run
    try:
        num_agents = input("\n🤖 How many agents do you want to run? (default: 1): ").strip()
        num_agents = int(num_agents) if num_agents else 1
    except ValueError:
        num_agents = 1
    
    print(f"\n🏃 Starting {num_agents} agent(s)...")
    
    if num_agents == 1:
        # Import the training function
        from sweep_train import sweep_train
        
        # Run single agent
        print("🔄 Running sweep agent...")
        wandb.agent(sweep_id, function=sweep_train, count=None)
    else:
        # Run multiple agents in parallel using subprocess
        print(f"🔄 Starting {num_agents} parallel agents...")
        processes = []
        
        # Get the current working directory to ensure proper execution
        current_dir = os.getcwd()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        for i in range(num_agents):
            print(f"  Starting agent {i+1}/{num_agents}...")
            
            # Create command to run sweep agent
            cmd = [
                sys.executable, 
                "-c",
                f"import sys; import os; "
                f"sys.path.append('{os.path.join(script_dir, '..')}'); "
                f"sys.path.append('backend'); "
                f"os.chdir('{current_dir}'); "
                f"import wandb; "
                f"from dotenv import load_dotenv; "
                f"load_dotenv(); "
                f"wandb.login(key=os.getenv('WANDB_API_KEY')); "
                f"from backend.sweep.sweep_train import sweep_train; "
                f"wandb.agent('{sweep_id}', function=sweep_train)"
            ]
            
            process = subprocess.Popen(cmd, cwd=current_dir)
            processes.append(process)
        
        print(f"✅ {num_agents} agents started!")
        print("💡 Press Ctrl+C to stop all agents")
        
        try:
            # Wait for all processes
            for process in processes:
                process.wait()
        except KeyboardInterrupt:
            print("\n⏹️  Stopping all agents...")
            for process in processes:
                process.terminate()
            print("✅ All agents stopped")

if __name__ == "__main__":
    main() 