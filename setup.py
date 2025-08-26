#!/usr/bin/env python3
"""
Setup script for the trading bot.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    
    directories = [
        "data",
        "data/market_data",
        "data/indicators",
        "data/backtest_results",
        "results",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created {directory}")

def main():
    """Main setup function."""
    print("=== Trading Bot Setup ===")
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements. Please check your Python environment.")
        return False
    
    # Create directories
    create_directories()
    
    print("\n✓ Setup completed successfully!")
    print("\nTo run the trading bot:")
    print("  python -m src/run_optimized_backtest.py")

    print("Fork this repository to make it your own with your custom strategies. Feel free to contribute!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 