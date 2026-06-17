#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 12:45:12 2026

@author: francescoceccarelli

UPP Optical Transmission Characterization
--------------------------------------
This script automates the total power transmission measurement for each 
input mode of a Universal Photonic Processor (UPP).

Workflow:
1. Cycle through input modes and measure the total output power.
2. Save the results to a CSV file.
3. Display results via console and a bar chart.
"""

import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lib.UPP import UniversalPhotonicProcessor

# --- CONFIGURATION ---
SWITCH_PORT = "ASRL3::INSTR"    # Lfiber switch resource
CHIP_ID = "BellOne"             # Identifier for the chip under test
NUM_MODES = 6                   # Number of modes of the UPP
WAVELENGTH = 1550               # Operating wavelength (nm)
LD_CURRENT = 90                 # Laser pump current (mA)

def run_optical_characterization():
    """
    Execute the optical transmission measurement on the UPP.
    """
    upp = None

    try:
        # Initialize the UPP controller
        upp = UniversalPhotonicProcessor(
            num_modes=NUM_MODES,
            wavelength_nm=WAVELENGTH,
            imax=5.0, 
            vmax=5.0, 
            switch_resource=SWITCH_PORT,
            verbose=False
        )
        
        print("[INFO] Transmission characterization started.")
        
        # Measure total transmission per input mode
        power_matrix = upp.measure_power_matrix()
        transmission = np.sum(power_matrix, axis=0)
        
        print("[INFO] Transmission characterization complete.")

    except Exception as e:
        print(f"[ERROR] Characterization failed: {e}.")
        return None

    finally:
        # Resource cleanup
        if upp is not None:
            upp.close()
            
    return transmission

def save_results(transmission):
    """
    Save transmission results to a CSV file with metadata.
    """
    output_folder = "results"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"[INFO] Created folder '{output_folder}'")

    file_date = time.strftime("%Y%m%d")
    full_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    filename = os.path.join(output_folder, f"{file_date}_{CHIP_ID}_transmission.csv")    
    
    try:
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            # Saving measurement settings in the header
            f.write("# UPP OPTICAL TRANSMISSION REPORT\n")
            f.write(f"# Date: {full_timestamp}\n")
            f.write(f"# Chip ID: {CHIP_ID}\n")
            f.write(f"# Number of Modes: {NUM_MODES}\n")
            f.write(f"# Wavelength: {WAVELENGTH} nm\n")
            f.write(f"# LD Pump Current: {LD_CURRENT} mA\n")
            f.write("# --------------------------------------\n")
            
            # Saving measured data
            df_results = pd.DataFrame({
                "Input Mode": range(0, NUM_MODES),
                "Total Power (W)": transmission
            })
            df_results.to_csv(f, index=False, lineterminator='\n')
            
        print(f"[INFO] Results and settings saved to: {filename}.")

        return filename
    
    except Exception as e:
        print(f"[ERROR] Failed to save CSV: {e}.")
        return None

def plot_results(transmission):
    """
    Visualize transmission in a bar plot.
    """
    plt.figure(figsize=(10, 5))
    
    modes = [f"In {i}" for i in range(NUM_MODES)]
    
    # Plotting in mW for better readability
    plt.bar(modes, transmission * 1e6, color='forestgreen', edgecolor='black', alpha=0.8)
    
    plt.ylabel(r'Total Output Power ($\mu$W)')
    plt.xlabel('Input Mode')
    plt.title(f'Optical Transmission - {CHIP_ID} (@{WAVELENGTH}nm, {LD_CURRENT}mA)')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.show()
    print("[INFO] Bar plot of the transmission is ready.")

# --- ENTRY POINT ---
if __name__ == "__main__":
    
    # 1. Run measurement
    transmission_data = run_optical_characterization()
    
    if transmission_data is not None:
        # 2. Save data
        save_results(transmission_data)
        
        # 3. Plot data
        plot_results(transmission_data)