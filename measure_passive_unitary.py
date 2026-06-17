#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 13:10:00 2026

@author: francescoceccarelli

UPP Passive Unitary Matrix Characterization
-------------------------------------------
This script automates the measurement of the NxN power matrix of a 
Universal Photonic Processor (UPP) and reconstructs the unitary amplitudes.

Workflow:
1. Scan and measure all input/output combinations.
2. Save the results to a CSV file.
3. Display the results on a heatmap.
"""

import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lib.UPP import UniversalPhotonicProcessor

# --- CONFIGURATION ---
SWITCH_PORT = "ASRL3::INSTR"    # Lfiber switch resource
CHIP_ID = "BellOne"             # Identifier for the chip under test
NUM_MODES = 6                   # Number of modes of the UPP
WAVELENGTH = 1550               # Operating wavelength (nm)
LD_CURRENT = 90                 # Laser pump current (mA)

def run_unitary_characterization():
    """
    Execute the passive unitary matrix measurement on the UPP.
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
        
        print("[INFO] Passive unitary characterization started.")
        
        # Measure and reconstruct unitary amplitudes |U_ij|
        # This method already applies Sinkhorn normalization
        unitary_matrix = upp.measure_unitary_matrix()
        
        print("[INFO] Passive unitary characterization complete.")

    except Exception as e:
        print(f"[ERROR] Characterization failed: {e}.")
        return None

    finally:
        # Resource cleanup
        if upp is not None:
            upp.close()
            
    return unitary_matrix

def save_results(matrix):
    """
    Save the passive unitary matrix to a CSV file with metadata.
    """
    output_folder = "results"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"[INFO] Created folder '{output_folder}'")

    file_date = time.strftime("%Y%m%d")
    full_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    filename = os.path.join(output_folder, f"{file_date}_{CHIP_ID}_passivematrix.csv")    
    
    try:
        # Flatten the matrix for saving or save as a block
        # Here we save it as a clean CSV table without indices for easy re-loading
        df_results = pd.DataFrame(matrix)
        
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            f.write("# UPP PASSIVE UNITARY MATRIX REPORT\n")
            f.write(f"# Date: {full_timestamp}\n")
            f.write(f"# Chip ID: {CHIP_ID}\n")
            f.write(f"# Number of Modes: {NUM_MODES}\n")
            f.write(f"# Wavelength: {WAVELENGTH} nm\n")
            f.write(f"# LD Pump Current: {LD_CURRENT} mA\n")
            f.write("# Data: Reconstructed amplitudes |U_ij|\n")
            f.write("# --------------------------------------\n")
            df_results.to_csv(f, index=False, header=False, lineterminator='\n')
            
        print(f"[INFO] Results and settings saved to: {filename}.")

        return filename
    
    except Exception as e:
        print(f"[ERROR] Failed to save CSV: {e}.")
        return None

def plot_results(matrix):
    """
    Visualize the passive unitary matrix as a heatmap.
    """
    plt.figure(figsize=(8, 6))
    
    # Create the heatmap
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap='viridis', 
                xticklabels=range(0, NUM_MODES), 
                yticklabels=range(0, NUM_MODES))
    
    plt.title(f'Passive Unitary Matrix Amplitudes |U| - {CHIP_ID}\n(@{WAVELENGTH}nm, {LD_CURRENT}mA)')
    plt.xlabel('Input Mode')
    plt.ylabel('Output Mode')
    
    plt.tight_layout()
    plt.show()
    print("[INFO] Heatmap of the passive unitary matrix is ready.")

# --- ENTRY POINT ---
if __name__ == "__main__":
    
    # 1. Run measurement
    u_matrix = run_unitary_characterization()
    
    if u_matrix is not None:
        # 2. Save data
        save_results(u_matrix)
        
        # 3. Plot data
        plot_results(u_matrix)