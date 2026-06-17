#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 00:02:48 2026

@author: francescoceccarelli

UPP Resistance Characterization
--------------------------------------
This script automates the resistance measurement of phase shifters in a 
Universal Photonic Processor (UPP). 

Workflow:
1. Load shifter mapping from an Excel file.
2. Apply a test current, measures resistance, and resets outputs to zero.
3. Save the results to a CSV file.
4. Display results via console and a bar chart.
"""

import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from lib.UPP import UniversalPhotonicProcessor

# --- CONFIGURATION ---
SERIAL_PORT = "COM4"            # Qontrol driver serial port
MAP_FILE = "map.xlsx"           # Excel file containing heater names
TEST_CURRENT = 2.0              # Current applied for measurement (mA)
I_COMPLIANCE = 5.0              # Max current safety limit (mA)
V_COMPLIANCE = 5.0              # Max voltage safety limit (V)
CHIP_ID = "BellOne"             # Identifier for the chip under test
NUM_MODES = 6                   # Number of modes of the UPP
WAVELENGTH = 1550               # Operating wavelength (nm)

def load_mapping(file_path):
    """
    Extract phase shifter mapping from the Excel file. The row index determines
    the driver's physical channel.
    """
    try:
        # Load the Excel file
        df = pd.read_excel(file_path)
        
        # Extract mapping from the first column and drop empty rows
        mapping = df.iloc[:, 0].dropna().astype(int).tolist() 
        
        print(f"[INFO] Loaded {len(mapping)} shifters from Excel.")
        return mapping
    
    except Exception as e:
        print(f"[ERROR] Excel loading failed: {e}.")
        return None

def run_characterization(mapping):
    """
    Execute the resistance measurement on the UPP.
    """
    upp = None

    try:
        # Initialize the high-level UPP controller
        upp = UniversalPhotonicProcessor(
            num_modes=NUM_MODES,
            wavelength_nm=WAVELENGTH,
            imax=I_COMPLIANCE,
            vmax=V_COMPLIANCE,
            q8iv_resource=SERIAL_PORT,
            shifter_channels=mapping,
            verbose=False
        )
        
        print("[INFO] Phase shifter resistance characterization started.")
        
        # Use the class method to measure all resistances
        resistances = upp.measure_shifter_resistances(test_current=TEST_CURRENT)
        
        print("[INFO] Phase shifter resistance characterization complete.")

    except Exception as e:
        print(f"[ERROR] Characterization failed: {e}.")
        return None

    finally:
        # Resource cleanup using the class close method
        if upp is not None:
            upp.close()
            
    return resistances

def save_results(channels, resistances):
    """
    Save results to a CSV file with a metadata header. Filename format:
    <YYYYMMDD>_<CHIP_ID>_resistances.csv
    """
    output_folder = "results"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"[INFO] Created folder '{output_folder}'")

    file_date = time.strftime("%Y%m%d")
    full_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    filename = os.path.join(output_folder, f"{file_date}_{CHIP_ID}_resistances.csv")    
    
    try:
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            # Saving measurement settings
            f.write("# UPP RESISTANCE CHARACTERIZATION REPORT\n")
            f.write(f"# Date: {full_timestamp}\n")
            f.write(f"# Chip ID: {CHIP_ID}\n")
            f.write(f"# Number of Modes: {NUM_MODES}\n")
            f.write(f"# Wavelength: {WAVELENGTH}\n")
            f.write(f"# Test Current: {TEST_CURRENT} mA\n")
            f.write(f"# I Compliance: {I_COMPLIANCE} mA\n")
            f.write(f"# V Compliance: {V_COMPLIANCE} V\n")
            f.write("# --------------------------------------\n")
            
            # Saving measured data
            df_results = pd.DataFrame({
                "Microheater": range(len(channels)),
                "Channel": channels,
                "Resistance": resistances
            })
            df_results.to_csv(f, index=False, lineterminator='\n')
            
        print(f"[INFO] Results and settings saved to: {filename}.")
        
        return filename
    
    except Exception as e:
        print(f"[ERROR] Failed to save CSV: {e}")
        return None

def plot_results(resistances):
    """
    Visualize results in a bar plot.
    """
    plt.figure(figsize=(10, 5))
    
    # Create x-axis labels with Microheater Index
    labels = [f"{i}" for i, res in enumerate(resistances)]
    
    # Plot the resistances
    plt.bar(labels, resistances, color='steelblue', edgecolor='black', alpha=0.8)
    
    # Decorate the plot
    plt.ylabel(r'Resistance ($\Omega$)')
    plt.xlabel('Microheater Index')
    plt.title(f'Resistance Characterization - {CHIP_ID} ({time.strftime("%Y-%m-%d")})')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add values on top of bars
    for i, res in enumerate(resistances):
        plt.text(i, res, f'{res:.0f}', ha='center', va='bottom', fontsize=9)

    plt.show()
    
    print("[INFO] Bar plot of the resistors is ready.")

# --- ENTRY POINT ---
if __name__ == "__main__":
    # 1. Load mapping
    mapping = load_mapping(MAP_FILE)
    
    if mapping:
        # 2. Run measurement
        resistances = run_characterization(mapping)
        
        if resistances is not None:
            # 3. Save data
            save_results(mapping, resistances)
            
            # 4. Plot data
            plot_results(resistances)