import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import re

def main():
    """
    Reads pressure data from multiple CSV files, calculates total pressure,
    and plots the total pressure versus time on a single graph.
    The plot is saved as a PNG file in the 'out' directory.
    """
    # 1. Set up argument parser to accept multiple files
    parser = argparse.ArgumentParser(description="Plot total pressure versus time from multiple CSV files.")
    parser.add_argument("--use-files", nargs='+', required=True, help="Paths to the input CSV files (e.g., file1.txt file2.txt).")
    parser.add_argument("--show", action="store_true", help="Display the plot window.")
    
    args = parser.parse_args()

    input_files = args.use_files
    show_plot = args.show

    # 2. Set up the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))

    # 3. Loop through each input file
    for input_file in input_files:
        # Check if the input file exists
        if not os.path.exists(input_file):
            print(f"Error: The file '{input_file}' was not found. Skipping.")
            continue

        # Read the data using pandas, skipping the header line
        print(f"Reading data from '{input_file}'...")
        try:
            data = pd.read_csv(
                input_file, 
                sep='\s+', 
                skiprows=1,
                names=['t', 'P_left', 'P_right', 'P_total_from_file']
            )
        except Exception as e:
            print(f"Error reading file '{input_file}': {e}. Skipping.")
            continue

        # Extract the necessary columns
        time = data['t']
        p_left = data['P_left']
        p_right = data['P_right']

        # Calculate total pressure from the two pressure columns
        total_pressure = p_left + p_right
        
        # Get the filename for the plot label
        file_name = os.path.basename(input_file)
        
        # Use regex to extract the L and N values from the filename
        match = re.search(r'L=(.+)_N=(\d+)', file_name)
        if match:
            legend_label = f'L={match.group(1)} N={match.group(2)}'
        else:
            legend_label = file_name
        
        # Plot the data
        plt.plot(time, total_pressure, label=legend_label, linewidth=2)

    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Total Pressure (Pa)", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 5. Save the plot
    output_dir = "out"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_filename = os.path.join(output_dir, "total_pressure_plot_multiple.png")
    plt.savefig(output_filename)
    print(f"Plot saved to '{output_filename}'")

    # 6. Show the plot if the --show flag is present
    if show_plot:
        plt.show()

if __name__ == "__main__":
    main()
