import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_benchmark_last_row(csv_path: Path, output_image_path: Path = None):
    if not csv_path.exists():
        print(f"Error: CSV file not found at '{csv_path}'")
        return

    # Read CSV
    df = pd.read_csv(csv_path)
    if df.empty:
        print("Error: The CSV file is empty.")
        return

    # Extract the last row
    last_row = df.iloc[-1]
    row_label = last_row["Graph_Name"] if "Graph_Name" in last_row else "Last Row"

    # Identify threshold columns dynamically using regex
    size_pattern = re.compile(r"^Mult_Size_([0-9.]+)$")
    time_pattern = re.compile(r"^Mult_Time_([0-9.]+)$")

    thresholds = []
    size_multipliers = []
    time_multipliers = []

    for col in df.columns:
        match_size = size_pattern.match(col)
        if match_size:
            th = float(match_size.group(1))
            time_col = f"Mult_Time_{match_size.group(1)}"

            if time_col in df.columns:
                thresholds.append(th)
                size_multipliers.append(float(last_row[col]))
                time_multipliers.append(float(last_row[time_col]))

    if not thresholds:
        print("Error: No multiplier columns found in the CSV.")
        return

    # Sort data by threshold value in ascending order
    sorted_data = sorted(zip(thresholds, size_multipliers, time_multipliers), key=lambda x: x[0])
    thresholds, size_multipliers, time_multipliers = zip(*sorted_data)

    # Plot settings
    plt.figure(figsize=(10, 6), dpi=120)
    plt.plot(
        thresholds,
        size_multipliers,
        marker="o",
        linewidth=2,
        color="#1f77b4",
        label="Size Multiplier (Batch / Classic)"
    )
    plt.plot(
        thresholds,
        time_multipliers,
        marker="s",
        linewidth=2,
        color="#d62728",
        label="Time Multiplier (Batch / Classic)"
    )

    # Add reference baseline line at y=1.0 (equivalent to classic)
    plt.axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="Classic Baseline (1.0x)")

    plt.title(f"PageRankFAS Performance by Threshold ({row_label})", fontsize=14, fontweight="bold")
    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("Multiplier vs. Classic", fontsize=12)
    plt.xticks(thresholds, rotation=45)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()

    # Save and display
    if output_image_path:
        plt.savefig(output_image_path, bbox_inches="tight")
        print(f"Plot saved to '{output_image_path}'.")

    plt.show()

if __name__ == "__main__":
    csv_file = Path(__file__).resolve().parent / "benchmark_CSVs" / "benchmark_results_erdosrenyi_0p65_to_0p85.csv"
    plot_file = Path(__file__).resolve().parent / "benchmark_CSVs" / "benchmark_plot_erdosrenyi_0p65_to_0p85.png"

    plot_benchmark_last_row(csv_path=csv_file, output_image_path=plot_file)