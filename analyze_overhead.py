#!/usr/bin/env python3
"""
Analyze performance overhead of Confidential Computing vs Standard Mode
"""

import pandas as pd
import numpy as np

# Data
non_cc_data = {
    'Concurrency': [1, 2, 4, 8, 16, 32],
    'TTFT_p50': [3.14, 6.18, 12.11, 24.09, 30.52, 30.23],
    'E2E_p50': [3.14, 6.18, 12.11, 24.09, 30.52, 30.23],
    'Per_Request_TPS_p50': [158.88, 80.78, 41.00, 20.73, 14.88, 16.50],
    'Overall_TPS': [152.87, 82.00, 43.17, 24.578, 16.38, 16.23]
}

cc_data = {
    'Concurrency': [1, 2, 4, 8, 16, 32],
    'TTFT_p50': [3.20, 6.24, 12.10, 23.92, 31.44, 30.75],
    'E2E_p50': [3.21, 6.24, 12.10, 23.92, 31.44, 30.75],
    'Per_Request_TPS_p50': [155.38, 79.77, 39.80, 20.89, 15.90, 16.26],
    'Overall_TPS': [145.98, 79.99, 42.86, 24.19, 16.19, 15.46]
}

df_non_cc = pd.DataFrame(non_cc_data)
df_cc = pd.DataFrame(cc_data)

# Calculate overhead percentage: ((CC - Non-CC) / Non-CC) * 100
# Positive = CC is slower, Negative = CC is faster
overhead = pd.DataFrame()
overhead['Concurrency'] = df_non_cc['Concurrency']

# For latency metrics (higher is worse)
overhead['TTFT_Overhead_%'] = ((df_cc['TTFT_p50'] - df_non_cc['TTFT_p50']) / df_non_cc['TTFT_p50'] * 100).round(2)
overhead['E2E_Overhead_%'] = ((df_cc['E2E_p50'] - df_non_cc['E2E_p50']) / df_non_cc['E2E_p50'] * 100).round(2)

# For throughput metrics (lower is worse)
overhead['Per_Request_TPS_Overhead_%'] = ((df_cc['Per_Request_TPS_p50'] - df_non_cc['Per_Request_TPS_p50']) / df_non_cc['Per_Request_TPS_p50'] * 100).round(2)
overhead['Overall_TPS_Overhead_%'] = ((df_cc['Overall_TPS'] - df_non_cc['Overall_TPS']) / df_non_cc['Overall_TPS'] * 100).round(2)

print("=" * 100)
print("Confidential Computing (CC) Performance Overhead vs Standard Mode")
print("=" * 100)
print("\nPositive % = CC is slower (worse)")
print("Negative % = CC is faster (better)\n")
print(overhead.to_string(index=False))

# Summary statistics
print("\n" + "=" * 100)
print("SUMMARY STATISTICS")
print("=" * 100)

summary = {
    'Metric': ['TTFT', 'End-to-End Latency', 'Per-Request TPS', 'Overall TPS'],
    'Min Overhead %': [
        overhead['TTFT_Overhead_%'].min(),
        overhead['E2E_Overhead_%'].min(),
        overhead['Per_Request_TPS_Overhead_%'].min(),
        overhead['Overall_TPS_Overhead_%'].min()
    ],
    'Max Overhead %': [
        overhead['TTFT_Overhead_%'].max(),
        overhead['E2E_Overhead_%'].max(),
        overhead['Per_Request_TPS_Overhead_%'].max(),
        overhead['Overall_TPS_Overhead_%'].max()
    ],
    'Mean Overhead %': [
        overhead['TTFT_Overhead_%'].mean().round(2),
        overhead['E2E_Overhead_%'].mean().round(2),
        overhead['Per_Request_TPS_Overhead_%'].mean().round(2),
        overhead['Overall_TPS_Overhead_%'].mean().round(2)
    ]
}

df_summary = pd.DataFrame(summary)
print(df_summary.to_string(index=False))

# Key insights
print("\n" + "=" * 100)
print("KEY INSIGHTS")
print("=" * 100)

ttft_mean = overhead['TTFT_Overhead_%'].mean()
e2e_mean = overhead['E2E_Overhead_%'].mean()
tps_mean = overhead['Per_Request_TPS_Overhead_%'].mean()
overall_tps_mean = overhead['Overall_TPS_Overhead_%'].mean()

print(f"\n1. TTFT Overhead: {ttft_mean:.2f}% (avg) - CC adds ~{abs(ttft_mean):.1f}% latency to first token")
print(f"2. End-to-End Overhead: {e2e_mean:.2f}% (avg) - CC adds ~{abs(e2e_mean):.1f}% to total request time")
print(f"3. Per-Request TPS: {tps_mean:.2f}% (avg) - CC reduces throughput by ~{abs(tps_mean):.1f}%")
print(f"4. Overall TPS: {overall_tps_mean:.2f}% (avg) - CC reduces aggregate throughput by ~{abs(overall_tps_mean):.1f}%")

# Best and worst case
best_concurrency = overhead.loc[overhead['Overall_TPS_Overhead_%'].idxmax(), 'Concurrency']
worst_concurrency = overhead.loc[overhead['Overall_TPS_Overhead_%'].idxmin(), 'Concurrency']

print(f"\n5. Best CC performance: Concurrency={best_concurrency} (lowest overhead)")
print(f"6. Worst CC performance: Concurrency={worst_concurrency} (highest overhead)")

print("\n" + "=" * 100)

# Save results to CSV
overhead.to_csv('overhead_analysis.csv', index=False)
print("\nResults saved to: overhead_analysis.csv")

# Save summary to CSV
df_summary.to_csv('overhead_summary.csv', index=False)
print("Summary saved to: overhead_summary.csv")
