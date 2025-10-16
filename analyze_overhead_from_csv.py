#!/usr/bin/env python3
"""
Analyze performance overhead of Confidential Computing vs Standard Mode
from aggregated benchmark CSV files
"""

import pandas as pd
import numpy as np
import sys

def load_and_extract_data(csv_file):
    """Load CSV and extract relevant columns"""
    df = pd.read_csv(csv_file)

    # Extract relevant columns
    data = {
        'Concurrency': df['traffic_level'].astype(int).tolist(),
        'TTFT_p50': df['ttft_p50'].tolist(),
        'E2E_p50': df['e2e_p50'].tolist(),
        'Per_Request_TPS_p50': df['user_tps_p50'].tolist(),
        'Overall_TPS': df['summary_job_level_tps'].tolist()
    }

    return pd.DataFrame(data)

def calculate_overhead(df_non_cc, df_cc):
    """Calculate overhead percentage: ((CC - Non-CC) / Non-CC) * 100"""
    overhead = pd.DataFrame()
    overhead['Concurrency'] = df_non_cc['Concurrency']

    # For latency metrics (higher is worse)
    overhead['TTFT_Overhead_%'] = ((df_cc['TTFT_p50'] - df_non_cc['TTFT_p50']) / df_non_cc['TTFT_p50'] * 100).round(2)
    overhead['E2E_Overhead_%'] = ((df_cc['E2E_p50'] - df_non_cc['E2E_p50']) / df_non_cc['E2E_p50'] * 100).round(2)

    # For throughput metrics (lower is worse)
    overhead['Per_Request_TPS_Overhead_%'] = ((df_cc['Per_Request_TPS_p50'] - df_non_cc['Per_Request_TPS_p50']) / df_non_cc['Per_Request_TPS_p50'] * 100).round(2)
    overhead['Overall_TPS_Overhead_%'] = ((df_cc['Overall_TPS'] - df_non_cc['Overall_TPS']) / df_non_cc['Overall_TPS'] * 100).round(2)

    return overhead

def print_comparison_table(df_non_cc, df_cc):
    """Print side-by-side comparison"""
    print("\n" + "=" * 120)
    print("DETAILED COMPARISON: Standard Mode vs Confidential Computing")
    print("=" * 120)

    comparison = pd.DataFrame()
    comparison['Concurrency'] = df_non_cc['Concurrency']
    comparison['TTFT_NonCC'] = df_non_cc['TTFT_p50'].round(2)
    comparison['TTFT_CC'] = df_cc['TTFT_p50'].round(2)
    comparison['E2E_NonCC'] = df_non_cc['E2E_p50'].round(2)
    comparison['E2E_CC'] = df_cc['E2E_p50'].round(2)
    comparison['TPS_NonCC'] = df_non_cc['Per_Request_TPS_p50'].round(2)
    comparison['TPS_CC'] = df_cc['Per_Request_TPS_p50'].round(2)
    comparison['Overall_TPS_NonCC'] = df_non_cc['Overall_TPS'].round(2)
    comparison['Overall_TPS_CC'] = df_cc['Overall_TPS'].round(2)

    print(comparison.to_string(index=False))

def main():
    # File paths
    non_cc_file = 'aggregated_report_noncc.csv'
    cc_file = 'aggregated_report_cc.csv'

    try:
        # Load data
        print(f"Loading {non_cc_file}...")
        df_non_cc = load_and_extract_data(non_cc_file)

        print(f"Loading {cc_file}...")
        df_cc = load_and_extract_data(cc_file)

        # Ensure same concurrency levels
        if not df_non_cc['Concurrency'].equals(df_cc['Concurrency']):
            print("WARNING: Concurrency levels don't match between files!")

        # Print comparison table
        print_comparison_table(df_non_cc, df_cc)

        # Calculate overhead
        overhead = calculate_overhead(df_non_cc, df_cc)

        print("\n" + "=" * 120)
        print("CONFIDENTIAL COMPUTING (CC) PERFORMANCE OVERHEAD vs STANDARD MODE")
        print("=" * 120)
        print("\nPositive % = CC is slower (worse performance)")
        print("Negative % = CC is faster (better performance)\n")
        print(overhead.to_string(index=False))

        # Summary statistics
        print("\n" + "=" * 120)
        print("SUMMARY STATISTICS")
        print("=" * 120)

        summary = {
            'Metric': ['TTFT', 'End-to-End Latency', 'Per-Request TPS', 'Overall TPS'],
            'Min Overhead %': [
                round(overhead['TTFT_Overhead_%'].min(), 2),
                round(overhead['E2E_Overhead_%'].min(), 2),
                round(overhead['Per_Request_TPS_Overhead_%'].min(), 2),
                round(overhead['Overall_TPS_Overhead_%'].min(), 2)
            ],
            'Max Overhead %': [
                round(overhead['TTFT_Overhead_%'].max(), 2),
                round(overhead['E2E_Overhead_%'].max(), 2),
                round(overhead['Per_Request_TPS_Overhead_%'].max(), 2),
                round(overhead['Overall_TPS_Overhead_%'].max(), 2)
            ],
            'Mean Overhead %': [
                round(overhead['TTFT_Overhead_%'].mean(), 2),
                round(overhead['E2E_Overhead_%'].mean(), 2),
                round(overhead['Per_Request_TPS_Overhead_%'].mean(), 2),
                round(overhead['Overall_TPS_Overhead_%'].mean(), 2)
            ],
            'Median Overhead %': [
                round(overhead['TTFT_Overhead_%'].median(), 2),
                round(overhead['E2E_Overhead_%'].median(), 2),
                round(overhead['Per_Request_TPS_Overhead_%'].median(), 2),
                round(overhead['Overall_TPS_Overhead_%'].median(), 2)
            ]
        }

        df_summary = pd.DataFrame(summary)
        print(df_summary.to_string(index=False))

        # Key insights
        print("\n" + "=" * 120)
        print("KEY INSIGHTS")
        print("=" * 120)

        ttft_mean = overhead['TTFT_Overhead_%'].mean()
        e2e_mean = overhead['E2E_Overhead_%'].mean()
        tps_mean = overhead['Per_Request_TPS_Overhead_%'].mean()
        overall_tps_mean = overhead['Overall_TPS_Overhead_%'].mean()

        print(f"\n1. TTFT Overhead: {ttft_mean:.2f}% (avg)")
        if ttft_mean > 0:
            print(f"   → CC adds ~{abs(ttft_mean):.1f}% latency to first token")
        else:
            print(f"   → CC is ~{abs(ttft_mean):.1f}% faster at first token (better!)")

        print(f"\n2. End-to-End Overhead: {e2e_mean:.2f}% (avg)")
        if e2e_mean > 0:
            print(f"   → CC adds ~{abs(e2e_mean):.1f}% to total request time")
        else:
            print(f"   → CC is ~{abs(e2e_mean):.1f}% faster for total requests (better!)")

        print(f"\n3. Per-Request TPS: {tps_mean:.2f}% (avg)")
        if tps_mean < 0:
            print(f"   → CC reduces throughput by ~{abs(tps_mean):.1f}%")
        else:
            print(f"   → CC increases throughput by ~{abs(tps_mean):.1f}% (better!)")

        print(f"\n4. Overall TPS: {overall_tps_mean:.2f}% (avg)")
        if overall_tps_mean < 0:
            print(f"   → CC reduces aggregate throughput by ~{abs(overall_tps_mean):.1f}%")
        else:
            print(f"   → CC increases aggregate throughput by ~{abs(overall_tps_mean):.1f}% (better!)")

        # Best and worst case
        best_idx = overhead['Overall_TPS_Overhead_%'].idxmax()
        worst_idx = overhead['Overall_TPS_Overhead_%'].idxmin()

        best_concurrency = overhead.loc[best_idx, 'Concurrency']
        best_overhead = overhead.loc[best_idx, 'Overall_TPS_Overhead_%']
        worst_concurrency = overhead.loc[worst_idx, 'Concurrency']
        worst_overhead = overhead.loc[worst_idx, 'Overall_TPS_Overhead_%']

        print(f"\n5. Best CC performance: Concurrency={best_concurrency} (overhead: {best_overhead:.2f}%)")
        print(f"6. Worst CC performance: Concurrency={worst_concurrency} (overhead: {worst_overhead:.2f}%)")

        # Overall conclusion
        print("\n" + "=" * 120)
        print("CONCLUSION")
        print("=" * 120)

        avg_overhead = overhead[['TTFT_Overhead_%', 'E2E_Overhead_%', 'Overall_TPS_Overhead_%']].mean().mean()

        if abs(avg_overhead) < 5:
            print(f"\n✓ CC has MINIMAL performance impact (~{abs(avg_overhead):.1f}% average overhead)")
            print("  Confidential Computing provides strong security with negligible performance cost.")
        elif abs(avg_overhead) < 15:
            print(f"\n✓ CC has LOW performance impact (~{abs(avg_overhead):.1f}% average overhead)")
            print("  Confidential Computing provides good security-performance tradeoff.")
        else:
            print(f"\n! CC has MODERATE performance impact (~{abs(avg_overhead):.1f}% average overhead)")
            print("  Consider the security benefits vs performance cost for your use case.")

        print("\n" + "=" * 120)

        # Save results to CSV
        overhead.to_csv('overhead_analysis_from_csv.csv', index=False)
        print("\nDetailed results saved to: overhead_analysis_from_csv.csv")

        df_summary.to_csv('overhead_summary_from_csv.csv', index=False)
        print("Summary statistics saved to: overhead_summary_from_csv.csv")

        # Save comparison table
        comparison = pd.DataFrame()
        comparison['Concurrency'] = df_non_cc['Concurrency']
        comparison['NonCC_TTFT'] = df_non_cc['TTFT_p50']
        comparison['CC_TTFT'] = df_cc['TTFT_p50']
        comparison['NonCC_E2E'] = df_non_cc['E2E_p50']
        comparison['CC_E2E'] = df_cc['E2E_p50']
        comparison['NonCC_TPS'] = df_non_cc['Per_Request_TPS_p50']
        comparison['CC_TPS'] = df_cc['Per_Request_TPS_p50']
        comparison['NonCC_Overall_TPS'] = df_non_cc['Overall_TPS']
        comparison['CC_Overall_TPS'] = df_cc['Overall_TPS']
        comparison.to_csv('comparison_table.csv', index=False)
        print("Side-by-side comparison saved to: comparison_table.csv")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(f"Please ensure both CSV files exist in the current directory:")
        print(f"  - {non_cc_file}")
        print(f"  - {cc_file}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
