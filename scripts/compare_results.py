#!/usr/bin/env python3

import pandas as pd
import sys

def compare_experiments(file1, file2, name1="Without fix", name2="With fix"):
    """Compare two experiment results and show the differences."""
    
    # Read the TSV files
    df1 = pd.read_csv(file1, sep='\t')
    df2 = pd.read_csv(file2, sep='\t')
    
    print(f"=== Comparison: {name1} vs {name2} ===\n")
    
    # Ensure both have the same subsections
    if not df1['Subsection'].equals(df2['Subsection']):
        print("WARNING: Different subsections in the two files!")
        return
    
    print(f"{'Subsection':<12} {'Query Time':<15} {'Recall':<15} {'RR@10':<15} {'Memory':<15} {'Build Time':<15}")
    print(f"{'='*12} {'='*15} {'='*15} {'='*15} {'='*15} {'='*15}")
    
    for i, subsection in enumerate(df1['Subsection']):
        qt1, qt2 = df1.iloc[i]['Query Time (microsecs)'], df2.iloc[i]['Query Time (microsecs)']
        recall1, recall2 = df1.iloc[i]['Recall'], df2.iloc[i]['Recall']
        rr1, rr2 = df1.iloc[i]['RR@10'], df2.iloc[i]['RR@10']
        mem1, mem2 = df1.iloc[i]['Memory Usage (Bytes)'], df2.iloc[i]['Memory Usage (Bytes)']
        bt1, bt2 = df1.iloc[i]['Building Time (secs)'], df2.iloc[i]['Building Time (secs)']
        
        qt_diff = ((qt2 - qt1) / qt1) * 100 if qt1 != 0 else 0
        recall_diff = ((recall2 - recall1) / recall1) * 100 if recall1 != 0 else 0
        rr_diff = ((rr2 - rr1) / rr1) * 100 if rr1 != 0 else 0
        mem_diff = ((mem2 - mem1) / mem1) * 100 if mem1 != 0 else 0
        bt_diff = ((bt2 - bt1) / bt1) * 100 if bt1 != 0 else 0
        
        print(f"{subsection:<12} {qt_diff:>+7.1f}%      {recall_diff:>+7.3f}%      {rr_diff:>+7.3f}%      {mem_diff:>+7.1f}%      {bt_diff:>+7.1f}%")
    
    print(f"\n{'='*100}")
    print("Summary of changes (positive = improvement with fix):")
    
    # Calculate averages
    avg_qt_diff = sum(((df2.iloc[i]['Query Time (microsecs)'] - df1.iloc[i]['Query Time (microsecs)']) / df1.iloc[i]['Query Time (microsecs)']) * 100 for i in range(len(df1))) / len(df1)
    avg_recall_diff = sum(((df2.iloc[i]['Recall'] - df1.iloc[i]['Recall']) / df1.iloc[i]['Recall']) * 100 for i in range(len(df1))) / len(df1)
    avg_rr_diff = sum(((df2.iloc[i]['RR@10'] - df1.iloc[i]['RR@10']) / df1.iloc[i]['RR@10']) * 100 for i in range(len(df1))) / len(df1)
    avg_mem_diff = (df2.iloc[0]['Memory Usage (Bytes)'] - df1.iloc[0]['Memory Usage (Bytes)']) / df1.iloc[0]['Memory Usage (Bytes)'] * 100
    avg_bt_diff = (df2.iloc[0]['Building Time (secs)'] - df1.iloc[0]['Building Time (secs)']) / df1.iloc[0]['Building Time (secs)'] * 100
    
    print(f"Average Query Time change: {avg_qt_diff:+.1f}% ({'faster' if avg_qt_diff < 0 else 'slower'})")
    print(f"Average Recall change: {avg_recall_diff:+.3f}%")
    print(f"Average RR@10 change: {avg_rr_diff:+.3f}%")
    print(f"Memory Usage change: {avg_mem_diff:+.1f}%")
    print(f"Building Time change: {avg_bt_diff:+.1f}%")
    
    print(f"\nDetailed Analysis:")
    print(f"- Memory usage increased by {df2.iloc[0]['Memory Usage (Bytes)'] - df1.iloc[0]['Memory Usage (Bytes)']:,} bytes")
    print(f"- Building time increased by {df2.iloc[0]['Building Time (secs)'] - df1.iloc[0]['Building Time (secs)']} seconds")
    
    # Check for stability improvements
    qt_std1 = df1['Query Time (microsecs)'].std()
    qt_std2 = df2['Query Time (microsecs)'].std()
    print(f"- Query time standard deviation: {qt_std1:.1f} → {qt_std2:.1f} ({'more stable' if qt_std2 < qt_std1 else 'less stable'})")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_results.py <file_without_fix> <file_with_fix>")
        sys.exit(1)
    
    compare_experiments(sys.argv[1], sys.argv[2])