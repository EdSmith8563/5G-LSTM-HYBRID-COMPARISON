import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATASET = 'DATA/FULL_FEATURE_SET/COMBINED_DATASET.csv'

def null_value_counts(df):
    null_counts = df.isnull().sum()
    max_len = max(len(col) for col in null_counts.index)
    header = f"{'Column'.ljust(max_len)} | {'Null Count':>10}"
    separator = "-" * len(header)
    print(f"\n{separator}")
    print(header)
    print(separator)
    for col, cnt in null_counts.items():
        print(f"{col.ljust(max_len)} | {cnt:10d}")

def value_ranges(df):
    numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
    max_len = max(len(col) for col in numeric_cols)
    header = f"{'Column'.ljust(max_len)} | {'Min':>12} | {'Max':>12}"
    separator = "-" * len(header)
    print(f"\n{separator}")
    print(header)
    print(separator)
    for col in numeric_cols:
        mn = df[col].min()
        mx = df[col].max()
        print(f"{col.ljust(max_len)} | {mn:12.2f} | {mx:12.2f}")

def rsrp_ranges(df):
    thresholds = [-200, -180, -160, -140, -120, -100, -80, -60, 0]
    labels = [f"{thresholds[i]} to {thresholds[i+1]}" for i in range(len(thresholds)-1)]
    df['RSRP_Range'] = pd.cut(df['RSRP'], bins=thresholds, labels=labels, right=False)
    range_counts = df['RSRP_Range'].value_counts().sort_index()
    max_label = max(len(str(l)) for l in labels)
    header = f"{'RSRP Range'.ljust(max_label)} | {'Instances':>10}"
    separator = "-" * len(header)
    print(f"\n{separator}")
    print(header)
    print(separator)
    for label in labels:
        cnt = range_counts.get(label, 0)
        print(f"{str(label).ljust(max_label)} | {cnt:10d}")
    mean_rsrp = df['RSRP'].mean()
    print(f"\nMean RSRP Value: {mean_rsrp:.2f}")

def plot_correlation_heatmap(df):
    num_df = df.select_dtypes(include=[float, int])
    exclude = ['RAWCELLID', 'Day']
    num_df = num_df.drop(columns=[c for c in exclude if c in num_df.columns], errors='ignore')
    corr = num_df.corr()
    labels = [f'$\\mathbf{{{c}}}$' if c == 'RSRP' else c for c in corr.columns]
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=labels, yticklabels=labels)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

def plot_CSI_series(df, fraction=0.05):
    df_temp = df.copy()
    df_temp['Timestamp'] = pd.to_datetime(df_temp['Timestamp'])
    df_temp.sort_values('Timestamp', inplace=True)
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    metrics = ['RSRP', 'RSRQ', 'SNR', 'CQI']
    for ax, metric in zip(axs.flat, metrics):
        n = len(df_temp)
        subset_len = int(np.ceil(fraction * n))
        subset = df_temp.iloc[:subset_len]
        ax.plot(subset.index, subset[metric], marker='o', label=metric)
        ax.set_title(f"{metric} ({fraction*100:.2f}% of dataset)")
        ax.set_xlabel("Record Number")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)
        step = max(1, subset_len // 10)
        xticks = np.arange(0, subset_len, step)
        xtick_labels = [subset.loc[i, 'Timestamp'].strftime('%Y-%m-%d %H:%M:%S') for i in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def main():
    df = pd.read_csv(DATASET, na_values=['-'])
    null_value_counts(df)
    value_ranges(df)
    rsrp_ranges(df)
    plot_correlation_heatmap(df)
    plot_CSI_series(df, fraction=0.002)

if __name__ == "__main__":
    main()
