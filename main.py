# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_presale_data(path: str) -> pd.DataFrame:
    # Load a specific sheet by name
    presale_df = pd.read_excel(path,
                       sheet_name='lhinu_bsc_presale_buys_26487457')

    # Dropping some values, and retyping the
    presale_df = presale_df[~presale_df['network'].isin(['MAX', 'SUM', 'MIN'])]
    presale_df.dropna(subset=['value_in_usd', 'timestamp'], inplace=True)
    presale_df['value_in_usd'] = presale_df['value_in_usd'].astype(float)
    presale_df['datetime_utc'] = pd.to_datetime(presale_df['datetime_utc'])

    return presale_df

def hist_of_values_in_usd(df: pd.DataFrame) -> int:
    # Histogram of Value in USD
    plt.hist(df['value_in_usd'],edgecolor="red", bins=10, color='darkgreen')  # Adjust the number of bins as needed
    plt.title('Histogram of Value in USD')
    plt.xlabel('Value in USD')
    plt.ylabel('Number of transfers')
    # plt.xscale('log')
    plt.yscale('log')

    # Save the plot
    plt.savefig('../Presale_plots/histogram_value_in_usd_log.png')
    # plt.show()
    return 0

def hist_of_timestamps(df: pd.DataFrame) -> int:
    # Histogram of Value in USD
    plt.hist(df['timestamp'], edgecolor="red", bins=3, color='skyblue')  # Adjust the number of bins as needed
    plt.title('Histogram of Timestamps')
    plt.xlabel('timestamp')
    plt.ylabel('Number of transfers')
    # plt.xscale('log')
    # plt.yscale('log')

    # Save the plot
    plt.savefig('../Presale_plots/histogram_timestamps.png')
    # plt.show()
    return 0

def hist_3d(df: pd.DataFrame) -> int:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = df['timestamp']  # Time on x-axis
    y = df['value_in_usd']  # Value on y-axis
    hist, xedges, yedges = np.histogram2d(x, y, bins=4)

    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.ravel()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', cmap='viridis')

    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Value in USD')
    ax.set_zlabel('Number of Transfers')

    plt.savefig('../Presale_plots/3d_histogram.png')
    plt.show()
    return 0




df = load_presale_data('../DATA_Love_Hate_Inu_presale.xlsx')
hist_of_values_in_usd(df)
hist_of_timestamps(df)
hist_3d(df)
