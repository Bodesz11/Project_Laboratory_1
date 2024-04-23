# Imports
import pandas as pd
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import json
import os
import openpyxl

import warnings
warnings.filterwarnings("ignore")



class Love_Hate_LNU_analiser:
    def __init__(self, data_path: str, data_sheet_in: str, data_sheet_out: str = '', network: str = ''):
        self.data_path = data_path
        self.data_sheet = data_sheet_in

        # Autmatically load the given data
        self.possible_networks = ['ETH', 'BSC']
        self.data_transfers_in = self.load_reference(data_path, data_sheet_in)
        self.data_transfers_out = self.load_reference(data_path, data_sheet_out)

        self.network_type = self.data_transfers_in['network'][0]
        if self.network_type not in self.possible_networks:
            warnings.warn('The network type is not defined!')
        self.network_type = network + self.network_type

    def load_reference(self, data_path: str, data_sheet: str):
        df = pd.read_excel(data_path, sheet_name=data_sheet)
        df = self.reformat_input_data(df)
        return df

    def reformat_input_data(self, df: pd.DataFrame):
        if 'payment_wallet' in df.keys():
            # incoming payments
            df['payment_wallet'] = df['payment_wallet'].fillna(method='bfill')
            df['payment_wallet'] = df['payment_wallet'].astype(str).str.lower()
        df['from_address'] = df['from_address'].astype(str).str.lower()
        df['to_address'] = df['to_address'].astype(str).str.lower()
        df['value_in_usd'] = df['value_in_usd'].astype(float)

        # Dropping min/max
        df = df[df['network'].isin(self.possible_networks)]

        return df

    def make_all_plots(self):
        # self.all_transaction_values(self.data_transfers_in, self.data_transfers_out)
        # self.common_buyers(self.data_transfers_in, bins=50)
        # self.most_amount_buyers(self.data_transfers_in, bins=50)
        # self.wallet_transfer_amount_and_count(self.data_transfers_out, self.data_transfers_in)
        self.time_factor(self.data_transfers_out, self.data_transfers_in)

    def all_transaction_values(self, data_transfers_in, data_transfers_out):
        # Logaritmus miatt meg kell szűrni az adatokat
        filtered_in_transfers = data_transfers_in[data_transfers_in['value_in_usd'] > 0]['value_in_usd']
        filtered_out_transfers = data_transfers_out[data_transfers_out['value_in_usd'] > 0]['value_in_usd']

        # Making the plot
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))

        axes[0].hist(np.log10(filtered_in_transfers), bins=50, color='skyblue', edgecolor='black')
        axes[0].set_title('Distribution of Transaction Values (Presale buy)')
        axes[0].set_xlabel('Transaction Value (USD)')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(axis='y', alpha=0.75)

        x_formatter = FuncFormatter(lambda x, pos: '${:,.0f}'.format(10 ** x))
        axes[0].xaxis.set_major_formatter(x_formatter)

        y_formatter = FuncFormatter(lambda x, pos: '{:,.0f}'.format(x))
        axes[0].yaxis.set_major_formatter(y_formatter)

        ticks = [1, 2, 3, 4, 5, 6]
        axes[0].set_xticks(ticks)

        axes[1].hist(np.log10(filtered_out_transfers), bins=50, color='lightgreen', edgecolor='black')
        axes[1].set_title('Distribution of Transaction Values (Wallet transfers)')
        axes[1].set_xlabel('Transaction Value (USD)')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(axis='y', alpha=0.75)
        axes[1].xaxis.set_major_formatter(x_formatter)
        axes[1].set_xticks(ticks)
        plt.tight_layout()
        plt.savefig(f'../Presale_plots/{self.network_type}all_transaction_values.pdf')
        plt.show()

    def plot_distribution(self, unique_addresses, values, ylabel, title, yticks, formatter, filename):
        plt.figure(figsize=(10, 6))
        plt.scatter(unique_addresses, values, color='skyblue', edgecolor='black')
        plt.title(title)
        plt.xlabel('')
        plt.ylabel(ylabel)
        plt.grid(axis='y', alpha=0.75)
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.yticks(yticks)
        plt.xticks([])
        plt.savefig(filename)
        plt.show()

    def common_buyers(self, data_transfers_in, bins=50):
        data_transfers_in['address_pair'] = data_transfers_in.apply(
            lambda row: tuple(sorted([row['from_address'], row['to_address']])), axis=1)

        transaction_counts = data_transfers_in.groupby('address_pair').size()
        transaction_counts_df = transaction_counts.reset_index(name='transaction_count')

        top_n_transactions = transaction_counts_df.nlargest(bins, 'transaction_count')
        top_n_transactions[['from_address', 'to_address']] = top_n_transactions['address_pair'].apply(pd.Series)
        top_n_transactions.drop('address_pair', axis=1, inplace=True)

        unique_addresses = list(
            set(top_n_transactions['to_address'].values.tolist() + top_n_transactions['from_address'].values.tolist()))[
                           :-1]

        values = np.log10(top_n_transactions['transaction_count'])
        ylabel = 'Frequency'
        title = 'Distribution of number of transactions per addresses'
        formatter = FuncFormatter(lambda x, pos: '{:,.0f}'.format(10 ** x))
        yticks = [1, 2, 3, 4]
        filename = f'../Presale_plots/{self.network_type}_top_frequency.pdf'

        self.plot_distribution(unique_addresses, values, ylabel, title, yticks, formatter, filename)

    def most_amount_buyers(self, data_transfers_in, bins=50):
        data_transfers_in['address_pair'] = data_transfers_in.apply(
            lambda row: tuple(sorted([row['from_address'], row['to_address']])), axis=1)
        sum_of_value_of_transactions = data_transfers_in.groupby('address_pair')['value_in_usd'].sum().reset_index()

        top_n_value = sum_of_value_of_transactions.nlargest(bins, 'value_in_usd')
        top_n_value[['from_address', 'to_address']] = top_n_value['address_pair'].apply(pd.Series)
        top_n_value.drop('address_pair', axis=1, inplace=True)

        unique_addresses = list(
            set(top_n_value['to_address'].values.tolist() + top_n_value['from_address'].values.tolist()))[:-1]

        values = np.log10(top_n_value['value_in_usd'])
        ylabel = 'Value'
        title = 'Distribution of biggest value of transactions per addresses'
        formatter = FuncFormatter(lambda x, pos: '${:,.0f}'.format(10 ** x))
        yticks = [4, 5, 6]
        filename = f'../Presale_plots/{self.network_type}_top_values.pdf'

        self.plot_distribution(unique_addresses, values, ylabel, title, yticks, formatter, filename)

    def wallet_transfer_amount_and_count(self, data_transfers_out, data_transfers_in):
        unique_payment_wallets = data_transfers_in['payment_wallet'].unique()
        filtered_wallet_transfers = data_transfers_out[data_transfers_out['from_address'].isin(unique_payment_wallets)]

        # Idő szerint mikor mentek az utalások és mekkora összeggel
        transfer_summary = filtered_wallet_transfers.groupby(['from_address', 'to_address']).agg(
            {'value_in_usd': ['count', 'sum'],
             'timestamp': ['min', 'max']}).reset_index()

        transfer_summary.columns = ['From Address', 'To Address', 'Transaction Count', 'Total Value USD',
                                    'Earliest Transaction', 'Latest Transaction']

        transfer_summary['Earliest Transaction'] = pd.to_datetime(transfer_summary['Earliest Transaction'], unit='s')
        transfer_summary['Latest Transaction'] = pd.to_datetime(transfer_summary['Latest Transaction'], unit='s')

        # TODO ????unique_addresses = list(set(transfer_summary['From Address'].values.tolist() + transfer_summary['To Address'].values.tolist()))[:-1]
        unique_addresses = list(transfer_summary['To Address'].values.tolist())

        # Létrehozni a plotot
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))

        # Elnevezzük csak a dolgokat + itt csinálunk log-os átalakítást
        transfer_summary.sort_values('Total Value USD', inplace=True, axis=0, ascending=False)
        print('unique: ', len(unique_addresses), 'transfer_sum: ', len(transfer_summary))
        axes[0].scatter(unique_addresses, np.log10(transfer_summary['Total Value USD']), color='skyblue',
                        edgecolor='black')
        axes[0].set_title('Distribution of biggest value of transactions per addresses (Wallet transfers)')
        axes[0].set_xlabel('')
        axes[0].set_ylabel('Value')
        axes[0].grid(axis='y', alpha=0.75)

        formatter = FuncFormatter(lambda x, pos: '${:,.0f}'.format(10 ** x))
        axes[0].yaxis.set_major_formatter(formatter)

        # X tengelyt még át kell alakítani, a ticks adja meg, hogy a 10 a hányadikonok legyenek a tengelyen
        ticks = [2, 3, 4, 5, 6, 7]
        axes[0].set_yticks(ticks)
        axes[0].set_xticks('')

        # Teljesen ugyanaz mint előbb
        transfer_summary.sort_values('Transaction Count', inplace=True, axis=0, ascending=False)
        axes[1].scatter(unique_addresses, transfer_summary['Transaction Count'], color='skyblue', edgecolor='black')
        axes[1].set_title('Distribution of number of transactions per addresses (Wallet transfers)')
        axes[1].set_xlabel('')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(axis='y', alpha=0.75)
        axes[1].set_xticks('')

        # Ábrázolás
        plt.tight_layout()
        plt.savefig(f'../Presale_plots/{self.network_type}_wallet_transfers_stats.pdf')
        plt.show()

    def time_factor(self, data_transfers_out, data_transfers_in):
        bins = 50
        bins += 1
        X_in = data_transfers_in['timestamp']
        X_out = data_transfers_out['timestamp']
        time_min = min(data_transfers_out['timestamp'].min(), data_transfers_in['timestamp'].min())
        time_max = max(data_transfers_out['timestamp'].max(), data_transfers_in['timestamp'].max())
        times = np.arange(time_min, time_max, (time_max - time_min) / bins)

        Y_in = data_transfers_in['value_in_usd']
        Y_out = data_transfers_out['value_in_usd']

        # Aggregating the data
        Y_in_aggregated = [sum([Y_in[j] for j in range(len(X_in)) if times[i] <= X_in[j] < times[i+1]])
                           for i in range(len(times) - 1)]
        Y_out_aggregated = [sum([Y_out[j] for j in range(len(X_out)) if times[i] <= X_out[j] < times[i+1]])
                           for i in range(len(times) - 1)]

        # Plot bars
        plt.plot(times[:-1], Y_in_aggregated, color='red', label='Income')
        plt.plot(times[:-1], Y_out_aggregated, color='blue', label='Outcome')

        # Add labels and legend
        plt.xlabel('Timestamp')
        plt.ylabel('Value (USD)')
        plt.title('Income and Outcome over Time')
        plt.legend()
        plt.tight_layout()

        # Show plot
        plt.savefig(f'../Presale_plots/{self.network_type}time_factor.pdf')
        plt.show()

data_path = '../Data/DATA_Love_Hate_Inu_presale.xlsx'
data_sheet_out = 'lhinu_payment_wallet_transfers_'
data_sheet_in = 'lhinu_presale_buy_result_167775'
network = 'Love_Hate_LNU'

analiser = Love_Hate_LNU_analiser(data_path=data_path,
                                  data_sheet_in=data_sheet_in,
                                  data_sheet_out=data_sheet_out,
                                  network=network)
analiser.make_all_plots()