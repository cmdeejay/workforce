import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from pandasai import SmartDataframe
from scipy.stats import pearsonr
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pandasai.llm import OpenAI

"""
Input:
- Weekly workload data 2023

Statistical methods:
- Descriptive statistics
- Kurtosis and skewness
- Normalized trend line
- Histogram
- Standardized histogram
- Correlation coefficient
- Seasonal decomposition
- Augmented Dickey-Fuller test

Data cleaning:
- Reshape data
- Standardize data
- Normalize data
- Remove duplicate rows
- Remove rows with missing values
- Remove rows with outliers
- Remove rows with invalid values
- Remove rows with invalid data types
- Remove rows with invalid formats
- Remove rows with invalid characters
- Remove rows with invalid length
- Remove rows with invalid ranges
- Remove rows with invalid patterns
"""

selected_df = pd.read_csv(
    "Workload/weekly_data.csv", parse_dates=["Date"], index_col="Date"
)

# Step 2: Filter the dataframe for the desired year (2023)
df_2023 = selected_df[selected_df.index.year == 2023]
df_2023.to_csv("2023.csv")
# Step 3: Perform further analysis or retrieve the desired data
# Example: Retrieve all data from 2023
data_2023 = df_2023


def normalize_data(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    normalized_data = scaler.transform(data)
    normalized_data = normalized_data.flatten()
    return normalized_data


def reshape_data(data):
    data = data.values.reshape(-1, 1)
    return data


def standardize_data(data):
    scaler = StandardScaler()
    scaler.fit(data)
    standardized_data = scaler.transform(data)
    standardized_data = standardized_data.flatten()
    return standardized_data


def seasonal_decomposition(data, column):
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.set_index("Date")
    result = seasonal_decompose(data[column], model="multiplicative")
    result.plot()
    plt.show()


def standardized_histogram(data):
    df = data.iloc[:, 2:16]
    df = df[
        [
            "Manual Transactions",
            "FXTM Manual Transactions",
            "Alpari Manual Transactions",
        ]
    ]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for i, column in enumerate(df.columns):
        n, bins, patches = axs[i].hist(df[column])
        axs[i].set_title(column)

        for count, patch in zip(n, patches):
            axs[i].annotate(
                f"{int(count)}",
                xy=(patch.get_x() + patch.get_width() / 2, patch.get_height()),
                xycoords="data",
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    # Adjust spacing between subplots
    # plt.tight_layout()

    # Display the histograms
    plt.show()


def histogram(df):
    df = df.iloc[:, 0:14]
    fig, axs = plt.subplots(1, 14, figsize=(70, 4))
    for i, column in enumerate(df.columns):
        n, bins, patches = axs[i].hist(df[column])
        axs[i].set_title(column)

        for count, patch in zip(n, patches):
            axs[i].annotate(
                f"{int(count)}",
                xy=(patch.get_x() + patch.get_width() / 2, patch.get_height()),
                xycoords="data",
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )
    # Adjust spacing between subplots
    plt.tight_layout()

    # Display the histograms
    plt.show()


def descriptive_statistics(df):
    df.describe().to_csv("workforce_descriptive_statistics.csv")
    print(df.describe())


def kurtosis_and_skewness(df):
    kurtosis_values = df.kurtosis()
    skewness_values = df.skew()
    print(f"Kurtosis: {kurtosis_values}")
    print(f"Skewness: {skewness_values}")


def normalized_trend_line(df):
    for column in df.columns:
        df[f"{column}_normalized"] = normalize_data(reshape_data(df[column]))
        df[f"{column}_standardized"] = standardize_data(reshape_data(df[column]))
    # print(df)
    df.to_csv("workforce_normalized_data.csv")

    # df["Manual Transactions_normalized"].plot()
    # df["Emails_normalized"].plot()
    # df["Jira_normalized"].plot()
    # df["Documents_normalized"].plot()

    # df["Manual Transactions_normalized"].plot()
    # df["FXTM Manual Transactions_normalized"].plot()
    # df["Alpari Manual Transactions_normalized"].plot()

    # df["Jira_normalized"].plot()
    # df["FXTM Jira_normalized"].plot()
    # df["Alpari Jira_normalized"].plot()
    # df["OB Jira_normalized"].plot()

    # df["Documents_normalized"].plot()
    # df["FXTM Documents_normalized"].plot()
    # df["Alpari Documents_normalized"].plot()
    # df["OB Documents_normalized"].plot()

    # df["FXTM Manual Transactions_normalized"].plot()
    # df["FXTM Jira_normalized"].plot()
    # df["FXTM Documents_normalized"].plot()
    # df["FXTM Emails_normalized"].plot()

    # df["Alpari Manual Transactions_normalized"].plot()
    # df["Alpari Jira_normalized"].plot()
    # df["Alpari Documents_normalized"].plot()
    # df["Alpari Emails_normalized"].plot()

    df["OB Jira_normalized"].plot()
    df["OB Documents_normalized"].plot()
    # df["OB Emails_normalized"].plot()

    plt.xlabel("Time")
    plt.ylabel("Normalized Variable Y")
    plt.legend()
    plt.title("2023 Workload Normalized")
    plt.show()


def adfuller_test(data):
    for column in data.columns:
        try:
            result = adfuller(data[column].values)
            print(f"{column} ADF Test")
            print(f"ADF Statistic: {result[0]}")
            print(f"p-value: {result[1]}")
            print("Critical Values:")
            for key, value in result[4].items():
                print(f"\t{key}: {value}")
        except:
            pass


def correlation_coefficient(df):
    correlation_matrix = pd.DataFrame(index=df.columns, columns=df.columns)

    # Loop through all pairs of columns
    for col1 in df.columns:
        for col2 in df.columns:
            # Calculate correlation coefficient and p-value
            try:
                corr, p_value = pearsonr(df[col1], df[col2])
            except:
                pass

            # Store correlation coefficient and p-value in the matrix
            correlation_matrix.loc[col1, col2] = f"{corr:.2f} ({p_value:.2f})"

    # Print the correlation coefficient matrix
    print(correlation_matrix)
    correlation_matrix.to_csv("workforce_correlation_matrix.csv")


# descriptive_statistics(selected_df)
# kurtosis_and_skewness(selected_df)
# adfuller_test(selected_df)
# seasonal_decomposition(df, 'Total Wd')
# histogram(data_2023)
# standardized_histogram(selected_df)
# normalized_trend_line(data_2023)
# correlation_coefficient(selected_df)


def tl_documents():
    df = pd.read_csv("Workload\Original\documents_original.csv")
    df["Processed Week"] = pd.to_datetime(df["Processed Week"], format="%d-%m-%y")
    df["Submit ts"] = pd.to_datetime(df["Submit ts"], format="%d/%m/%Y %H:%M:%S")
    df["Submit ts"] = df["Submit ts"].dt.date

    df.to_csv("Workload\Cleaned\documents_cleaned.csv", index=False)
    print(df.head())


def tl_transactions():
    df = pd.read_csv("Workload\Original\manual_transactions_original.csv")
    df["Processed Week"] = pd.to_datetime(df["Processed Week"], format="%d-%m-%y")
    df["Create Week"] = pd.to_datetime(df["Create Week"], format="%d-%m-%y")

    df.to_csv("Workload\Cleaned\manual_transactions_cleaned.csv", index=False)
    print(df.head())


def tl_emails():
    df = pd.read_csv("Workload\Original\emails_original.csv")
    df["Processed Week"] = pd.to_datetime(df["Processed Week"], format="%d-%m-%y")

    df.to_csv("Workload\Cleaned\emails_cleaned.csv", index=False)
    print(df.head())


# df1 = pd.read_excel("documents.xlsx", sheet_name="Sheet2")
# df["Processed Week"] = pd.to_datetime(df["Processed Week"],format='%d-%m-%y')
# df1["Processed Week"] = pd.to_datetime(df1["Processed Week"],format='%d-%m-%y')
# df2 = pd.concat([df, df1])
# df2.to_csv("documents_cleaned.csv")
