import pandas as pd

CSV_PATH = "Weather_2025.csv"

def check_bias():
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    city = "Quetta"
    df_city = df[df['city'] == city]
    
    df_ref = df_city[df_city['year'] < 2025]
    df_curr = df_city[df_city['year'] >= 2025]
    
    # Seasonality correction
    curr_months = df_curr['month'].unique()
    df_ref = df_ref[df_ref['month'].isin(curr_months)]
    
    print(f"--- {city} Bias Correction ---")
    with open("bias_output.txt", "w") as f:
        for col in ['tmin', 'tmax', 'wspd', 'pressure']:
            ref_mean = df_ref[col].mean()
            curr_mean = df_curr[col].mean()
            diff = curr_mean - ref_mean
            f.write(f"{col}: Ref={ref_mean:.2f}, Curr={curr_mean:.2f}, Diff={diff:.2f}\n")
            print(f"{col}: Ref={ref_mean:.2f}, Curr={curr_mean:.2f}, Diff={diff:.2f}")

if __name__ == "__main__":
    check_bias()
