using CSV, DataFrames

# Load the dataset
df = DataFrames.DataFrame(CSV.File("C:/Users/sethl/OneDrive/Desktop/data/cleaned_small_df.csv"))

# Function to remove outliers based on IQR for specific columns
function remove_outliers(df, column)
    Q1 = quantile(df[!, column], 0.25)
    Q3 = quantile(df[!, column], 0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return filter(row -> lower_bound <= row[column] <= upper_bound, df)
end

# Clean the data by removing outliers
df_cleaned = df |> x -> remove_outliers(x, :P1_lpo_loRa) |> x -> remove_outliers(x, :P2_lpo_loRa)



CSV.write("C:/Users/sethl/OneDrive/Desktop/data/final_df.csv", df_cleaned)