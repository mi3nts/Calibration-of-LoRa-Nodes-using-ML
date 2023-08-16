using CSV, DataFrames

# Load the datasets
small_df = DataFrames.DataFrame(CSV.File("C:/Users/sethl/OneDrive/Desktop/data/small_df.csv"))
calibrate_df = DataFrames.DataFrame(CSV.File("C:/Users/sethl/OneDrive/Desktop/data/calibrate.csv"))

# Cleaning function
function clean_outliers!(df::DataFrame)
    # Temperature: Remove values above 45°C
    filter!(row -> row[:Temperature_loRa] <= 45, df)

    # Pressure: Assuming outliers are outside the range [90000, 120000] based on your input
    filter!(row -> 90000 <= row[:Pressure_loRa] <= 110000, df)

    # Humidity: Assuming outliers are outside the range [0, 100] as typical humidity values are percentages
    filter!(row -> 0 <= row[:Humidity_loRa] <= 100, df)
    
    return df
end

# Clean the datasets
clean_outliers!(small_df)
clean_outliers!(calibrate_df)

# Save the cleaned datasets (optional)
CSV.write("C:/Users/sethl/OneDrive/Desktop/SethRepo/firmware/data/cleaned_small_df.csv", small_df)
CSV.write("C:/Users/sethl/OneDrive/Desktop/SethRepo/firmware/data/cleaned_calibrate.csv", calibrate_df)