using DataFrames, CSV, Dates

include("CleanData.jl")

#get all LoRa csv files
dir = readdir("C:/Users/sethl/OneDrive/Desktop/data/LoRa Data"; join=true)



#Delete CSV files that dont have humidity, temperature, and pressure data
filter!(a -> !occursin("477b41f20047002e", a), dir)

#Sort CSV files by date
global row_names = Vector{Date}()
for data in dir
    name = string(split.(data, "_")[2, 1])
    name = string(split.(name, ".")[1, 1])
    name = Dates.Date(name, "yyyy-mm-dd")
    global row_names = push!(row_names, name)
end

sort!(row_names)

global df = DataFrames.DataFrame(dateTime = String[], NH3 = Float64[], CO2 = Float64[], NO2 = Float64[], C3H8 = Float64[], C4H10 = Float64[], CH4 = Float64[], H2 = Float64[],
C2H5OH = Float64[], P1_lpo = Float64[], P1_ratio = Float64[], P1_conc = Float64[], P2_lpo = Float64[], P2_ratio = Float64[], P2_conc = Float64[], Temperature = Float64[],
Pressure = Float64[], Humidity = Float64[])

#Populates dataframe. Errors indicate that the data is corrupt and will be thrown out. Data is automatically cleaned out.
for name in union(Dates.format.(row_names, "yyyy-mm-dd"))
    for i in 1:length(dir)
        if occursin(name, dir[i])
            try
                append!(df, DataFrames.DataFrame(CSV.File(dir[i])), cols = :intersect, promote = false)
            catch e
            end
        end
    end
end

#Rename LoRa column row_names
for col in names(df)
    if col != "dateTime"
        rename!(df, col => col * "_loRa")
    end
end

#turn dateTime from string into dateTime object
rename!(df, :dateTime => :dateString)
df[!, :dateTime] = Vector{DateTime}(undef, nrow(df))
delete_row_num = Vector{Int}()
for (index, row) in enumerate(eachrow(df))
    try
        df[index, :dateTime] = Dates.DateTime(df[index, :dateString], "yyyy-mm-dd HH:MM:SS")
    catch e
        push!(delete_row_num, index)
    end
end

#delete rows that had corrupted dateTimes and delete dateString value
delete!(df, delete_row_num)
df = select!(df, Not(:dateString))

#Average and round LoRa Values to the nearest 30 seconds
for index in 1:nrow(df)
    
    df[index, :dateTime] = round(df[index, :dateTime], Dates.Second(30))

end
unique!(df, :dateTime)


#Combine Palas and LoRa Data

palas_data = PalasData()
global final_df = DataFrames.DataFrame()
for data in palas_data
    combined_df = leftjoin(data, df, on = :dateTime)
    global final_df = vcat(final_df, combined_df)
end

dropmissing!(final_df, :NH3_loRa)
CSV.write("C:/Users/sethl/OneDrive/Desktop/SethRepo/firmware/combined_df.csv", final_df)

