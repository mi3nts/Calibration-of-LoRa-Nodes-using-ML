using DataFrames, CSV, Dates
include("C:/Users/sethl/OneDrive/Desktop/SethRepo/firmware/PalasData.jl")
#get all LoRa csv files
dir = readdir("C:/Users/sethl/OneDrive/Desktop/data/LoRa Data"; join=true)
#df_original = DataFrames.DataFrame(CSV.File("C:/Users/sethl/OneDrive/Desktop/data/calibrate.csv")) 

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
Pressure = Float64[], Humidity = Float64[], Latitude = Float64[], Longitude = Float64[])

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
df.dateTime = round.(df.dateTime, Dates.Second(30))


#Check Longitude and Latitude to make sure the LoRa Node is housed outside at waterview, then drop lat values
df = df[(df[:, :Latitude_loRa] .>= 32.990) .& (df[:, :Latitude_loRa] .<= 32.995) .&
    (df[:, :Longitude_loRa] .>= -96.760) .& (df[:, :Longitude_loRa] .<= -96.755), :]

select!(df, Not([:Latitude_loRa, :Longitude_loRa]))


#Get unique only by dateTime
unique!(df, :dateTime)

#Combine Palas and LoRa Data
palas_data = PalasData()
global final_df = DataFrames.DataFrame()
for data in palas_data
    combined_df = leftjoin(data, df, on = :dateTime)
    global final_df = vcat(final_df, combined_df)
end

#drop missing values
dropmissing!(final_df, :NH3_loRa)
dropmissing!(final_df, :CO2_loRa)

#remove latitude and longitude values from final_df
select!(final_df, Not([:latitude, :longitude]))

#modify calibrate.csv (original test csv data) values to be consistent with final_df 
#=
select!(df_original, Not([:CO_loRa, :temperaturePalas, :humidityPalas, :pressurehPalas]))

for col_name in names(df_original)
    if occursin("grimm", col_name)
        select!(df_original, Not(col_name))
    end
end

df_original.dateTime = map(x -> strip(split(x, '+')[1]), df_original.dateTime)
df_original.dateTime = DateTime.(df_original.dateTime, "dd/mm/yyyy HH:MM:SS")

#create CO2_LoRa Row in df_original 
df_original[!, :CO2_loRa] = fill(0, nrow(df_original))

#Combine the two dataframes
final_df = append!(final_df, df_original, cols = :setequal, promote = false)

=#

#convert every value except dateTime to float64 for consistency
for col in names(final_df)
    if cmp(col, "dateTime") != 0
        final_df[!, col] = convert.(Float64, final_df[!, col])
    end
end

sort!(final_df, :dateTime)
#CSV.write("C:/Users/sethl/OneDrive/Desktop/SethRepo/firmware/old/small_df.csv", final_df)