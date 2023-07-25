using DataFrames, Dates, XLSX, CSV, Statistics

include("PalasData.jl")

xl = XLSX.readxlsx("C:/Users/sethl/OneDrive/Desktop/data/LoRa Ids.xlsx")

#get lora ids and directories of data
id = xl["Sheet1"]
dir1 = readdir("C:/Users/sethl/OneDrive/Desktop/data/Mints LoRa Data/b827eb52fc29")
dir2 = readdir("C:/Users/sethl/OneDrive/Desktop/data/Mints LoRa Data/b827ebf74482")

#create array with precise directories of all lora node csv files using lora id file
global full_dir = []
for lora_id in id["C2:C129"]
    for file_dir1 in dir1
        if occursin(lora_id, file_dir1)
            global full_dir = vcat(full_dir, "C:/Users/sethl/OneDrive/Desktop/data/Mints LoRa Data/b827eb52fc29/$lora_id/" .* readdir("C:/Users/sethl/OneDrive/Desktop/data/Mints LoRa Data/b827eb52fc29/$lora_id/"))
        end
    end
    for file_dir2 in dir2
        if occursin(lora_id, file_dir2)
            global full_dir = vcat(full_dir, "C:/Users/sethl/OneDrive/Desktop/data/Mints LoRa Data/b827ebf74482/$lora_id/" .* readdir("C:/Users/sethl/OneDrive/Desktop/data/Mints LoRa Data/b827ebf74482/$lora_id/"))
        end
    end
end

#Sort CSV files by date in order
global row_names = Vector{Date}()
for data in full_dir
    name = string(split.(data, "_")[2, 1])
    name = string(split.(name, ".")[1, 1])
    name = Dates.Date(name, "yyyy-mm-dd")
    global row_names = push!(row_names, name)
end

sort!(row_names)
filter!(d -> (month(d) in [5, 6, 7]) && year(d) == 2020, row_names)

global df = DataFrames.DataFrame(dateTime = String[], NH3 = Float64[], CO2 = Float64[], NO2 = Float64[], C3H8 = Float64[], C4H10 = Float64[], CH4 = Float64[], H2 = Float64[],
C2H5OH = Float64[], P1_lpo = Float64[], P1_ratio = Float64[], P1_conc = Float64[], P2_lpo = Float64[], P2_ratio = Float64[], P2_conc = Float64[], Temperature = Float64[],
Pressure = Float64[], Humidity = Float64[], Latitude = Float64[], Longitude = Float64[])

#Populates dataframe. Errors indicate that the data is corrupt and will be thrown out. Data is automatically cleaned out.
#excludes csv files with non-functioning BME sensors or corrupt data
for name in union(Dates.format.(row_names, "yyyy-mm-dd"))
    for dir_path in full_dir
        if occursin(name, dir_path)
            try
                temp_df = DataFrames.DataFrame(CSV.File(dir_path))
                if all(temp_df[:, :Temperature] .== 0) || all(temp_df[:, :Humidity] .== 0) || all(temp_df[:, :Pressure] .== 0)
                else
                    append!(df, temp_df, cols = :intersect, promote = false)
                end
            catch e
            end
        end
    end
end


#Rename LoRa columns with _loRa
for col in names(df)
    if col != "dateTime"
        rename!(df, col => col * "_loRa")
    end
end


#turn dateTime from string into dateTime object
rename!(df, :dateTime => :dateString)
df[!, :dateTime] = Vector{DateTime}(undef, nrow(df))
delete_row_num = Vector{Int}()
for index in 1:nrow(df)
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

#average out similar dateTimes
df_grouped = combine(groupby(df, :dateTime), names(df, Not(:dateTime)) .=> mean)

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

#convert every value except dateTime to float64 for consistency
for col in names(final_df)
    if cmp(col, "dateTime") != 0
        final_df[!, col] = convert.(Float64, final_df[!, col])
    end
end

#sort final_df by date
sort!(final_df, :dateTime)

#print unique days
date_df = DataFrames.DataFrame()
date_df[:, :date] = Date.(final_df[:, :dateTime])
unique!(date_df, :date)
println(date_df.date)

unique!(final_df, :dateTime)

CSV.write("C:/Users/sethl/OneDrive/Desktop/data/combined_df.csv", final_df)