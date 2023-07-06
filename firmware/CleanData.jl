using DataFrames, CSV, Dates
df = DataFrames.DataFrame(CSV.File("C:/Users/sethl/OneDrive/Desktop/data/palasAllGPSJuly2020_palasCarGPS.csv"))

# remove empty columns from dataframe


for name in names(df)
    if occursin("Column", name)
        global df = select!(df, Not(name))
    #= convert columns from strings to floats (and delete rows if it fails)
    else
        if !occursin("datetime", lowercase(lowername))
            
        end
    =#
    end
end


# initial cleaning NaN or missing values

for (counter, n) in enumerate(names(df))
    try
        #filter!(counter => x -> !(ismissing(x) || isnothing(x) || isnan(x)), df)

        #cleaning out infinite data
        #filter!(counter => x -> !(x == Inf), df)

        
    catch e
    end
end


# conversion to datetime
# rename dateTime to dateString, then iterate over each dateString to convert it to dateTime object
# using Dates.DateTime inside a try catch statement. if it fails to convert - delete the row

rename!(df, :dateTime => :dateString)
df[!, :dateTime] = Vector{DateTime}(undef, nrow(df))
delete_row_num = Vector{Int}()
for (index, row) in enumerate(eachrow(df))
    try
        df[index, :dateTime] = Dates.DateTime(df[index, :dateString], "dd-u-yyyy HH:MM:SS")
    catch e
        push!(delete_row_num, index)
    end
end

#delete rows that had corrupted dateTimes and delete dateString value
delete!(df, delete_row_num)
df = select!(df, Not(:dateString))


#sort only 2020 data
delete_row_num2 = Vector{Int}()
for (index,i) in enumerate(df.dateTime)
    if year(i) == 2019
        push!(delete_row_num2, index)
    end
end
delete!(df, delete_row_num2)
    


#Check if GPS coordinates align with waterview building/location

df.latitude .= round.(df.latitude, digits=3)

for (counter, name) in enumerate(names(df))
    if name == "latitude"
        filter!(counter => x -> (x >= 32.990 && x <= 32.995), df)
    elseif name == "longitude"
        filter!(counter => x -> (x >= -96.760 && x <= -96.755), df)
    end
end

#groupby month and print out how which days of data are collected in each month
df[!, :dateMonth] = Dates.month.(df.dateTime)
for i in groupby(df, :dateMonth)
    i = i[!, Not(:dateMonth)]
  
    #find dateTime when dCn = infinity
    #=for (index, row) in enumerate(eachrow(i))
        if i[index, :dCnPalas] == Inf
            println(i[index, :dateTime])
        end
    end
    =#  
    #println(Dates.monthname(i[1, :dateTime]) * ": " .* string.(unique(Dates.day.(i.dateTime))))
    println(size(i))
    println(describe(i))
end


