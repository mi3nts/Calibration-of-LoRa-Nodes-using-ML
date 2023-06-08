using Dates, DataFrames, CSV, Plots, MLJ
using ScikitLearn.CrossValidation: train_test_split

filepath = "C:/Users/sethl/OneDrive/Desktop/Calibration-of-LoRa-Nodes-using-Machine-Learning-main/calibrate.csv"
df = DataFrames.DataFrame(CSV.File(filepath))

#variables
y_grimm = []
y_Palas = []
x = []

#cleaning data
df.dateTime = map(x -> strip(split(x, '+')[1]), df.dateTime)
df.dateTime = DateTime.(df.dateTime, "dd/mm/yyyy HH:MM:SS")
for col in names(df)
    if eltype(df[!, col]) == Int64
        df[!, col] = float(df[!, col])
    end
end

#fill arrays with column names from the dataframe
col_name = names(df)
push!(x, "dateTime")
for i in col_name
    if occursin("_grimm", i)
        push!(y_grimm, i)
    end
    if occursin("_loRa", i)
        push!(x, i)
    end
    if occursin("Palas", i)
        push!(y_Palas, i)
    end
end

#Dictionary

grimm = Dict{String, DataFrame}()
for i in y_grimm
    grimm_cols = push!(x, i)
    grimm[replace(i, "_grimm" => "")]  = df[!, grimm_cols]
    pop!(x)
end


Palas = Dict{String, DataFrame}()
for i in y_Palas
    Palas_cols = push!(x, i)
    Palas[replace(i, "_Palas" => "")] = df[!, Palas_cols]
    pop!(x)
end

# Partition data for training and testing
# In this for loop, k = key and v = value

for (k,v) in grimm

    X = select(grimm[k], Not(k * "_grimm"))
    y = select(grimm[k], k * "_grimm")
    (X_train, X_test), (y_train, y_test) = partition((X,y), 0.8, multi=true)
    println("x train " * string(size(X_train)))
    println("x test " * string(size(X_test)))
    println("y train " * string(size(y_train)))
    println("y test" * string(size(y_test)))

end     

#----------------------------------Supervised Learning-----------------------------------------------# 


# microgram/meter cubed for pm
