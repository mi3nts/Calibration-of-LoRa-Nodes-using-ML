using Pkg
Pkg.activate("D:/UTD/UTDSummer2023/Calibration-of-LoRa-Nodes-using-ML/")
using Dates, DataFrames, CSV, MLJ, Metrics, LaTeXStrings, StatsPlots, Measures, Distributions, ShapML,ScikitLearn
gr()

#Load in dataframe
filepath = "D:/UTD/UTDSummer2023/Calibration-of-LoRa-Nodes-using-ML/data/calibrate.csv"
df = DataFrames.DataFrame(CSV.File(filepath))

#include plotting functions from PlotFunctions.jl and Feature Importance function from FeatureImportance.jl
include("PlotFunctions.jl")
include("FeatureImportance.jl")

#include functions from models file
include("models/LinearRegression.jl")
include("models/NeuralNetworkRegression.jl")
include("models/SVRRegression.jl")
include("models/GaussianProcessRegression.jl")
include("models/DecisionTreeRegression.jl")
include("models/RandomForestRegression.jl")


#variables
y_grimm = []
y_Palas = []
x = []


#Cleaning data. Converting all numeric values in dataframe into Float64.
df.dateTime = map(x -> strip(split(x, '+')[1]), df.dateTime)
df.dateTime = DateTime.(df.dateTime, "dd/mm/yyyy HH:MM:SS")
for col in names(df)
    if eltype(df[:, col]) == Int64  
        df[!, col] = float(df[!, col])
    end
end

#Converts all data in the dataframe to Float32 if the model (like Neural Network Regression) requires it
#=
for col in names(df)
    if eltype(df[:, col]) == Float64 || eltype(df[:, col]) == Int64
        df[!, col] = convert.(Float32, df[!, col])
    end
end
=#


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

#Creating Dictionaries, where key is the value being measured, which is assigned to a dataframe with data from the LoRa Nodes

grimm = Dict{String, DataFrame}()
for i in y_grimm
    grimm_cols = push!(x, i)
    grimm[replace(i, "_grimm" => "")]  = df[:, grimm_cols]
    pop!(x)
end


Palas = Dict{String, DataFrame}()
for i in y_Palas
    Palas_cols = push!(x, i)
    Palas[replace(i, "Palas" => "")] = df[:, Palas_cols]
    pop!(x)
end

# Remove non-PM variables from Palas
Palas_delete = ["pressureh", "temperature", "humidity"]
for key in Palas_delete
    global Palas = delete!(Palas, key)
end



#----------------------------------Supervised Learning-----------------------------------------------# 
# ---------------------------Grimm Data-----------------------------------# 

# Partition data for training and testing
# In this for loop, k = key and v = value. This loops over every grimm value type (like pm1) to be trained on

println("--------------Grimm Data---------------")
for (k,v) in grimm

    #setting up data to be trained and tested on
    X = DataFrames.select(grimm[k], Not(k * "_grimm"))
    X = X[!, Not("dateTime")]
    y = DataFrames.select(grimm[k], k * "_grimm")
    (X_train, X_test), (y_train, y_test) = partition((X,y), rng=124, 0.8, multi=true)

    #wholedata is used for feature importance
    wholedata = grimm[k]
    wholedata = wholedata[!, Not("dateTime")]

    #--------------------------Regression Functions--------------------------#

    # Run linear regression function from LinearRegression.jl
    #LinearRegression(k, X_train, y_train, X_test, y_test, wholedata)

    # Run neural network regression function from NeuralNetworkRegression.jl
    #NeuralNetworkRegression(k, X_train, y_train, X_test, y_test, wholedata)

    # Run SVR function from SVR.jl
    #SVRRegression(k, X_train, y_train, X_test, y_test, wholedata)

    # Run Gaussian Process regression function from GaussianProcessRegressor.jl
    #GaussianProcessRegression(k, X_train, y_train, X_test, y_test, wholedata)

    # Run Decision Tree function from DecisionTreeRegression.jl
    #DecisionTreeRegression(k, X_train, y_train, X_test, y_test, wholedata)

    # Run Random Forest Tree function from RandomForestRegression.jl
    #RandomForestRegression(k, X_train, y_train, X_test, y_test, wholedata)

end

# ---------------------------Palas Data-----------------------------------# 

# Partition data for training and testing
# In this for loop, k = key and v = value. This loops over every Palas type (like pm4) to be trained on

println("--------------Palas Data---------------")

for (k,v) in Palas
    
    #setting up data to be trained and tested on
    X = DataFrames.select(Palas[k], Not(k * "Palas"))
    X = X[!, Not("dateTime")]
    y = DataFrames.select(Palas[k], k * "Palas")
    (X_train, X_test), (y_train, y_test) = partition((X,y), rng=123, 0.80, multi=true)

    #wholedata is used for feature importance
    wholedata = Palas[k]
    wholedata = wholedata[!, Not("dateTime")]


    #--------------------------Regression Functions--------------------------#

    # Run linear regression function from LinearRegression.jl
    LinearRegression(k, X_train, y_train, X_test, y_test, wholedata)

    # Run neural network regression function from NeuralNetworkRegression.jl
    #NeuralNetworkRegression(k, X_train, y_train, X_test, y_test, wholedata)

    # Run SVR function from SVR.jl
    #SVRRegression(k, X_train, y_train, X_test, y_test, wholedata)

    # Run Gaussian Process regression function from GaussianProcessRegressor.jl
    #GaussianProcessRegression(k, X_train, y_train, X_test, y_test, wholedata)

    # Run Decision Tree function from DecisionTreeRegression.jl
    #DecisionTreeRegression(k, X_train, y_train, X_test, y_test, wholedata)

    # Run Random Forest Tree function from RandomForestRegression.jl
    #RandomForestRegression(k, X_train, y_train, X_test, y_test, wholedata)


end