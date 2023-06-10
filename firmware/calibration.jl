using Dates, DataFrames, CSV, MLJ, MLJScikitLearnInterface, Metrics, StatsPlots, Plots
LinearRegressor = @load LinearRegressor pkg=MLJScikitLearnInterface
gr()
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

# Dictionary

grimm = Dict{String, DataFrame}()
for i in y_grimm
    grimm_cols = push!(x, i)
    grimm[replace(i, "_grimm" => "")]  = df[!, grimm_cols]
    pop!(x)
end


Palas = Dict{String, DataFrame}()
for i in y_Palas
    Palas_cols = push!(x, i)
    Palas[replace(i, "Palas" => "")] = df[!, Palas_cols]
    pop!(x)
end

# Remove non-PM variables from Palas
Palas_delete = ["pressureh", "humidity"]
Palas = delete!(Palas, Palas_delete)


#----------------------------------Supervised Learning-----------------------------------------------# 
# ---------------------------Grimm Data-----------------------------------# 

# Partition data for training and testing
# In this for loop, k = key and v = value

println("--------------Grimm Data---------------")
for (k,v) in grimm

    X = DataFrames.select(grimm[k], Not(k * "_grimm"))
    y = DataFrames.select(grimm[k], k * "_grimm")
    (X_train, X_test), (y_train, y_test) = partition((X,y), rng=123, 0.8, multi=true)

    #=
    println("x train " * string(size(X_train)))
    println("x test " * string(size(X_test)))
    println("y train " * string(size(y_train)))
    println("y test" * string(size(y_test)))
    =#
   

    # Linear Regression
    model = LinearRegressor()
    lm = machine(model, X_train[!, Not("dateTime")], vec(Matrix(y_train)))
    MLJ.fit!(lm, verbosity = 0)
    predict_train = MLJ.predict(lm, X_train[!, Not("dateTime")])
    predict_test = MLJ.predict(lm, X_test[!, Not("dateTime")])
    r2_score_train = r2_score(predict_train, Matrix(y_train))
    r2_score_test = r2_score(predict_test, Matrix(y_test))
    println("test r2 value for " * k * " = " * string(r2_score_test))

    # Histogram
    error_train = Matrix(y_train) - predict_train
    error_test = Matrix(y_test) - predict_test
    bin_range = range(-10, 10, length=100)
    #display(histogram(error_train, label="train", bins=bin_range, color="green"))
    #display(histogram(error_test, label="test", bins=bin_range, color="red"))

    # Bar Graph comparison
    compare_data = hcat(Matrix(y_test)[1:20],predict_test[1:20])
    group_num = repeat(["Actual", "Predicted"], inner = 20)
    nam = repeat(1:20, outer = 2)
    #display(groupedbar(nam, compare_data, group = group_num, ylabel = k, xlabel = "Test Groups",
    #title = "Comparison between actual vs predicted test values"))

    # Scatter Plots, Error histograms MSE, RMSE, Time Series
    train_label = "Training Data R² = " * string(r2_score_train)
    test_label = "Testing Data R² = " * string(r2_score_test)
    p = plot(Matrix(y_train), Matrix(y_train), seriestype=:line, linewidth = 2, color = "blue", label = "1:1", title = "Scatter plot for " * k, xlabel = "Actual " * k, ylabel = "Estimated " * k)
    p = plot!(Matrix(y_train), predict_train, seriestype=:scatter, color = "red", label = train_label)
    p = plot!(Matrix(y_test), predict_test, seriestype=:scatter, color = "green", label = test_label)
    display(p)

end

# ---------------------------Palas Data-----------------------------------# 

# Partition data for training and testing
# In this for loop, k = key and v = value

println("--------------Palas Data---------------")
for (k,v) in Palas
    
    X = DataFrames.select(Palas[k], Not(k * "Palas"))
    y = DataFrames.select(Palas[k], k * "Palas")
    (X_train, X_test), (y_train, y_test) = partition((X,y), rng=123, 0.8, multi=true)
    
    #=
    println("x train " * string(size(X_train)))
    println("x test " * string(size(X_test)))
    println("y train " * string(size(y_train)))
    println("y test" * string(size(y_test)))
    =#
   
    # Linear Regression
    model = LinearRegressor()
    lm = machine(model, X_train[!, Not("dateTime")], vec(Matrix(y_train)))
    MLJ.fit!(lm, verbosity = 0)
    predict_train = MLJ.predict(lm, X_train[!, Not("dateTime")])
    predict_test = MLJ.predict(lm, X_test[!, Not("dateTime")])
    r2_score_train = r2_score(predict_train, Matrix(y_train))
    r2_score_test = r2_score(predict_test, Matrix(y_test))
    println("test r2 value for " * k * " = " * string(r2_score_test))


end