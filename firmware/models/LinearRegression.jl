LinearRegressor = @load LinearRegressor pkg=MLJScikitLearnInterface
#gr()

# Linear Regression Function

function LinearRegression(k, X_train, y_train, X_test, y_test)

    # Training model
    lm = machine(LinearRegressor(), X_train[!, Not("dateTime")], vec(Matrix(y_train)))
    MLJ.fit!(lm, verbosity = 0)
    predict_train = MLJ.predict(lm, X_train[!, Not("dateTime")])
    predict_test = MLJ.predict(lm, X_test[!, Not("dateTime")])
    r2_score_train = r2_score(predict_train, Matrix(y_train))
    r2_score_test = r2_score(predict_test, Matrix(y_test))
    println("Linear Regression: test r2 value for " * k * " = " * string(r2_score_test))

    #LaTex Formatting
    if k[1:2] == "pm" && k != "pmTotal"
        if occursin(k, "pm2_5")
            k = replace(k, "_" => ".")
        end
        k = "pm" * latexstring("_{" * k[3:length(k)] * "}") * " (µg/m" * latexstring("^{3}") * ")"
    end

    # Histogram - plotting error
    error_train = Matrix(y_train) - predict_train
    error_test = Matrix(y_test) - predict_test
    bin_range = range(-10, 10, length=100)
    #display(histogram(error_train, label="train", bins=bin_range, color="green"))
    #display(histogram(error_test, label="test", bins=bin_range, color="red"))

    # Bar Graph comparison
    compare_data = hcat(Matrix(y_test)[1:9],predict_test[1:9])
    group_num = repeat(["Actual", "Predicted"], inner = 9)
    nam = repeat("G" .* string.(1:9), outer = 2)
    #display(groupedbar(nam, compare_data, group = group_num, ylabel = "\n" * k * " values", xlabel = "Test Groups",
    #title = "Comparison between actual vs predicted values"))

    
    # Scatter Plots 
    train_label = "Training Data R² = " * string(r2_score_train)
    test_label = "Testing Data R² = " * string(r2_score_test)
    p = Plots.plot(Matrix(y_train), Matrix(y_train), seriestype=:line, linewidth = 2, color = "blue", label = "1:1",
    xlabel = "Actual " * k, ylabel = "\nEstimated " * k)
    p = Plots.plot!(Matrix(y_train), predict_train, seriestype=:scatter, color = "red", label = train_label)
    p = Plots.plot!(Matrix(y_test), predict_test, seriestype=:scatter, color = "green", label = test_label)
    p = Plots.title!("\nLinear Regression Scatter Plot for " * k)
    
    display(p)

end


