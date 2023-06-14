LinearRegressor = @load LinearRegressor pkg=MLJScikitLearnInterface verbosity = 0

# Linear Regression Function

function LinearRegression(k, X_train, y_train, X_test, y_test, wholedata)

    # Training model
    lm = machine(LinearRegressor(), X_train, vec(Matrix(y_train)))
    MLJ.fit!(lm, verbosity = 0)
    predict_train = MLJ.predict(lm, X_train)
    predict_test = MLJ.predict(lm, X_test)
    #r2_score_test = round(r2_score(predict_test, Matrix(y_test)), digits=3)
    #println("Linear Regression: test r2 value for " * k * " = " * string(r2_score_test))


    # Calculating Feature Importance using the FeatureImportance Function from FeatureImportance.jl
    data_plot = FeatureImportance(wholedata, k, lm)


    #LaTex Formatting
    if k[1:2] == "pm" && k != "pmTotal"
        if occursin(k, "pm2_5")
            k = replace(k, "_" => ".")
        end
        k = "pm" * latexstring("_{" * k[3:length(k)] * "}") * " (µg/m" * latexstring("^{3}") * ")"
    end

    #Plotting Functions, "test" will plot the test data, whereas "train" will plot the train data.
    #Only PlotScatter does not use "train" or "test"
    PlotHistogram(y_test, predict_test, k, "test")
    PlotBarComparison(y_test, predict_test, k, "test")
    PlotScatter(y_train, y_test, predict_train, predict_test, k)
    PlotQQ(predict_test, k, "test")
    PlotFeatureImportance(data_plot, k)


end


