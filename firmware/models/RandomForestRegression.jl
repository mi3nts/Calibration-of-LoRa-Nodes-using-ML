RandomForestRegressor = @load RandomForestRegressor pkg=MLJScikitLearnInterface verbosity = 0

# Random Forest Regression Function

function RandomForestRegression(k, X_train, y_train, X_test, y_test, wholedata)

    # Training model
    rfr = machine(RandomForestRegressor(), X_train[!, Not("dateTime")], vec(Matrix(y_train)))
    MLJ.fit!(rfr, verbosity = 0)
    predict_train = MLJ.predict(rfr, X_train[!, Not("dateTime")])
    predict_test = MLJ.predict(rfr, X_test[!, Not("dateTime")])
    #r2_score_test = round(r2_score(predict_test, Matrix(y_test)), digits=3)
    #println("Random Forest Regression: test r2 value for " * k * " = " * string(r2_score_test))

    # Calculating Feature Importance
    explain = copy(wholedata[1:300 , :])
    reference = copy(wholedata)

    if findfirst(t -> occursin("Palas", t), names(explain)) != nothing
        explain = select(explain, Not(k * "Palas"))
        reference = select(reference, Not(k * "Palas"))
    elseif findfirst(t -> occursin("_grimm", t), names(explain)) != nothing
        explain = select(explain, Not(k * "_grimm"))
        reference = select(reference, Not(k * "_grimm"))
    else
        println("Error: DataFrame - wholedata - doesn't contain Palas or _grimm data")
        return nothing
    end

    sample_size = 60

    function predict_function(model, data)
        data_pred = DataFrame(y_pred = MLJ.predict(model, data))
        return data_pred
    end

    data_shap = ShapML.shap(explain = explain, reference = reference, model = rfr,
    predict_function = predict_function, sample_size = sample_size, seed = 1)
    
    mean_effect = [:shap_effect] => x -> mean(abs.(x))
    data_plot = DataFrames.combine(groupby(data_shap, :feature_name), mean_effect)
    rename!(data_plot, :shap_effect_function => :mean_effect)
    data_plot = sort!(data_plot, order(:mean_effect, rev = true))


    #LaTex Formatting
    if k[1:2] == "pm" && k != "pmTotal"
        if occursin(k, "pm2_5")
            k = replace(k, "_" => ".")
        end
        k = "pm" * latexstring("_{" * k[3:length(k)] * "}") * " (µg/m" * latexstring("^{3}") * ")"
    end

    #Plotting Functions, "test" will plot the test data, whereas "train" will plot the train data. Only PlotScatter does not use "train" or "test"
    PlotHistogram(y_test, predict_test, k, "test")
    PlotBarComparison(y_test, predict_test, k, "test")
    PlotScatter(y_train, y_test, predict_train, predict_test, k)
    PlotQQ(predict_test, k, "test")
    PlotFeatureImportance(data_plot, k)

    

end
