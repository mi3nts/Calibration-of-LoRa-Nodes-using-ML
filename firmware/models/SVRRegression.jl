SVMRegressor = @load SVMRegressor pkg=MLJScikitLearnInterface
#gr()

#EpsilonSVR function

function SVRRegression(k, X_train, y_train, X_test, y_test)

    # Training Model
    svr = machine(SVMRegressor(kernel="rbf"), X_train[!, Not("dateTime")], vec(Matrix(y_train)))
    MLJ.fit!(svr, verbosity = 0)
    predict_train = MLJ.predict(svr, X_train[!, Not("dateTime")])
    predict_test = MLJ.predict(svr, X_test[!, Not("dateTime")])
    r2_score_train = r2_score(predict_train, Matrix(y_train))
    r2_score_test = r2_score(predict_test, Matrix(y_test))
    println("SVR: test r2 value for " * k * " = " * string(r2_score_test))

    #LaTex Formatting
    if k[1:2] == "pm" && k != "pmTotal"
        if occursin(k, "pm2_5")
            k = replace(k, "_" => ".")
        end
        k = "pm" * latexstring("_{" * k[3:length(k)] * "}") * " (µg/m" * latexstring("^{3}") * ")"
    end

    # Scatter Plots 
    train_label = "Training Data R² = " * string(r2_score_train)
    test_label = "Testing Data R² = " * string(r2_score_test)
    p = Plots.plot(Matrix(y_train), Matrix(y_train), seriestype=:line, linewidth = 2, color = "blue", label = "1:1",
    xlabel = "Actual " * k, ylabel = "\nEstimated " * k)
    p = Plots.plot!(Matrix(y_train), predict_train, seriestype=:scatter, color = "red", label = train_label)
    p = Plots.plot!(Matrix(y_test), predict_test, seriestype=:scatter, color = "green", label = test_label)
    p = Plots.title!("\nSVR Scatter plot for " * k)
    
    display(p)

end