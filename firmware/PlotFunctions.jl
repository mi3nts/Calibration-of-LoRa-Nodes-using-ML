
 #---------------------------------Plotting Function --------------------------------------#


# Plotting Histogram - representing error between actual vs predicted values
function PlotHistogram(y_test, predict_test, k, kcopy)

    error_test = Matrix(y_test) - predict_test
    bin_range = range(-10, 10, length=60)
    p = histogram(error_test, label="Data", bins=bin_range, color="red", title= "\nEstimation Error for " * k * " Values",
    xlabel="Error of " * k * " Value", ylabel="Frequency")
    display(p)
    savefig(p, "C:/Users/sethl/OneDrive/Desktop/plotimages/histogram" * kcopy)

end

# Plotting Bar Graph Comparison of Actual vs Predicted Values
function PlotBarComparison(y_test, predict_test, k, kcopy)

    compare_data = hcat(Matrix(y_test)[1:9],predict_test[1:9])
    group_num = repeat(["Actual", "Predicted"], inner = 9)
    nam = repeat("G" .* string.(1:9), outer = 2)
    p = groupedbar(nam, compare_data, group = group_num, ylabel = "\n" * k * " values", xlabel = "Groups",
    title = "\nActual vs Predicted " * k * " Values")
    display(p)
    savefig(p, "C:/Users/sethl/OneDrive/Desktop/plotimages/barcomparison" * kcopy)

end

# Plotting Scatter Plots 
function PlotScatter(y_train, y_test, predict_train, predict_test, k, kcopy)

    r2_score_train = round(r2_score(predict_train, Matrix(y_train)), digits=3)
    r2_score_test = round(r2_score(predict_test, Matrix(y_test)), digits=3)
    #train_label = "Training Data R² = " * string(r2_score_train)
    #test_label = "Testing Data R² = " * string(r2_score_test)
    p = scatterresult(vec(Matrix(y_train)), predict_train,
                      vec(Matrix(y_test)), predict_test;
                      xlabel="Actual " * k,
                      ylabel="\nPredicted " * k,
                      plot_title= "Scatter Plot Fit")
                      
    #old scatter plot code
    #p = Plots.plot(Matrix(y_train), Matrix(y_train), seriestype=:line, linewidth = 2, color = "blue", label = "1:1",
    #xlabel = "Actual " * k, ylabel = "\nEstimated " * k)
    #p = Plots.plot!(Matrix(y_train), predict_train, seriestype=:scatter, color = "red", label = train_label)
    #p = Plots.plot!(Matrix(y_test), predict_test, seriestype=:scatter, color = "green", label = test_label)
    #p = Plots.title!("\nScatter Plot for " * k)
    
    display(p)
    savefig(p, "C:/Users/sethl/OneDrive/Desktop/plotimages/scatterplot" * kcopy)

end

# Plotting QQ Plots of actual data quantiles vs estimated data quantiles
function PlotQQ(y_test, predict_test, k, kcopy)

    p = Plots.plot(qqplot(vec(Matrix(y_test)), predict_test), title = "\nQuantile-Quantile Plot for " * k, 
    xlabel = "Actual Quantile", ylabel = "Estimated Quantile")
    y_test_quantile = quantile(vec(Matrix(y_test)), [0,0.25,0.5,0.75,1])
    y_predict_quantile = quantile(predict_test, [0,0.25,0.5,0.75,1])
    p = Plots.plot!(y_test_quantile, y_predict_quantile, seriestype=:scatter, color = "red", marker = :xcross)

    #plot quantile markers at 0, 0.25, 0.5, 0.75, and 1
    for i in 1:5
        p = Plots.annotate!(y_test_quantile[i] + p[1][1][:x][2]*0.022, y_predict_quantile[i] - p[1][1][:x][2]*0.01,
        text(round(Int, 100*(i-1)/4), :red, :center, 9))
    end

    display(p)
    savefig(p, "C:/Users/sethl/OneDrive/Desktop/plotimages/QQ-Plot" * kcopy)
end

# Plotting Feature importance
function PlotFeatureImportance(data_plot, k, kcopy)
    #Format axis
    data_plot.feature_name = replace.(data_plot.feature_name, "_" => " ")
    data_plot.feature_name = replace.(data_plot.feature_name, "lpo" => "LPO")
    data_plot.feature_name = replace.(data_plot.feature_name, "loRa" => "")
    data_plot.feature_name = replace.(data_plot.feature_name, "conc" => "concentration")
    data_plot.feature_name = replace.(data_plot.feature_name, "2" => latexstring("_2"))
    data_plot.feature_name = replace.(data_plot.feature_name, "3" => latexstring("_3"))
    data_plot.feature_name = replace.(data_plot.feature_name, "4" => latexstring("_4"))
    data_plot.feature_name = replace.(data_plot.feature_name, "5" => latexstring("_5"))
    data_plot.feature_name = replace.(data_plot.feature_name, "8" => latexstring("_8"))
    data_plot.feature_name = replace.(data_plot.feature_name, "10" => latexstring("_{10}"))
    data_plot.feature_name = replace.(data_plot.feature_name, "P1" => ">1µm" * latexstring("_{}"))
    data_plot.feature_name = replace.(data_plot.feature_name, "P" * latexstring("_2") => ">2.5µm" * latexstring("_{}"))
    data_plot.feature_name = replace.(data_plot.feature_name, "Temperature" => "Temperature" * latexstring("_{}"))
    data_plot.feature_name = replace.(data_plot.feature_name, "Pressure" => "Pressure" * latexstring("_{}"))
    data_plot.feature_name = replace.(data_plot.feature_name, "Humidity" => "Humidity" * latexstring("_{}"))

    p = Plots.bar(data_plot[:, :relative_importance], title="\nFeature Importance for " * k,
    yticks=(1:nrows(data_plot), data_plot[:, :feature_name]), bottom_margin=0mm, 
    xlabel="Relative Importance", legend = false, orientation=:h, xlims=(-0.01, 1.01))
    display(p)
    savefig(p, "C:/Users/sethl/OneDrive/Desktop/plotimages/featureimportance" * kcopy)
    
end

function PlotTimeSeries(wholedata, model, k)
    DateTime = vec(Matrix(select(wholedata, :dateTime)))

    if findfirst(t -> occursin("Palas", t), names(wholedata)) != nothing
        actualvalue = wholedata[:, k * "Palas"]
        wholedata = select(wholedata, Not(k * "Palas"))
        wholedata = select(wholedata, Not("dateTime"))
    elseif findfirst(t -> occursin("_grimm", t), names(wholedata)) != nothing
        actualvalue = wholedata[:, k * "_grimm"]
        wholedata = select(wholedata, Not(k * "_grimm"))
        wholedata = select(wholedata, Not("dateTime"))
    else
        println("Warning: DataFrame - wholedata - doesn't contain Palas or _grimm data")
    end

    #estimatedvalue = MLJ.predict(model, wholedata)

    
end