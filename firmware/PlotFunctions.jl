
 #---------------------------------Plotting Function --------------------------------------#


# Plotting Histogram - representing error between actual vs predicted values
function PlotHistogram(y_test, predict_test, k, type)
    if lowercase(type) == "train" 
        color = "green"
    elseif lowercase(type) == "test"
        color = "red"
    else
        println("Improper type was entered. Specify either train or test.")
        return nothing
    end

    error_test = Matrix(y_test) - predict_test
    bin_range = range(-10, 10, length=60)
    display(histogram(error_test, label="Data", bins=bin_range, color=color, title= "\nEstimation Error for " * k * " Values",
    xlabel="(Actual - Predicted) " * k * " Value", ylabel="Frequency"))

end

# Plotting Bar Graph Comparison of Actual vs Predicted Values
function PlotBarComparison(y_test, predict_test, k)

    compare_data = hcat(Matrix(y_test)[1:9],predict_test[1:9])
    group_num = repeat(["Actual", "Predicted"], inner = 9)
    nam = repeat("G" .* string.(1:9), outer = 2)
    display(groupedbar(nam, compare_data, group = group_num, ylabel = "\n" * k * " values", xlabel = "Groups",
    title = "\nActual vs Predicted " * k * " Values"))

end

# Plotting Scatter Plots 
function PlotScatter(y_train, y_test, predict_train, predict_test, k, kcopy)

    r2_score_train = round(r2_score(predict_train, Matrix(y_train)), digits=3)
    r2_score_test = round(r2_score(predict_test, Matrix(y_test)), digits=3)
    train_label = "Training Data R² = " * string(r2_score_train)
    test_label = "Testing Data R² = " * string(r2_score_test)
    p = Plots.plot(Matrix(y_train), Matrix(y_train), seriestype=:line, linewidth = 2, color = "blue", label = "1:1",
    xlabel = "Actual " * k, ylabel = "\nEstimated " * k)
    p = Plots.plot!(Matrix(y_train), predict_train, seriestype=:scatter, color = "red", label = train_label)
    p = Plots.plot!(Matrix(y_test), predict_test, seriestype=:scatter, color = "green", label = test_label)
    p = Plots.title!("\nScatter Plot for " * k)
    
    display(p)
    savefig(p, "C:/Users/sethl/OneDrive/Desktop/SethRepo/firmware/plotimages/scatterplot" * kcopy)

end

# Plotting QQ Plots vs No3rmal Distribution
function PlotQQ(y_test, predict_test, k, kcopy)

    p = Plots.plot(qqplot(vec(Matrix(y_test)), predict_test), title = "\nQuantile-Quantile Plot for " * k, 
    xlabel = "Actual Quantile", ylabel = "Estimated Quantile")
    #p = Plots.plot!(quantile(vec(Matrix(y_test)), [0, 0.25, 0.5, 0.75, 1]), quantile(predict_test, [0, 0.25, 0.5, 0.75, 1]), seriestype=:scatter, color = "red")
    display(p)
    savefig(p, "C:/Users/sethl/OneDrive/Desktop/SethRepo/firmware/plotimages/QQ-Plot" * kcopy)

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
    data_plot.feature_name = replace.(data_plot.feature_name, "P1" => "PM" * latexstring("_{2.5}") * "~" * "PM" * latexstring("_{10}"))
    data_plot.feature_name = replace.(data_plot.feature_name, "P2" => "PM" * latexstring("_{10}"))

    p = Plots.bar(data_plot[:, :relative_importance], title="\nFeature Importance for " * k,
    yticks=(1:16,data_plot[:, :feature_name]), bottom_margin=0mm, 
    xlabel="Relative Importance", legend = false, orientation=:h, xlims=(-0.01, 1.01))
    display(p)
    savefig(p, "C:/Users/sethl/OneDrive/Desktop/SethRepo/firmware/plotimages/featureimportance" * kcopy)
    
end