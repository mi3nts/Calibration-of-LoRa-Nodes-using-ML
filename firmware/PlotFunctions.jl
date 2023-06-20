
 #---------------------------------Plotting Function --------------------------------------#


# Plotting Histogram - representing error between actual vs predicted values
function PlotHistogram(y_test, predict_test, k, type)
    plot_text = ""
    if lowercase(type) == "train" 
        plot_text = "Train"
        color = "green"
    elseif lowercase(type) == "test"
        plot_text = "Test"
        color = "red"
    else
        println("Improper type was entered. Specify either train or test.")
        return nothing
    end

    error_test = Matrix(y_test) - predict_test
    bin_range = range(-10, 10, length=60)
    display(histogram(error_test, label=plot_text * " Data", bins=bin_range, color=color, title= "\n" * plot_text * " Data: Estimation Error for " * k * " Values",
    xlabel="(Actual - Predicted) " * k * " Value", ylabel="Frequency"))

end

# Plotting Bar Graph Comparison of Actual vs Predicted Values
function PlotBarComparison(y_test, predict_test, k, type)
    
    plot_text = ""
    if lowercase(type) == "train" 
        plot_text = "Train"
    elseif lowercase(type) == "test"
        plot_text = "Test"
    else
        println("Improper type was entered. Specify either train or test.")
        return nothing
    end

    compare_data = hcat(Matrix(y_test)[1:9],predict_test[1:9])
    group_num = repeat(["Actual", "Predicted"], inner = 9)
    nam = repeat("G" .* string.(1:9), outer = 2)
    display(groupedbar(nam, compare_data, group = group_num, ylabel = "\n" * k * " values", xlabel = "Groups",
    title = "\n" * plot_text * " Data: Actual vs Predicted " * k * " Values"))

end

# Plotting Scatter Plots 
function PlotScatter(y_train, y_test, predict_train, predict_test, k)

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
end

# Plotting QQ Plots vs Normal Distribution
function PlotQQ(predict_test, k, type)
    
    plot_text = ""
    if lowercase(type) == "train" 
        plot_text = "Train"
    elseif lowercase(type) == "test"
        plot_text = "Test"
    else
        println("Improper type was entered. Specify either train or test.")
        return nothing
    end

    p = Plots.plot(qqplot(Normal, predict_test), title = "\n" * plot_text * " Data: Q-Q Plot of Estimated " * k * " Values", 
    xlabel = "Normal Theoretical Quantiles", ylabel = "Sample Quantiles")
    display(p)

end

# Plotting Feature importance
function PlotFeatureImportance(data_plot, k)

    p = Plots.bar(data_plot[:, :relative_importance], title="\nFeature Importance for " * k,
    xticks=(1:17,data_plot[:, :feature_name]), bottom_margin=10mm, xrotation=90, 
    ylabel="Relative Importance", legend = false)
    display(p)
    
end