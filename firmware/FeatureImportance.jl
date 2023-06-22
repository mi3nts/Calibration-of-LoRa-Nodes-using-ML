#Feature Importance Function. Returns a dataframe with feature importance values.
#Use the PlotFeatureImportance Function from PlotFunctions.jl in order to plot the feature importance.

function FeatureImportance(wholedata, k, trainedmodel)

    #Used to clean the wholedata dataframe
    if findfirst(t -> occursin("Palas", t), names(wholedata)) != nothing
        wholedata = select(wholedata, Not(k * "Palas"))
    elseif findfirst(t -> occursin("_grimm", t), names(wholedata)) != nothing
        wholedata = select(wholedata, Not(k * "_grimm"))
    else
        println("Warning: DataFrame - wholedata - doesn't contain Palas or _grimm data")
    end

    sample_size = 100

    function predict_function(model, data)
        return DataFrame(y_pred = MLJ.predict(model, data))
    end

    #generating data_shap, to be used for feature importance
    data_shap = ShapML.shap(explain =  wholedata[1:300 , :], reference = wholedata, model = trainedmodel,
    predict_function = predict_function, sample_size = sample_size, seed = 1)
    
    #generating data_plot from data_shap + calculating the mean_effect for feature importance
    data_plot = DataFrames.combine(groupby(data_shap, :feature_name), :shap_effect => (x -> mean(abs.(x))) => :mean_effect)

    #create relative importance
    data_plot.relative_importance = (data_plot.mean_effect)./maximum(data_plot.mean_effect) 
    data_plot = sort!(data_plot, order(:relative_importance))

    #returning data_plot to be plotted
    return data_plot
end