#Feature Importance Function. Returns a dataframe with feature importance values.
#Use the PlotFeatureImportance Function from PlotFunctions.jl in order to plot the feature importance.

function FeatureImportance(wholedata, k, trainedmodel)

    #copies data from the wholedata dataframe, which was taken either from the dictionary Palas[k] or grimm[k]
    explain = wholedata[1:300 , :]
    reference = copy(wholedata)

    #Used to clean the dataframe
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

    #generating data_shap, to be used for feature importance
    data_shap = ShapML.shap(explain = explain, reference = reference, model = trainedmodel,
    predict_function = predict_function, sample_size = sample_size, seed = 1)
    
    #generating data_plot from data_shap + calculating the mean_effect for feature importance
    mean_effect = [:shap_effect] => x -> mean(abs.(x))
    data_plot = DataFrames.combine(groupby(data_shap, :feature_name), mean_effect)
    rename!(data_plot, :shap_effect_function => :mean_effect)
    data_plot = sort!(data_plot, order(:mean_effect, rev = true))
    
    #returning data_plot to be plotted
    return data_plot
end