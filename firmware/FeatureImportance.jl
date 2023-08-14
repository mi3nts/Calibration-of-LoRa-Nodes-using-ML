#Feature Importance Function. Returns a dataframe with feature importance values.
#Use the PlotFeatureImportance Function from PlotFunctions.jl in order to plot the feature importance.

function FeatureImportance(wholedata, k, trainedmodel)
wholedata = DataFrames.select(wholedata, Not("dateTime"))
    #Used to clean the wholedata dataframe
    if findfirst(t -> occursin("Palas", t), names(wholedata)) != nothing
        wholedata = DataFrames.select(wholedata, Not(k * "Palas"))
    elseif findfirst(t -> occursin("_grimm", t), names(wholedata)) != nothing
        wholedata = DataFrames.select(wholedata, Not(k * "_grimm"))
    else
        println("Warning: DataFrame - wholedata - doesn't contain Palas or _grimm data")
    end

    #scale data when dealing with feature importance
    sklearn_preprocessing = pyimport("sklearn.preprocessing")
    scaler = sklearn_preprocessing.StandardScaler()
    wholedata_scaled = scaler.fit_transform(Matrix(wholedata))
    
    col_name = names(wholedata)
    wholedata = DataFrames.DataFrame(pyconvert(Matrix{Float32}, wholedata_scaled), :auto)

    for (n, old_col) in enumerate(names(wholedata))
        rename!(wholedata, Symbol(old_col) => Symbol(col_name[n]))
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