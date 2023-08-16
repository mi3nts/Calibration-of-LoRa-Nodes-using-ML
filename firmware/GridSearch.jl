#Tuning through grid search

function GridSearch(mlmodel, X, y)
    
    #hyperparameters
    r1 = MLJ.range(Float64, :min_weight_fraction_leaf, lower=0, upper=0.5, scale=:linear)
    #r2 = MLJ.range(Int, :min_samples_leaf, lower=0, upper=5, scale=:linear)  
    grid = Grid(resolution=6)
    tuned_model = TunedModel(model = mlmodel, tuning=grid, resampling = Holdout(fraction_train=0.8), range = [r1], measure=rsquared)
    mach = machine(tuned_model, X, vec(Matrix(y)))
    fit!(mach, verbosity = 0)
    display(fitted_params(mach).best_model)

end