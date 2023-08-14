#Tuning through grid search

function GridSearch(mlmodel, X, y)
    
    #hyperparameters
    layer_options = [
    (128,),
    (128, 256),
    (128, 256, 128),
    (128,256,256,128),
    (128,256,128,128),

    ]


    r1 = MLJ.range(mlmodel, :builder, values=[
    MLJFlux.MLP(hidden=arch, σ=Flux.relu) for arch in layer_options])
    
    #r2 = MLJ.range(mlmodel, :max_depth, lower=20, upper=40, scale=:linear)  
    grid = Grid(resolution=10)
    tuned_model = TunedModel(model = mlmodel, tuning=grid, resampling = Holdout(fraction_train=0.8), range = [r1], measure=rsquared)
    mach = machine(tuned_model, X, vec(Matrix(y)))
    fit!(mach, verbosity = 0)
    display(fitted_params(mach).best_model)

end