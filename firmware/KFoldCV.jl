# K fold cross Validation

function KFoldCV(X, y, k_fold, k)
    a = collect(MLBase.Kfold(size(X)[1], k_fold))
    i= 1

    for i in 1:k_fold
        row=a[i]
        temp_X_train = X[row,:]
        temp_y_train = y[row,:]

        temp_X_test = X[setdiff(1:end, row),:]
        temp_y_test = y[setdiff(1:end, row),:]
        
        dtr = machine(DecisionTreeRegressor(), temp_X_train, vec(Matrix(temp_y_train)))
        MLJ.fit!(dtr, verbosity = 0)

        temp_predict_y_train = MLJ.predict(dtr, temp_X_train)
        temp_predict_y_test = MLJ.predict(dtr, temp_X_test)
    
        temp_mse_test = round(mse(temp_predict_y_test, Matrix(temp_y_test)), digits=3)
        #println("Linear Regression: test mse value for " * k * " = " * string(temp_mse_test))
        temp_rmse_test = sqrt(temp_mse_test)
        #println("Linear Regression: test rmse value for " * k * " = " * string(temp_rmse_test))
        r2_score_test = round(r2_score(temp_predict_y_test, Matrix(temp_y_test)), digits=3)
        println("R squared error for $k , fold $i is ",r2_score_test)
    end
end