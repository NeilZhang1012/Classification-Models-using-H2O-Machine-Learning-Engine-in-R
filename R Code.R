library(dplyr)
library(h2o)
h2o.init(nthreads = -1)

url_income_rawdata_train <- "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
destfile_income_rawdata_train <- "income_rawdata_train.data"
download.file(url_income_rawdata_train, destfile_income_rawdata_train)
income_rawdata_train <- read.table("income_rawdata_train.data", sep = ",", strip.white = TRUE, na.strings = "?", stringsAsFactors = FALSE)
url_income_rawdata_test <- "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
destfile_income_rawdata_test <- "income_rawdata_test.data"
download.file(url_income_rawdata_test, destfile_income_rawdata_test)
income_rawdata_test <- read.table("income_rawdata_test.data", sep = ",", skip = 1, strip.white = TRUE, na.strings = "?", stringsAsFactors = FALSE)
income_rawdata_test$V15 = substr(income_rawdata_test$V15, 1, nchar(income_rawdata_test$V15)-1)
income <- rbind(income_rawdata_train, income_rawdata_test)
income <- na.omit(income)
names(income) <- c("age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income_50K")
income %>% group_by(income_50K) %>% summarise(Number_of_Observations = n())

write.csv(income, file="income.csv", row.names = FALSE)
income_h2o <- h2o.importFile("income.csv")
income_h2o_split <- h2o.splitFrame(income_h2o, ratios = c(0.6,0.2), seed = 1234)
income_h2o_train <- income_h2o_split[[1]]
income_h2o_valid <- income_h2o_split[[2]]
income_h2o_test <- income_h2o_split[[3]]

predictors <- names(income_h2o_train)[-15]
hyper_params <- list(alpha = seq(from = 0, to = 1, by = 0.001),
                     lambda = seq(from = 0, to = 1, by = 0.000001)
                     )
search_criteria <- list(strategy = "RandomDiscrete",
                        max_runtime_secs = 10*3600,
                        max_models = 100,
                        stopping_metric = "AUC", 
                        stopping_tolerance = 0.00001, 
                        stopping_rounds = 5, 
                        seed = 1234
                        )
models_glm <- h2o.grid(algorithm = "glm", grid_id = "grd_glm", x = predictors, y = "income_50K", training_frame = income_h2o_train, validation_frame = income_h2o_valid, nfolds = 0, family = "binomial", 
                       hyper_params = hyper_params, search_criteria = search_criteria, stopping_metric = "AUC", stopping_tolerance = 1e-5, stopping_rounds = 5, seed = 1234)
models_glm_sort <- h2o.getGrid(grid_id = "grd_glm", sort_by = "auc", decreasing = TRUE)
models_glm_best <- h2o.getModel(models_glm_sort@model_ids[[1]])
models_glm_best@allparameters
models_glm_best@model$validation_metrics@metrics$AUC
perf_glm_best <- h2o.performance(models_glm_best, income_h2o_valid)
plot(perf_glm_best, type="roc", main="ROC Curve for Best Logistic Regression Model")
h2o.varimp(models_glm_best)

hyper_params <- list(ntrees = 10000,  ## early stopping
                     max_depth = 5:15, 
                     min_rows = c(1,5,10,20,50,100),
                     nbins = c(30,100,300),
                     nbins_cats = c(64,256,1024),
                     sample_rate = c(0.7,1),
                     mtries = c(-1,2,6)
                     )
search_criteria <- list(strategy = "RandomDiscrete",
                        max_runtime_secs = 10*3600,
                        max_models = 100,
                        stopping_metric = "AUC", 
                        stopping_tolerance = 0.00001, 
                        stopping_rounds = 5, 
                        seed = 1234
                        )
models_rf <- h2o.grid(algorithm = "randomForest", grid_id = "grd_rf", x = predictors, y = "income_50K", training_frame = income_h2o_train, validation_frame = income_h2o_valid, nfolds = 0, hyper_params = hyper_params, 
                      search_criteria = search_criteria, stopping_metric = "AUC", stopping_tolerance = 1e-3, stopping_rounds = 2, seed = 1234)
models_rf_sort <- h2o.getGrid(grid_id = "grd_rf", sort_by = "auc", decreasing = TRUE)
models_rf_best <- h2o.getModel(models_rf_sort@model_ids[[1]])
models_rf_best@allparameters
models_rf_best@model$validation_metrics@metrics$AUC
perf_rf_best <- h2o.performance(models_rf_best, income_h2o_valid)
plot(perf_rf_best, type="roc", main="ROC Curve for Best Random Forest Model")
h2o.varimp(models_rf_best)

hyper_params <- list(ntrees = 10000,  ## early stopping
                     max_depth = 5:15, 
                     min_rows = c(1,5,10,20,50,100),
                     learn_rate = c(0.001,0.01,0.1),  
                     learn_rate_annealing = c(0.99,0.999,1),
                     sample_rate = c(0.7,1),
                     col_sample_rate = c(0.7,1),
                     nbins = c(30,100,300),
                     nbins_cats = c(64,256,1024)
                     )
search_criteria <- list(strategy = "RandomDiscrete",
                        max_runtime_secs = 10*3600,
                        max_models = 100,
                        stopping_metric = "AUC", 
                        stopping_tolerance = 0.00001, 
                        stopping_rounds = 5, 
                        seed = 1234
                        )
models_gbm <- h2o.grid(algorithm = "gbm", grid_id = "grd_gbm", x = predictors, y = "income_50K", training_frame = income_h2o_train, validation_frame = income_h2o_valid, nfolds = 0, hyper_params = hyper_params, 
                       search_criteria = search_criteria, stopping_metric = "AUC", stopping_tolerance = 1e-3, stopping_rounds = 2, seed = 1234)
models_gbm_sort <- h2o.getGrid(grid_id = "grd_gbm", sort_by = "auc", decreasing = TRUE)
models_gbm_best <- h2o.getModel(models_gbm_sort@model_ids[[1]])
models_gbm_best@allparameters
models_gbm_best@model$validation_metrics@metrics$AUC
perf_gbm_best <- h2o.performance(models_gbm_best, income_h2o_valid)
plot(perf_gbm_best, type="roc", main="ROC Curve for Best Gradient Boosting Model")

hyper_params <- list(activation = c("Rectifier", "Maxout", "Tanh", "RectifierWithDropout", "MaxoutWithDropout", "TanhWithDropout"), 
                     hidden = list(c(50, 50, 50, 50), c(200, 200), c(200, 200, 200), c(200, 200, 200, 200)), 
                     epochs = c(50, 100, 200), 
                     l1 = c(0, 0.00001, 0.0001), 
                     l2 = c(0, 0.00001, 0.0001), 
                     adaptive_rate = c(TRUE, FALSE), 
                     rate = c(0, 0.1, 0.005, 0.001), 
                     rate_annealing = c(1e-8, 1e-7, 1e-6), 
                     rho = c(0.9, 0.95, 0.99, 0.999), 
                     epsilon = c(1e-10, 1e-8, 1e-6, 1e-4), 
                     momentum_start = c(0, 0.5),
                     momentum_stable = c(0.99, 0.5, 0), 
                     input_dropout_ratio = c(0, 0.1, 0.2)
                     )
search_criteria <- list(strategy = "RandomDiscrete",
                        max_runtime_secs = 10*3600,
                        max_models = 100,
                        stopping_metric = "AUC", 
                        stopping_tolerance = 0.00001, 
                        stopping_rounds = 5, 
                        seed = 1234
                        )
models_dl <- h2o.grid(algorithm = "deeplearning", grid_id = "grd_dl", x = predictors, y = "income_50K", training_frame = income_h2o_train, validation_frame = income_h2o_valid, nfolds = 0, hyper_params = hyper_params, 
                      search_criteria = search_criteria, stopping_metric = "AUC", stopping_tolerance = 1e-3, stopping_rounds = 2, seed = 1234)
models_dl_sort <- h2o.getGrid(grid_id = "grd_dl", sort_by = "auc", decreasing = TRUE)
models_dl_best <- h2o.getModel(models_dl_sort@model_ids[[1]])
models_dl_best@allparameters
models_dl_best@model$validation_metrics@metrics$AUC
perf_dl_best <- h2o.performance(models_dl_best, income_h2o_valid)
plot(perf_dl_best, type="roc", main="ROC Curve for Best Neural Network Model")

md_lr <- h2o.glm(x = predictors, y = "income_50K", training_frame = income_h2o_train, nfolds = 10, fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE, seed = 1234, family = "binomial", alpha = 0.144, lambda = 0.005041)
md_rf <- h2o.randomForest(x = predictors, y = "income_50K", training_frame = income_h2o_train, nfolds = 10, fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE, ntrees = 10000, max_depth = 15, min_rows = 10, 
                          nbins = 30, nbins_cats = 64, mtries = 2, sample_rate =1, stopping_metric = "AUC", stopping_tolerance = 1e-3, stopping_rounds = 2, seed = 1234 )
md_gbm <- h2o.gbm(x = predictors, y = "income_50K", training_frame = income_h2o_train, nfolds = 10, fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE, ntrees = 10000, max_depth = 7, min_rows = 5, nbins = 300, 
                  nbins_cats = 64, learn_rate = 0.1, learn_rate_annealing = 1, sample_rate = 1, col_sample_rate = 1, stopping_metric = "AUC", stopping_tolerance = 1e-3, stopping_rounds = 2, seed = 1234 )
md_dl <- h2o.deeplearning(x = predictors, y = "income_50K", training_frame = income_h2o_train, nfolds = 10, fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE, activation = "Rectifier", hidden = c(200, 200), 
                          epochs = 200, adaptive_rate = FALSE, rho = 0.95, epsilon = 1e-10, rate = 0.005, rate_annealing = 1e-6, momentum_start = 0.5, momentum_stable = 0.99, input_dropout_ratio = 0.1, l1 = 1e-4, l2 = 0, 
                          stopping_metric = "AUC", stopping_tolerance = 1e-3, stopping_rounds = 2, seed = 1234 )
md_ens <- h2o.stackedEnsemble(x = predictors, y = "income_50K", training_frame = income_h2o_train, base_models = list(md_lr@model_id, md_rf@model_id, md_gbm@model_id, md_dl@model_id))

h2o.auc(h2o.performance(md_lr, income_h2o_valid))
h2o.auc(h2o.performance(md_rf, income_h2o_valid))
h2o.auc(h2o.performance(md_gbm, income_h2o_valid))
h2o.auc(h2o.performance(md_dl, income_h2o_valid))
h2o.auc(h2o.performance(md_ens, income_h2o_valid))
h2o.getModel(md_ens@model$metalearner$name)@model$coefficients_table

h2o.auc(h2o.performance(md_gbm, income_h2o_test))
perf_best_model <- h2o.performance(md_gbm, income_h2o_test)
plot(perf_best_model, type="roc", main="ROC Curve for Best Gradient Boosting Model using Grid Random Search")


