# ============================================================
# ST3189: Machine Learning - Assessed Coursework Project
# Candidate Number: 230573190
# Dataset: English Premier League Match Data (2000/01 - 2017/18)
# Source: https://www.kaggle.com/datasets/saife245/english-premier-league
#         Download 'final_dataset.csv' and set it as the working directory
# ============================================================
# NOTE: Run each section in order.
# Install packages:
install.packages(c("tidyverse", "RColorBrewer", "mlr3verse",
                   "mlr3pipelines", "mlr3learners", "mlr3viz",
                   "glmnet", "ranger", "kknn", "precrec"))
# ============================================================


# ---- 0. LIBRARIES & COLOUR PALETTE -------------------------
library(tidyverse)
library(RColorBrewer)
library(mlr3verse)
library(mlr3pipelines)
library(mlr3learners)
library(mlr3viz)
library(glmnet)
library(precrec)

colours <- brewer.pal(n = 12, name = "Paired")
set.seed(42)


# ---- 1. DATA LOADING & INSPECTION --------------------------
data <- read.csv("final_dataset.csv", stringsAsFactors = FALSE)

cat("=== Dataset Overview ===\n")
cat("Dimensions:", nrow(data), "rows x", ncol(data), "columns\n")
cat("Date range:", data$Date[1], "to", data$Date[nrow(data)], "\n")
cat("Unique teams:", length(unique(c(data$HomeTeam, data$AwayTeam))), "\n")
cat("\nFull-time result distribution:\n")
print(table(data$FTR))
cat("Home win rate:", round(mean(data$FTR == "H"), 3), "\n")

cat("\nSummary of key features:\n")
print(summary(data[, c("HTP", "ATP", "HTGD", "ATGD", "HTFormPts",
                       "DiffPts", "FTHG")]))


# ---- 2. FEATURE SELECTION & TARGET VARIABLES ---------------
features <- c("HTP", "ATP", "HTGD", "ATGD",
              "HTFormPts", "ATFormPts",
              "DiffPts", "DiffFormPts",
              "HTWinStreak3", "ATWinStreak3",
              "HTLossStreak3", "ATLossStreak3",
              "MW")

data$FTR_fac <- factor(
  ifelse(data$FTR == "H", "HomeWin", "NotHomeWin"),
  levels = c("HomeWin", "NotHomeWin")
)
data$FTR_bin <- as.integer(data$FTR == "H")

n         <- nrow(data)
train_idx <- 1:floor(0.8 * n)
test_idx  <- (floor(0.8 * n) + 1):n

cat("\nTrain observations:", length(train_idx), "\n")
cat("Test  observations:", length(test_idx),  "\n")


# ---- 3. EXPLORATORY DATA ANALYSIS --------------------------
par(mfrow = c(2, 2))

hist(data$FTHG, breaks = 0:10 - 0.5, col = colours[2],
     xlab = "Goals Scored", main = "Home Goals Distribution", xaxt = "n")
axis(1, at = 0:9)

hist(data$FTAG, breaks = 0:8 - 0.5, col = colours[4],
     xlab = "Goals Scored", main = "Away Goals Distribution", xaxt = "n")
axis(1, at = 0:7)

boxplot(HTP ~ FTR_bin, data = data,
        names = c("Not Home Win", "Home Win"),
        col = colours[c(4, 2)],
        ylab = "Home Points Per Game (HTP)",
        main = "Home Strength by Outcome")

boxplot(HTGD ~ FTR_bin, data = data,
        names = c("Not Home Win", "Home Win"),
        col = colours[c(4, 2)],
        ylab = "Home Goal Diff Per Game (HTGD)",
        main = "Home Goal Diff by Outcome")

par(mfrow = c(1, 1))

cor_mat <- cor(data[train_idx, features], use = "complete.obs")
cat("\nTop pairwise correlations (|r| > 0.6):\n")
for (i in 1:(length(features) - 1)) {
  for (j in (i + 1):length(features)) {
    r <- cor_mat[i, j]
    if (abs(r) > 0.6)
      cat(sprintf("  %s ~ %s : %.3f\n", features[i], features[j], r))
  }
}


# ---- 4. UNSUPERVISED LEARNING ------------------------------

## 4.1  Principal Component Analysis -------------------------
X_scaled <- scale(data[, features])
pca_out  <- prcomp(X_scaled[train_idx, ], scale. = FALSE)

pve     <- (pca_out$sdev^2) / sum(pca_out$sdev^2)
cum_pve <- cumsum(pve)

cat("\n=== PCA Results ===\n")
cat(sprintf("PC1: %.1f%%  |  PC2: %.1f%%  |  Cumulative (2 PCs): %.1f%%\n",
            pve[1] * 100, pve[2] * 100, cum_pve[2] * 100))
cat(sprintf("Cumulative (5 PCs): %.1f%%\n", cum_pve[5] * 100))

par(mfrow = c(1, 2))
plot(seq_along(pve), pve * 100, type = "b", pch = 19, col = colours[2],
     xlab = "Principal Component", ylab = "% Variance Explained",
     main = "Scree Plot", ylim = c(0, 35))
abline(v = 2, lty = 2, col = "grey60")

plot(seq_along(cum_pve), cum_pve * 100, type = "b", pch = 19, col = colours[4],
     xlab = "Principal Component", ylab = "Cumulative % Variance",
     main = "Cumulative Explained Variance", ylim = c(0, 105))
abline(h = 80, lty = 2, col = "grey60")
par(mfrow = c(1, 1))

cat("\nPC1 loadings (sorted by |loading|):\n")
print(round(sort(abs(pca_out$rotation[, 1]), decreasing = TRUE), 3))
cat("\nPC2 loadings (sorted by |loading|):\n")
print(round(sort(abs(pca_out$rotation[, 2]), decreasing = TRUE), 3))

biplot(pca_out, scale = 0, cex = c(0.3, 0.8),
       col = c(colours[2], colours[8]),
       main = "PCA Biplot (PC1 vs PC2) - Training Set")


## 4.2  K-means Clustering ------------------------------------
wss <- sapply(1:8, function(k) {
  kmeans(X_scaled[train_idx, ], centers = k, nstart = 25)$tot.withinss
})

plot(1:8, wss, type = "b", pch = 19, col = colours[2],
     xlab = "Number of Clusters (K)", ylab = "Total Within-Cluster SS",
     main = "K-means Elbow Plot")
abline(v = 3, lty = 2, col = "grey60")

set.seed(42)
km3 <- kmeans(X_scaled[train_idx, ], centers = 3, nstart = 25)
cat("\n=== K-means (K=3) Cluster Sizes ===\n")
print(table(km3$cluster))

df_cl <- data.frame(
  data[train_idx, c(features, "FTHG")],
  FTR_bin = data$FTR_bin[train_idx],
  cluster  = factor(km3$cluster)
)

# Adjustment: aggregate returns cluster as factor; round() only numeric columns
profile <- aggregate(
  cbind(DiffPts, DiffFormPts, HTGD, ATGD, FTHG, FTR_bin) ~ cluster,
  data = df_cl, FUN = mean
)
cat("\n=== Cluster Profiles ===\n")
profile_print        <- profile
profile_print[, -1]  <- round(profile_print[, -1], 3)
print(profile_print)

scores         <- as.data.frame(pca_out$x[, 1:2])
scores$cluster <- factor(km3$cluster)

plot(scores$PC1, scores$PC2,
     col  = colours[as.integer(scores$cluster) * 2],
     pch  = 20, cex = 0.5,
     xlab = "PC1 - Match Competitiveness (31.6%)",
     ylab = "PC2 - Overall Match Quality (24.1%)",
     main = "K-means Clusters (K=3) in PCA Space")
legend("topright",
       legend = c("Cluster 1: Balanced",
                  "Cluster 2: Home Favoured",
                  "Cluster 3: Away Favoured"),
       col = colours[c(2, 4, 6)], pch = 20, cex = 0.8)


## 4.3  Hierarchical Clustering (300-obs subsample) -----------
set.seed(42)
sub_idx  <- sample(train_idx, 300)
dist_mat <- dist(X_scaled[sub_idx, ], method = "euclidean")
hc_comp  <- hclust(dist_mat, method = "complete")

plot(hc_comp, labels = FALSE, hang = -1,
     main = "Hierarchical Clustering - Complete Linkage (n=300)",
     xlab = "", sub = "")
abline(h = 4.8, lty = 2, col = "red")
legend("topright", legend = "Cut at h=4.8 (3 clusters)",
       lty = 2, col = "red", cex = 0.8)

hc_labels <- cutree(hc_comp, k = 3)
cat("\nHierarchical clustering (subsample) sizes:\n")
print(table(hc_labels))


# ---- 5. REGRESSION: PREDICTING HOME GOALS ------------------

## 5.1  Task and measure
reg_data <- data.frame(data[, features], FTHG = data$FTHG)
task_reg  <- TaskRegr$new("epl_goals", backend = reg_data, target = "FTHG")
m_mse     <- msr("regr.mse")

## 5.2  OLS Linear Regression
glrn_lm <- GraphLearner$new(po("scale") %>>% po("learner", lrn("regr.lm")))
glrn_lm$train(task_reg, row_ids = train_idx)
mse_lm <- glrn_lm$predict(task_reg, row_ids = test_idx)$score(m_mse)

## 5.3  Ridge Regression
lr_ridge <- lrn("regr.cv_glmnet")
lr_ridge$param_set$values <- list(alpha  = 0,
                                  lambda = 10^seq(-3, 3, length = 100),
                                  nfolds = 5)
glrn_ridge     <- GraphLearner$new(po("scale") %>>% po("learner", lr_ridge))
glrn_ridge$id  <- "scale.regr.ridge"          # unique ID required for benchmark
glrn_ridge$train(task_reg, row_ids = train_idx)
mse_ridge <- glrn_ridge$predict(task_reg, row_ids = test_idx)$score(m_mse)

## 5.4  Lasso Regression
lr_lasso <- lrn("regr.cv_glmnet")
lr_lasso$param_set$values <- list(alpha  = 1,
                                  lambda = 10^seq(-3, 3, length = 100),
                                  nfolds = 5)
glrn_lasso     <- GraphLearner$new(po("scale") %>>% po("learner", lr_lasso))
glrn_lasso$id  <- "scale.regr.lasso"          # unique ID required for benchmark
glrn_lasso$train(task_reg, row_ids = train_idx)
mse_lasso <- glrn_lasso$predict(task_reg, row_ids = test_idx)$score(m_mse)

## 5.5  Random Forest Regressor
# Adjustment: set num.trees via param_set$values, not as a direct lrn() argument
lr_rfr <- lrn("regr.ranger")
lr_rfr$param_set$values <- list(num.trees = 300, min.node.size = 5)
glrn_rfr <- GraphLearner$new(po("scale") %>>% po("learner", lr_rfr))
glrn_rfr$train(task_reg, row_ids = train_idx)
mse_rfr <- glrn_rfr$predict(task_reg, row_ids = test_idx)$score(m_mse)

## 5.6  Null baseline
baseline_pred <- rep(mean(data$FTHG[train_idx]), length(test_idx))
mse_null      <- mean((data$FTHG[test_idx] - baseline_pred)^2)

## 5.7  Regression results
cat("\n=== Regression Results (Test Set) ===\n")
reg_table <- data.frame(
  Model     = c("Null Baseline", "OLS Linear", "Ridge",
                "Lasso", "Random Forest"),
  Test_MSE  = round(c(mse_null, mse_lm, mse_ridge, mse_lasso, mse_rfr), 4),
  Test_RMSE = round(sqrt(c(mse_null, mse_lm, mse_ridge, mse_lasso, mse_rfr)), 4)
)
print(reg_table)

## 5.8  5-fold cross-validation benchmark
set.seed(1)
bmr_reg <- benchmark(
  benchmark_grid(
    tasks       = task_reg,
    learners    = list(glrn_lm, glrn_ridge, glrn_lasso, glrn_rfr),
    resamplings = rsmp("cv", folds = 5)
  )
)
cat("\n5-fold Cross-validated MSE:\n")
print(bmr_reg$aggregate(m_mse)[, c("learner_id", "regr.mse")])

## 5.9  Lasso selected features
X_tr     <- scale(as.matrix(data[train_idx, features]))
y_tr_reg <- data$FTHG[train_idx]
cv_las   <- cv.glmnet(X_tr, y_tr_reg, alpha = 1, nfolds = 5)
las_coef <- coef(cv_las, s = cv_las$lambda.min)
nz_idx   <- which(as.numeric(las_coef) != 0)
nz       <- las_coef[nz_idx, , drop = FALSE]
cat("\nLasso non-zero coefficients (at lambda.min):\n")
print(round(as.matrix(nz), 4))

## 5.10 Ridge coefficient path plot
rfit <- glmnet(X_tr, y_tr_reg, alpha = 0)
plot(rfit, xvar = "lambda", label = TRUE, col = colours,
     main = "Ridge Regression - Coefficient Paths")


# ---- 6. CLASSIFICATION: PREDICTING HOME WIN ----------------

## 6.1  Task and measures
cls_data <- data.frame(data[, features], FTR = data$FTR_fac)
task_cls  <- as_task_classif(cls_data, target = "FTR", positive = "HomeWin")
m_auc     <- msr("classif.auc")
m_acc     <- msr("classif.acc")

## 6.2  Logistic Regression (no regularisation)
ll_logit   <- lrn("classif.log_reg", predict_type = "prob")
glrn_logit <- GraphLearner$new(po("scale") %>>% po("learner", ll_logit))
glrn_logit$train(task_cls, row_ids = train_idx)
pred_logit <- glrn_logit$predict(task_cls, row_ids = test_idx)

## 6.3  Logistic Ridge (L2)
# Adjustment: removed family = "binomial" — set automatically by mlr3 for classif tasks
ll_lr <- lrn("classif.cv_glmnet", predict_type = "prob")
ll_lr$param_set$values <- list(alpha  = 0,
                               nfolds = 5,
                               lambda = 10^seq(-4, 2, length = 100))
glrn_lr     <- GraphLearner$new(po("scale") %>>% po("learner", ll_lr))
glrn_lr$id  <- "scale.classif.logistic_ridge" # unique ID required for benchmark
glrn_lr$train(task_cls, row_ids = train_idx)
pred_lr <- glrn_lr$predict(task_cls, row_ids = test_idx)

## 6.4  Logistic Lasso (L1)
ll_ll <- lrn("classif.cv_glmnet", predict_type = "prob")
ll_ll$param_set$values <- list(alpha  = 1,
                               nfolds = 5,
                               lambda = 10^seq(-4, 2, length = 100))
glrn_ll     <- GraphLearner$new(po("scale") %>>% po("learner", ll_ll))
glrn_ll$id  <- "scale.classif.logistic_lasso" # unique ID required for benchmark
glrn_ll$train(task_cls, row_ids = train_idx)
pred_ll <- glrn_ll$predict(task_cls, row_ids = test_idx)

## 6.5  K-Nearest Neighbours (k = 15)
# predict_type = "prob" required so benchmark can compute AUC and plot ROC curves
ll_knn   <- lrn("classif.kknn", k = 15, predict_type = "prob")
glrn_knn <- GraphLearner$new(po("scale") %>>% po("learner", ll_knn))
glrn_knn$train(task_cls, row_ids = train_idx)
pred_knn <- glrn_knn$predict(task_cls, row_ids = test_idx)

## 6.6  Random Forest Classifier
# Adjustment: set num.trees via param_set$values
ll_rfc <- lrn("classif.ranger", predict_type = "prob")
ll_rfc$param_set$values <- list(num.trees = 300)
glrn_rfc <- GraphLearner$new(po("scale") %>>% po("learner", ll_rfc))
glrn_rfc$train(task_cls, row_ids = train_idx)
pred_rfc <- glrn_rfc$predict(task_cls, row_ids = test_idx)

## 6.7  Classification results table
cat("\n=== Classification Results (Test Set) ===\n")
cls_table <- data.frame(
  Model    = c("Logistic (OLS)", "Logistic (Ridge)", "Logistic (Lasso)",
               "KNN (k=15)", "Random Forest"),
  Test_AUC = round(c(pred_logit$score(m_auc),
                     pred_lr$score(m_auc),
                     pred_ll$score(m_auc),
                     pred_knn$score(m_auc),   # now available with predict_type = "prob"
                     pred_rfc$score(m_auc)), 4),
  Test_Acc = round(c(pred_logit$score(m_acc),
                     pred_lr$score(m_acc),
                     pred_ll$score(m_acc),
                     pred_knn$score(m_acc),
                     pred_rfc$score(m_acc)), 4)
)
print(cls_table)

## 6.8  5-fold cross-validation benchmark
set.seed(1)
bmr_cls <- benchmark(
  benchmark_grid(
    tasks       = task_cls,
    learners    = list(glrn_logit, glrn_lr, glrn_ll, glrn_knn, glrn_rfc),
    resamplings = rsmp("cv", folds = 5)
  )
)
cat("\n5-fold Cross-validated AUC:\n")
print(bmr_cls$aggregate(m_auc)[, c("learner_id", "classif.auc")])

## 6.9  Confusion matrix and metrics — Logistic Ridge
cat("\nConfusion Matrix - Logistic Ridge (Test Set):\n")
cm <- pred_lr$confusion
print(cm)

TP <- cm["HomeWin",    "HomeWin"]
TN <- cm["NotHomeWin", "NotHomeWin"]
FP <- cm["HomeWin",    "NotHomeWin"]
FN <- cm["NotHomeWin", "HomeWin"]
cat(sprintf("Sensitivity (Recall): %.4f\n", TP / (TP + FN)))
cat(sprintf("Specificity:          %.4f\n", TN / (TN + FP)))
cat(sprintf("Accuracy:             %.4f\n", (TP + TN) / sum(cm)))
cat(sprintf("AUC:                  %.4f\n", pred_lr$score(m_auc)))

## 6.10 ROC curves — requires mlr3viz (loaded at top)
autoplot(bmr_cls, type = "roc") +
  ggplot2::labs(title = "ROC Curves - 5-fold CV (Classification Models)")

## 6.11 Logistic Ridge coefficients (via glmnet directly for interpretability)
X_tr_cls  <- scale(as.matrix(data[train_idx, features]))
y_tr_cls  <- data$FTR_bin[train_idx]
cv_lr_cls <- cv.glmnet(X_tr_cls, y_tr_cls,
                       alpha = 0, family = "binomial", nfolds = 5)

lr_cls_coef <- as.numeric(
  coef(cv_lr_cls, s = cv_lr_cls$lambda.min)
)[-1]   # drop intercept

coef_df <- data.frame(
  Feature     = features,
  Coefficient = lr_cls_coef
)
coef_df <- coef_df[order(abs(coef_df$Coefficient), decreasing = TRUE), ]

cat("\nLogistic Ridge Coefficients (sorted by |coef|):\n")
# Adjustment: print only numeric rounding — Feature column is character
coef_df_print              <- coef_df
coef_df_print$Coefficient  <- round(coef_df_print$Coefficient, 4)
print(coef_df_print)

barplot(coef_df$Coefficient,
        names.arg = coef_df$Feature,
        las = 2,
        col = ifelse(coef_df$Coefficient > 0, colours[2], colours[6]),
        main = "Logistic Ridge - Coefficient Magnitudes",
        ylab = "Coefficient (standardised features)",
        cex.names = 0.75)
abline(h = 0, lty = 2)


# ============================================================
# END OF SCRIPT
# ============================================================