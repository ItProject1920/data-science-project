library("recommenderlab")
data("MovieLense")
MovieLense1 <- MovieLense[rowCounts(MovieLense) >1,]

#UBCF Predictions
train <- MovieLense1[1:900]
rec <- Recommender(train, method = "UBCF", 
                   param=list(method="Cosine"))
pre <- predict(rec, MovieLense[1], n = 10)
as(pre, "list")

#IBCF Predictions
rec1 <- Recommender(train, method = "IBCF",
                    param=list(method="Cosine"))
pre1 <- predict(rec1, MovieLense1[1], n = 10)
as(pre1, "list")

#Evaluation
recommenderRegistry$get_entries(dataType = "realRatingMatrix")
scheme <- evaluationScheme(MovieLense, method = "split", train = .9,
                           k = 1, given = 10, goodRating = 4)

algorithms <- list(
 # "user-based CF" = list(name="UBCF", param=list(normalize = "Z-score",
 #                                                method="Cosine",
 #                                                nn=30)),
 # "item-based CF" = list(name="IBCF", param=list(normalize = "Z-score"
 # ))
  IBCF_cos = list(name = "IBCF", param = list(method = "cosine")),
  IBCF_cor = list(name = "IBCF", param = list(method = "pearson")),
  UBCF_cos = list(name = "UBCF", param = list(method = "cosine")),
  UBCF_cor = list(name = "UBCF", param = list(method = "pearson"))
)

results <- evaluate(scheme,algorithms, n=c(1, 3, 5, 10, 15, 20))

#Graph Plot
plot(results, annotate = 1:2, legend="topright")
plot(results, "prec/rec", annotate=2)