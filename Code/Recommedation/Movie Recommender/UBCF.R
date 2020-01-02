movies=read.csv("movies.csv")
ratings=read.csv("ratings1.csv")

library(stringi)
library(reshape2)
library(recommenderlab)

##creating a rating matrix
ratingmat = dcast(ratings, userId~movieId, value.var = "rating", na.rm=FALSE)
ratingmat = as.matrix(ratingmat[,-1])

ratingmat = as(ratingmat, "realRatingMatrix")
ratingmat = normalize(ratingmat)

##train the model on 550 rows out of 610
mov <- ratingmat[rowCounts(ratingmat) >1,]
train <- mov[1:550]
rec <- Recommender(train, method = "UBCF", 
                   param=list(method="Cosine"))

##obtaining list of recommended movies for first user
pre <- predict(rec, ratingmat[1], n = 10)


pre_List = as(pre, "list")

##fetching information from original csv by joining on movieID
library(dplyr)

##df is data frame

pre_df=data.frame(pre_List)

colnames(pre_df)="movieId"

pre_df$movieId=as.numeric(levels(pre_df$movieId))

names=left_join(pre_df, movies, by="movieId")

names
