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
rec <- Recommender(train, method = "IBCF", 
                   param=list(method="Cosine"))

##obtaining list of recommended movies for first user
pre_ibcf <- predict(rec, ratingmat[1], n = 10)


pre_List_ibcf = as(pre_ibcf, "list")

##fetching information from original csv by joining on movieID
library(dplyr)

##df is data frame

pre_df_ibcf=data.frame(pre_List_ibcf)

colnames(pre_df_ibcf)="movieId"

pre_df_ibcf$movieId=as.numeric(levels(pre_df_ibcf$movieId))

names_ibcf=left_join(pre_df_ibcf, movies, by="movieId")

names_ibcf
