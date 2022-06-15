# Incense

This is a simple interface for Torch for R, which is useful when creating a relatively simple network quickly.

NOTE: This library is under development.

## Example
```{r}
library(incense)
data(iris)
train_ind <- sample.int(nrow(iris),floor(nrow(iris)*0.8))
train_data <- iris[train_ind,]
val_data <- iris[-train_ind,]
dls <- prepare_dataloader(train_x=train_data[,-5],
                          train_y=as.integer(train_data[,5]),
                          xtype=torch_float32(), ytype=torch_long(),
                          val_x=val_data[,-5],
                          val_y=as.integer(val_data[,5]))

res <- train(dl=dls$dataloader$train,
             val_dl=dls$dataloader$validate,
             topology=list(
               list("linear",4,10),
               list("relu"),
               list("linear",10,3)
             ),
             optim="adam",
             loss="crossentropy",
             nepoch=30,
             lr=0.01
             )

predict(res$model,val_data[,1:4])
```

