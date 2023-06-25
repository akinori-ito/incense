#' Do training of the neural network
#' @param dl A dataloader of the training data
#' @param val_dl A dataloader of the validation data
#' @param topology Topology of the network
#' @param model model to train
#' @param optim Name of the optimizer
#' @param loss Name of the loss function
#' @param nepoch Number of epochs
#' @param lr The learning rate
#' @param save_model If TRUE, save the models obtained at each epoch
#' @param save_filename Template of the saved files. %d is replaced with the epoch number
#' @param check Check the dimension of the parameter and network, type of the data, etc.
#' @param verbose Output the loss during the training
#' @param device torch_device for training
#' @param hook hook function that is executed after every epoch, which should have three arguments: model, training dataloader, and validation dataloader
#' @returns A list of the trained model, the training loss and the validation loss
#' @export
train <- function(dl,val_dl=NULL,topology=NULL,model=NULL,optim,loss,nepoch,lr=0.01,
                  save_model=FALSE,save_filename="model%d.torch",
                  check=TRUE,verbose=TRUE,device=NULL, hook=NULL) {
  if (is.null(model)) {
    if (is.null(topology)) {
      stop("Specify either topology or models")
    }
    model <- generate_net(topology,device=device)
  } else if (!is.null(topology)) {
    warning("Both model and topology are given; ignoring topology")
  } else {
    topology <- model$topology
  }
  optim <- choose_optim(optim,model$parameters,lr=lr)
  lossfunc <- choose_loss(loss)
  consistency <- get_consistency(topology)
  train_loss <- c()
  valid_loss <- c()
  for (epoch in seq_len(nepoch)) {
    if (verbose)
      cat("Epoch",epoch,"\n")
    model$train()
    bloss <- c()
    iter <- dl$.iter()
    while (TRUE) {
      b <- iter$.next()
      if (!is.list(b)) break
      optim$zero_grad()
      if (check)
        dim_check(consistency,dim(b$x))
      output <- model$forward(b$x)
      y <- torch::torch_squeeze(b$y)
      if (check)
        loss_check(loss,output,y)
      L <- lossfunc(output,y)
      L$backward()
      optim$step()
      bloss <- c(bloss,L$item())
      if (verbose)
        cat(".")
    }
    train_loss <- c(train_loss,mean(bloss))
    if (verbose)
      cat("\ntrain_loss=",mean(bloss),"\n")
    if (!is.null(val_dl)) {
      model$eval()
      vloss <- c()
      iter <- val_dl$.iter()
      while (TRUE) {
        b <- iter$.next()
        if (!is.list(b)) break
        if (check)
          dim_check(consistency,dim(b$x))
        output <- model$forward(b$x)
        y <- torch::torch_squeeze(b$y)
        if (check)
          loss_check(loss,output,y)
        L <- lossfunc(output,y)
        vloss <- c(vloss,L$item())
        if (verbose)
          cat(".")
      }
      valid_loss <- c(valid_loss,mean(vloss))
      if (verbose)
        cat("\nvalid_loss=",mean(vloss))
    }
    if (verbose)
      cat("\n")
    if (save_model) {
      torch::torch_save(model,sprintf(save_filename,epoch))
    }
    if (!is.null(hook)) {
      hook(model,dl,val_dl)
    } 
  }
  list(model=model,
       train_loss=train_loss,
       valid_loss=valid_loss)
}

#' Predict
#' @param model The neural network (nn_module)
#' @param x A matrix or torch_tensor
#' @return An output matrix
#' @export
predict.nn_module <- function(model,x) {
  if (!is.matrix(x)) {
    x <- as.matrix(x)
  }
  if (!inherits(x,"torch_tensor")) {
    x <- torch::torch_tensor(x,dtype=torch::torch_float32(),device=model$device)
  }
  y <- model$forward(x)
  as.array(y$cpu())
}
