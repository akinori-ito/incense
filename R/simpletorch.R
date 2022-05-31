get_size <- function(x) {
  dd <- dim(x)
  if (is.null(dd)) {
    return(length(x))
  }
  dd[1]
}

check_size_xy <- function(x,y) {
  xsize <- get_size(x)
  ysize <- get_size(y)
  if (xsize != ysize) {
    stop("Size of x and y differs")
  }
}

#'
#' Create dataset from data
#' @param name Name of the dataset
#' @param train_x A matrix for training data
#' @param train_y A matrix/vector for training label
#' @param val_x A matrix for validation data (can be NULL)
#' @param val_y A matrix/vector for validation label (can be NULL)
#' @param xtype torch_dtype of x
#' @param ytype torch_dtype of y
#' @param device torch_device
#' @returns A list of two datasets, "training" and "validation"
#'          (validation is NULL if val_x or val_y is not given)
#' @export
#' @importFrom torch dataset torch_tensor torch_float32 torch_long
create_dataset <- function(name,train_x,train_y,
                           val_x=NULL, val_y=NULL,
                           xtype,ytype,device) {
  ds <- torch::dataset(
    name = name,
    initialize = function(x,y,xtype,ytype,device) {
      if (!inherits(x,"torch_tensor")) {
        x <- torch::torch_tensor(as.matrix(x),dtype=xtype,device=device)
      }
      if (!inherits(y,"torch_tensor")) {
        y <- torch::torch_tensor(as.matrix(y),dtype=ytype,device=device)
      }
      self$x <- x
      self$y <- y
      self$size <- dim(x)[1]
    },
    .getitem = function(i) {
      list(x=self$x[i],y=self$y[i])
    },
    .length = function(i) {
      self$size
    }
  )
  check_size_xy(train_x, train_y)
  train_ds <- ds(train_x,train_y,xtype,ytype,device)
  if (is.null(val_x) || is.null(val_y)) {
    val_ds <- NULL
    if (!is.null(val_x) || !is.null(val_y)) {
      warning("One of val_x and val_y is null, and the other one is non-null")
    }
  } else {
    check_size_xy(val_x, val_y)
    val_ds <- ds(val_x, val_y, xtype, ytype, device)
  }
  list(train=train_ds, validate=val_ds)
}

#' Create dataloader from data
#' @param train_x A matrix for training data
#' @param train_y A matrix/vector for training label
#' @param val_x A matrix for validation data (can be NULL)
#' @param val_y A matrix/vector for validation label (can be NULL)
#' @param xtype torch_dtype of x
#' @param ytype torch_dtype of y
#' @param device torch_device
#' @param batchsize The batch size (default: 16)
#' @param train_shuffle Shuffle the training data or not (default: TRUE)
#' @param val_shuffle Shuffle the validation data or not (default: FALSE)
#' @param name Name of the dataloader (optional)
#' @returns A list of two lists; "dataloader" has two data loaders
#'          "train" and "validate", and "dataset" also has "train" and "validate"
#' @export
#' @importFrom torch dataloader torch_device is_torch_device
prepare_dataloader <- function(train_x, train_y,
                               xtype, ytype,
                               batchsize=16,
                               train_shuffle=TRUE,
                               val_shuffle=FALSE,
                               name="default",
                               val_x=NULL, val_y=NULL,
                               device=NULL) {
  if (is.null(device)) {
    device <- torch::torch_device("CPU")
  }
  if (!torch::is_torch_device(device)) {
    stop(device,"is not a torch device")
  }
  if (!torch::is_torch_dtype(xtype)) {
    stop(xtype,"is not a torch dtype")
  }
  if (!torch::is_torch_dtype(ytype)) {
    stop(ytype, "is not a torch dtype")
  }
  ds <- create_dataset(
    name, train_x, train_y, val_x, val_y, xtype, ytype, device
  )
  train_dl <- torch::dataloader(ds$train, batch_size=batchsize, shuffle=train_shuffle)
  if (!is.null(ds$validate)) {
    val_dl <- torch::dataloader(ds$validate, batch_size=batchsize, shuffle=val_shuffle)
  } else {
    val_dl <- NULL
  }
  list(dataloader=list(
    train=train_dl, validate=val_dl),
    dataset=ds)
}

#' @importFrom torch optim_adadelta optim_adagrad optim_adam optim_asgd optim_lbfgs optim_rmsprop optim_sgd
choose_optim <- function(optim_name,params,...) {
  opt <- switch(
    optim_name,
    "adadelta"=optim_adadelta,
    "adagrad"=optim_adagrad,
    "adam"=optim_adam,
    "asgd"=optim_asgd,
    "lbfgs"=optim_lbfgs,
    "rmsprop"=optim_rmsprop,
    "sgd"=optim_sgd
  )
  opt(params,...)
}

#' @importFrom torch nn_linear nn_conv1d nn_conv2d nn_max_pool1d nn_max_pool2d
#'                   nn_max_unpool1d nn_max_unpool2d nn_dropout nn_dropout2d
#'                   nn_layer_norm nn_lstm nn_relu nn_sigmoid nn_tanh nn_softmax
#'                   nn_flatten nn_unflatten
choose_net <- function(topolist) {
  name <- topolist[[1]]
  if (length(topolist) == 1) {
    arg <- NULL
  } else {
    arg <- topolist[2:length(topolist)]
  }
  nnfunc <- switch(
    name,
    "linear"=torch::nn_linear,
    "conv1d"=torch::nn_conv1d,
    "conv2d"=torch::nn_conv2d,
    "maxpool1d"=torch::nn_max_pool1d,
    "maxpool2d"=torch::nn_max_pool2d,
    "maxunpool1d"=torch::nn_max_unpool1d,
    "maxunpool2d"=torch::nn_max_unpool2d,
    "dropout"=torch::nn_dropout,
    "dropout2d"=torch::nn_dropout2d,
    "layernorm"=torch::nn_layer_norm,
    "batchnorm1d"=torch::nn_batch_norm1d,
    "batchnorm2d"=torch::nn_batch_norm2d,
    "lstm"=torch::nn_lstm,
    "relu"=torch::nn_relu,
    "sigmoid"=torch::nn_sigmoid,
    "tanh"=torch::nn_tanh,
    "softmax"=torch::nn_softmax,
    "flatten"=torch::nn_flatten,
    "unflutten"=torch::nn_unflatten,
    "embedding"=torch::nn_embedding
  )
  if (is.null(arg)) {
    nnfunc()
  } else {
    do.call(nnfunc,arg)
  }
}

#' @importFrom torch nn_mse_loss nn_nll_loss nn_bce_loss 
#'                   nn_cross_entropy_loss nn_l1_loss nn_kl_div_loss
choose_loss <- function(name) {
  switch(
    name,
    "mse"=torch::nn_mse_loss(),
    "nll"=torch::nn_nll_loss(),
    "binary_crossentropy"=torch::nn_bce_loss(),
    "crossentropy"=torch::nn_cross_entropy_loss(),
    "L1"=torch::nn_l1_loss(),
    "KL"=torch::nn_kl_div_loss()
  )
}

#' Generate network
#' @param topology Topology parameter of the network
#' @param name Name of the network
#' @param device device of the network
#' @export
#' @importFrom torch nn_module nn_module_list
generate_net <- function(topology,name="defaultnet",device=NULL) {
  net <- nn_module(
    name,
    initialize = function(topo) {
      ml <- list()
      for (i in seq_along(topo)) {
        ml[[i]] <- choose_net(topo[[i]])
      }
      self$ml <- torch::nn_module_list(ml)
    },
    forward = function(x) {
      for (i in seq_along(self$ml)) {
        x <- self$ml[[i]](x)
      }
      x
    },
    device = device
  )
  model <- net(topology)
  if (!is.null(device)) {
    model$to(device=device)
  }
  model
}

#' Do training of the neural network
#' @param dl A dataloader of the training data
#' @param val_dl A dataloader of the validation data
#' @param topology Topology of the network
#' @param optim Name of the optimizer
#' @param loss Name of the loss function
#' @param nepoch Number of epochs
#' @param lr The learning rate
#' @param save_model If TRUE, save the models obtained at each epoch
#' @param save_filename Template of the saved files. %d is replaced with the epoch number
#' @param check Check the dimension of the parameter and network, type of the data, etc.
#' @param verbose Output the loss during the training
#' @param device torch_device for training
#' @returns A list of the trained model, the training loss and the validation loss
#' @export
train <- function(dl,val_dl=NULL,topology,optim,loss,nepoch,lr=0.01,
                  save_model=FALSE,save_filename="model%d.torch",
                  check=TRUE,verbose=TRUE,device=NULL) {
  model <- generate_net(topology,device=device)
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
      cat("train_loss=",mean(bloss)," ")
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
        vloss <- c(bloss,L$item())
        if (verbose)
          cat(".")
      }
      valid_loss <- c(valid_loss,mean(vloss))
      if (verbose)
        cat("valid_loss=",mean(vloss))
    }
    if (verbose)
      cat("\n")
    if (save_model) {
      torch::torch_save(model,sprintf(save_filename,epoch))
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
