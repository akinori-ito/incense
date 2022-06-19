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

