
#' @importFrom torch optim_adadelta optim_adagrad optim_adam optim_asgd optim_lbfgs optim_rmsprop optim_sgd
choose_optim <- function(optim_name,params,...) {
  opt <- switch(
    optim_name,
    "adadelta"=torch::optim_adadelta,
    "adagrad"=torch::optim_adagrad,
    "adam"=torch::optim_adam,
    "asgd"=torch::optim_asgd,
    "lbfgs"=torch::optim_lbfgs,
    "rmsprop"=torch::optim_rmsprop,
    "sgd"=torch::optim_sgd
  )
  opt(params,...)
}

transpose <- torch::nn_module(
  "transpose",
  initialize = function(dim0,dim1) {
    self$i <- dim0
    self$j <- dim1
  },
  forward = function(x) {
    torch_transpose(x,self$i,self$j)
  }
)

xlstm <- torch::nn_module(
  "xlstm",
  initialize = function(input_size, hidden_size, output_type="last",...) {
    self$lstm <- torch::nn_lstm(input_size,hidden_size,...)
    self$output_last <- ifelse(output_type == "last",TRUE,FALSE)
  },
  forward = function(x) {
    y <- self$lstm(x)
    if (self$output_last) {
      seqlen <- dim(y)[1]
      y[[1]][seqlen,,]
    } else {
      y[[1]]
    }
  }
)

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
    "lstm"=xlstm,
    "relu"=torch::nn_relu,
    "sigmoid"=torch::nn_sigmoid,
    "tanh"=torch::nn_tanh,
    "softmax"=torch::nn_softmax,
    "flatten"=torch::nn_flatten,
    "unflutten"=torch::nn_unflatten,
    "embedding"=torch::nn_embedding,
    "transpose"=transpose,
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
  net <- torch::nn_module(
    name,
    initialize = function(topo) {
      ml <- list()
      for (i in seq_along(topo)) {
        ml[[i]] <- choose_net(topo[[i]])
      }
      self$ml <- torch::nn_module_list(ml)
      self$topology <- topo
    },
    forward = function(x) {
      for (i in seq_along(self$ml)) {
        layer <- self$ml[[i]]
        x <- layer(x)
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
