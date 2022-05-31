# library(R6)
# 
# topology <- list(
#   list("conv2d",in_channel=3,out_channel=10,kernel_size=3),
#   list("relu"),
#   list("maxpool2d",kernel_size=2),
#   list("flatten",start_dim=2),
#   list("linear",in_feature=2250,out_feature=3),
#   list("softmax",dim=2)
# )


get_value <- function(x,pos,name,default=NULL) {
  v <- NULL
  if (is.null(x$name)) {
    if (pos <= length(x)) {
      v <- x[[pos]]
    }
  } else {
    v <- x$name
  }
  if (is.null(v)) {
    v <- default
  }
  if (is.null(v)) {
    stop("Value for",name,"is NULL")
  }
  v
}

#' @importFrom R6 R6Class
CheckerList <- R6::R6Class(
  classname = "CheckerList",
  public = list(
    checker = NULL,
    initialize = function(...) {
      self$checker <- list(...)
    },
    eval = function(indim) {
      ndim <- rep(0,length(self$checker))
      for (j in seq_along(self$checker)) {
        ndim[j] <- self$checker[[j]]$outsize(indim[j])
      }
      ndim
    },
    append = function(checker) {
      self$checker <- base::append(self$checker,checker)
    },
    length = function() {
      length(self$checker)
    }
  )
)

IdentityChecker <- R6::R6Class(
  classname = "Identity",
  public = list(
    initialize = function() {},
    outsize = function(insize) {insize}
  )
)

ConstChecker <- R6::R6Class(
  classname = "Const",
  public = list(
    const = NULL,
    initialize = function(x) {
      self$const <- x
    },
    outsize = function(insize) {self$const}
  )
)

ConvolutionChecker <- R6::R6Class(
  classname = "ConvolutionChecker",
  public = list(
    padding = 0,
    dilation = 1,
    kernel_size = 1,
    stride = 1,
    initialize = function(padding,dilation,kernel_size, stride) {
      self$padding <- padding
      self$dilation <- dilation
      self$kernel_size <- kernel_size
      self$stride <- stride
    },
    outsize = function(insize) {
      floor((insize+2*self$padding-self$dilation*(self$kernel_size-1)-1)/self$stride+1)
    }
  )
)

UnConvolutionChecker <- R6::R6Class(
  classname = "UnConvolutionChecker",
  public = list(
    padding = 0,
    kernel_size = 1,
    stride = 1,
    initialize = function(padding,kernel_size, stride) {
      self$padding <- padding
      self$kernel_size <- kernel_size
      self$stride <- stride
    },
    outsize = function(insize) {
      floor((insize-1)*self$stride-2*self$padding+self$kernel_size)
    }
  )
)

FlattenCheckerList <- R6::R6Class(
  classname = "CheckerList",
  public = list(
    start_dim = NULL,
    end_dim = NULL,
    ldim = NULL,
    initialize = function(ldim, start_dim, end_dim) {
      self$ldim <- ldim
      self$start_dim <- start_dim
      self$end_dim <- end_dim
    },
    eval = function(indim) {
      d <- length(indim)
      if (d != self$ldim) {
        stop(paste("Input dimension",d,"does not match the model dimension",self$ldim))
      }
      start_dim <- self$start_dim
      if (self$end_dim == -1) 
        end_dim <- d
      else
        end_dim <- self$end_dim
      sdim <- 1
      outdim <- c()
      for (i in seq_along(indim)) {
        # cat("i=",i,"indim[i]=",indim[i],
        #     "start_dim=",start_dim,
        #     "end_dim=",end_dim,
        #     "outdim=",outdim,"sdim=",sdim,"\n")
        if (i < start_dim) {
          outdim <- c(outdim,indim[i])
        } else if (i <= end_dim) {
          sdim <- sdim*indim[i]
        } else {
          if (!is.null(sdim)) {
            outdim <- c(outdim,sdim)
            sdim <- NULL
          }
          outdim <- c(outdim,indim[i])
        }
      }
      if (!is.null(sdim))
        outdim <- c(outdim,sdim)
      outdim
    },
    length = function() {
      if (self$end_dim == -1) 
        end_dim <- length(d)
      else
        end_dim <- self$end_dim
      self$ldim-(end_dim-self$start_dim)      
    }
  )
)


get_consistency <- function(topo) {
  chooseval <- function(x,n) {ifelse(length(x) < n, x[1], x[n])}
  dimlist <- vector(mode="list",length=length(topo))
  for (layer in seq_along(topo)) {
    l <- topo[[layer]]
    llen <- length(l)
    module_name <- l[[1]]
    if (module_name == "linear") {
      in_dim <- get_value(l,2,"in_features")
      out_dim <- get_value(l,3,"out_features")
      dimlist[[layer]] <- list(
        indim=c(NA,in_dim), 
        outdim=CheckerList$new(IdentityChecker$new(),ConstChecker$new(out_dim)))
    } else if (module_name == "conv1d") {
      in_channel <- get_value(l,2,"in_channels")
      out_channel <- get_value(l,3,"out_channels")
      ksize <- get_value(l,4,"kernel_size")
      stride <- get_value(l,5,"stride",1)
      padding <- get_value(l,6,"padding",0)
      dilation <- get_value(l,7,"dilation",1)
      check <- ConvolutionChecker$new(padding,dilation,ksize, stride)
      dimlist[[layer]]$indim <- c(NA,in_channel,NA)
      dimlist[[layer]]$outdim <- CheckerList$new(
        IdentityChecker$new(),
        ConstChecker$new(out_channel),
        check)
    } else if (module_name == "conv2d") {
      in_channel <- get_value(l,2,"in_channels")
      out_channel <- get_value(l,3,"out_channels")
      ksize <- get_value(l,4,"kernel_size")
      stride <- get_value(l,5,"stride",1)
      padding <- get_value(l,6,"padding",0)
      dilation <- get_value(l,7,"dilation",1)
      check1 <- ConvolutionChecker$new(
        chooseval(padding,1),
        chooseval(dilation,1),
        chooseval(ksize,1),
        chooseval(stride,1)
      )
      check2 <- ConvolutionChecker$new(
        chooseval(padding,2),
        chooseval(dilation,2),
        chooseval(ksize,2),
        chooseval(stride,2)
      )
      dimlist[[layer]]$indim <- c(NA,in_channel,NA,NA)
      dimlist[[layer]]$outdim <- CheckerList$new(IdentityChecker$new(),
                                      ConstChecker$new(out_channel),
                                      check1,check2)
    } else if (module_name == "maxpool1d") {
      if (layer == 1) {
        stop("Do not use maxpool at the first layer")
      }
      ksize <- get_value(l,2,"kernel_size")
      stride <- get_value(l,3,"stride",ksize)
      padding <- get_value(l,4,"padding",0)
      dilation <- get_value(l,5,"dilation",1)
      check <- ConvolutionChecker$new(padding,dilation,ksize, stride)
      dimlist[[layer]]$indim <- c(NA,NA,NA)
      dimlist[[layer]]$outdim <- CheckerList$new(IdentityChecker$new(),
                                      ConstChecker$new(out_channel),
                                      check)
    } else if (module_name == "maxpool2d") {
      if (layer == 1) {
        stop("Do not use maxpool at the first layer")
      }
      ksize <- get_value(l,2,"kernel_size")
      stride <- get_value(l,3,"stride",ksize)
      padding <- get_value(l,4,"padding",0)
      dilation <- get_value(l,5,"dilation",1)
      check1 <- ConvolutionChecker$new(
        chooseval(padding,1),
        chooseval(dilation,1),
        chooseval(ksize,1),
        chooseval(stride,1)
      )
      check2 <- ConvolutionChecker$new(
        chooseval(padding,2),
        chooseval(dilation,2),
        chooseval(ksize,2),
        chooseval(stride,2)
      )
      dimlist[[layer]]$indim <- c(NA,NA,NA,NA)
      dimlist[[layer]]$outdim <- CheckerList$new(IdentityChecker$new(),
                                      ConstChecker$new(out_channel),
                                      check1,check2)
    } else if (module_name == "maxunpool1d") {
      if (layer == 1) {
        stop("Do not use maxunpool at the first layer")
      }
      ksize <- get_value(l,2,"kernel_size")
      stride <- get_value(l,3,"stride",ksize)
      padding <- get_value(l,4,"padding",0)
      check <- UnConvolutionChecker$new(padding,ksize, stride)
      dimlist[[layer]]$indim <- dim_check(dimlist,to=layer-1)
      dimlist[[layer]]$outdim <- CheckerList$new(IdentityChecker$new(),
                                      ConstChecker$new(out_channel),
                                      check)
    } else if (module_name == "maxunpool2d") {
      if (layer == 1) {
        stop("Do not use maxpool at the first layer")
      }
      ksize <- get_value(l,2,"kernel_size")
      stride <- get_value(l,3,"stride",ksize)
      padding <- get_value(l,4,"padding",0)
      check1 <- UnConvolutionChecker$new(
        chooseval(padding,1),
        chooseval(ksize,1),
        chooseval(stride,1)
      )
      check2 <- UnConvolutionChecker$new(
        chooseval(padding,2),
        chooseval(ksize,2),
        chooseval(stride,2)
      )
      dimlist[[layer]]$indim <- dim_check(dimlist,to=layer-1)
      dimlist[[layer]]$outdim <- CheckerList$new(IdentityChecker$new(),
                                      ConstChecker$new(out_channel),
                                      check1,check2)
    } else if (module_name %in% c("dropout","dropout2d","relu","sigmoid","tanh",
                                  "softmax","layernorm","batchnorm1d","batchnorm2d")) {
      if (layer == 1) {
        stop("No input dimension specified at the first layer")
      }
      d <- dimlist[[layer-1]]$outdim
      dimlist[[layer]]$indim <- rep(NA,d$length())
      dimlist[[layer]]$outdim <- CheckerList$new()
      for (i in seq_len(d$length())) {
        dimlist[[layer]]$outdim$append(IdentityChecker$new())
      }
    } else if (module_name == "flatten") {
      start_dim <- get_value(l,2,"start_dim",1)
      end_dim <- get_value(l,3,"end_dim",-1)
      if (layer == 1) {
        stop("Please do not use flatten at the first layer")
      }
      indim <- dim_check(dimlist,to=layer-1)
      dimlist[[layer]]$indim <- indim
      dimlist[[layer]]$outdim <- FlattenCheckerList$new(length(indim),start_dim,end_dim)
    } else {
      stop(paste("Module",module_name,"not supported"))
    }
    dimlist[[layer]]$name <- module_name
  }
  dimlist
}

dim_unify <- function(x,y) {
  if (length(x) != length(y)) {
    return(FALSE)
  }
  for (i in seq_along(x)) {
    if (is.na(x[i]) || is.na(y[i])) next
    if (x[i] != y[i]) return(FALSE)
  }
  return(TRUE)
}

dim_check <- function(consistency, dimension=NULL, 
                      from=1, to=Inf, verbose=FALSE) {
  if (is.null(dimension)) dimension <- consistency[[from]]$indim
  for (i in from:min(to,length(consistency))) {
    layer <- consistency[[i]]
    if (!dim_unify(dimension,layer$indim)) {
      stop("Dimension inconsistency: net=",
           paste(layer$indim,collapse=","),
           " check=",
           paste(dimension,collapse=","))
    }
    ndim <- layer$outdim$eval(dimension)
    if (verbose) {
      cat("Layer",i,"(",layer$name,"):",dimension,"-> ")
      cat(ndim,"\n")
    }
    dimension <- ndim
  }
  dimension
}

losserror <- function(errtype,loss,out,ref) {
  if (errtype == "dim") {
    outdim <- dim(out)
    refdim <- dim(ref)
    stop(paste("Incorrect dimension for loss calculation:",
               "loss=",loss,
               "output dimension=",outdim,
               "target dimension=",refdim))
  } else if (errtype == "type") {
    stop(paste("Incorrect type for loss calculation:",
               "loss=",loss,
               "output type=",out$dtype,
               "target type=",ref$dtype))
  } else if (errtype == "value") {
    stop(paste("Incorrect reference value for loss calculation:",
               "loss=",loss,
               "target values=",as.array(ref$cpu())))
    
  }
}

#' @importFrom torch torch_long
loss_check <- function(loss,out,ref) {
  outdim <- dim(out)
  refdim <- dim(ref)
  if (loss %in% c("mse","L1","KL","binary_crossentropy")) {
    if (any(outdim != refdim)) 
      losserror("dim",loss,out,ref)
  } else if (loss %in% c("nll","crossentropy")) {
    if (length(outdim) != length(refdim)+1) 
      losserror("dim",loss,out,ref)
    ncat <- outdim[2]
    if (ref$dtype != torch::torch_long()) {
      losserror("type",loss,out,ref)
    }
    aref <- as.array(ref$cpu())
    if (any(aref < 1 | ncat < aref)) {
      losserror("value",loss,out,ref)
    }
  }
}