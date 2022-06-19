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

