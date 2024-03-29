% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/data.R
\name{prepare_dataloader}
\alias{prepare_dataloader}
\title{Create dataloader from data}
\usage{
prepare_dataloader(
  train_x,
  train_y,
  xtype,
  ytype,
  batchsize = 16,
  train_shuffle = TRUE,
  val_shuffle = FALSE,
  name = "default",
  val_x = NULL,
  val_y = NULL,
  device = NULL
)
}
\arguments{
\item{train_x}{A matrix for training data}

\item{train_y}{A matrix/vector for training label}

\item{xtype}{torch_dtype of x}

\item{ytype}{torch_dtype of y}

\item{batchsize}{The batch size (default: 16)}

\item{train_shuffle}{Shuffle the training data or not (default: TRUE)}

\item{val_shuffle}{Shuffle the validation data or not (default: FALSE)}

\item{name}{Name of the dataloader (optional)}

\item{val_x}{A matrix for validation data (can be NULL)}

\item{val_y}{A matrix/vector for validation label (can be NULL)}

\item{device}{torch_device}
}
\value{
A list of two lists; "dataloader" has two data loaders
"train" and "validate", and "dataset" also has "train" and "validate"
}
\description{
Create dataloader from data
}
