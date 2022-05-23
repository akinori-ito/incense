<!-- toc -->

5月 23, 2022

# DESCRIPTION

```
Package: simpleTorch
Title: What the Package Does (One Line, Title Case)
Version: 0.0.0.9000
Authors@R: 
    person("Akinori", "Ito", , "aito@spcom.ecei.tohoku.ac.jp", 
    role = c("aut", "cre"))
Description: A simple interface to torch for developing simple NN.
License: MIT + file LICENSE
Encoding: UTF-8
Roxygen: list(markdown = TRUE)
RoxygenNote: 7.1.2
Imports: torch
```


# `create_dataset`

Create dataset from data


## Description

Create dataset from data


## Usage

```r
create_dataset(
  name,
  train_x,
  train_y,
  val_x = NULL,
  val_y = NULL,
  xtype,
  ytype,
  device
)
```


## Arguments

Argument      |Description
------------- |----------------
`name`     |     Name of the dataset
`train_x`     |     A matrix for training data
`train_y`     |     A matrix/vector for training label
`val_x`     |     A matrix for validation data (can be NULL)
`val_y`     |     A matrix/vector for validation label (can be NULL)
`xtype`     |     torch_dtype of x
`ytype`     |     torch_dtype of y
`device`     |     torch_device


## Value

A list of two datasets, "training" and "validation"
 (validation is NULL if val_x or val_y is not given)


# `generate_net`

Generate network


## Description

Generate network


## Usage

```r
generate_net(topology, name = "defaultnet", device = NULL)
```


## Arguments

Argument      |Description
------------- |----------------
`topology`     |     Topology parameter of the network
`name`     |     Name of the network
`device`     |     device of the network


# `prepare_dataloader`

Create dataloader from data


## Description

Create dataloader from data


## Usage

```r
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
```


## Arguments

Argument      |Description
------------- |----------------
`train_x`     |     A matrix for training data
`train_y`     |     A matrix/vector for training label
`xtype`     |     torch_dtype of x
`ytype`     |     torch_dtype of y
`batchsize`     |     The batch size (default: 16)
`train_shuffle`     |     Shuffle the training data or not (default: TRUE)
`val_shuffle`     |     Shuffle the validation data or not (default: FALSE)
`name`     |     Name of the dataloader (optional)
`val_x`     |     A matrix for validation data (can be NULL)
`val_y`     |     A matrix/vector for validation label (can be NULL)
`device`     |     torch_device


## Value

A list of two lists; "dataloader" has two data loaders
 "train" and "validate", and "dataset" also has "train" and "validate"


