dl <- prepare_dataloader(
  train_x = matrix(as.integer(runif(100*10)*10)+1,nrow=100,ncol=10),
  train_y = runif(100),
  xtype = torch_long(),
  ytype = torch_float32(),
)

topology <-list(
  # B x L
  list("embedding",num_embeddings=10,embedding_dim=20), # B x L x 20
  list("transpose",1,2), # L x B x 20
  list("lstm",input_size=20,hidden_size=15,output_type="all"), # L x B x 15
  list("transpose",1,2), # B x L x 15
  list("transpose",2,3), # B x 15 x L
  list("conv1d",in_channel=15,out_channels=1,kernel_size=3), # B x 1 x 15-2
  list("flatten",start_dim=2),
  list("tanh"),
  list("linear",in_features=8,1)) 
dim_check(get_consistency(topology),verbose=TRUE)

res <- train(
  dl$dataloader$train,
  topology=topology,
  optim="adam",
  loss="mse",
  nepoch=10)

