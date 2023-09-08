cat ../labeldata3/train_ev.json ../labeldata3/train_inter.json  >  train.json
shuf train.json -o  train_random.json