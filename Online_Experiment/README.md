# Online Algorithm Experiment with House image
## Steps to reproduce
- Run the HouseOnline.m function twice once with rand_state1 and again with rand_state2
ex.
`HouseOnline('../Data/rand_state1')`
`HouseOnline('../Data/rand_state2')`

We split up the monte carlos over two jobs on our server for a total of 30 monte carlos.

After running the function twice run plotsOnline.m which will load in the two mat files that were generated and concatenate them together before plotting the result. 
