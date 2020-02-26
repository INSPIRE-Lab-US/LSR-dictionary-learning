# Online Algorithm Experiment with House image
## Steps to reproduce
- Run the OnlineExperiment/HouseOnline.m function twice once with `Data/rand_state1` and again with `Data/rand_state2`
+ ex.
- `HouseOnline('../Data/rand_state1')`
- `HouseOnline('../Data/rand_state2')`

We split up the monte carlos over two jobs on our server for a total of 30 monte carlos.

After running the function twice (preferably at the same time as 2 jobs) it will save 2 new `.mat` files copy those new `.mat` files to your local machine and run the  `plotsOnline.m` script which will load in the two `.mat` files that were generated and concatenate them together before plotting the result. 
