!/bin/bash
cd /ldaphome/gonthier/wildcat/
source /cal/softs/anaconda/anaconda3/bin/activate PyTorch
/cal/softs/anaconda/anaconda3/bin/python -u -m wildcat.ClusterRun ../data/ >> WILDCAT_results.txt


