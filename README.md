## Dyn-Backdoor
This is a Pytorch implementation of the paper: Dyn-Backdoor: Backdoor Attack on Dynamic Link Prediction. 
#Requirements
Install PyTorch following the instuctions on the [official website] (https://pytorch.org/). The code has been tested over PyTorch 1.7.0 versions.

Then install the other dependencies.
```
pip install -r requirements.txt
```
# Test run
Unzip the dataset file
```
unzip dataset.zip
```

and run

```
python D_B_G.py
```

The default parameters are not the best performing-hyper-parameters used to reproduce our results in the paper. Hyper-parameters need to be specified through the commandline arguments. Please refer to our paper for the details of how we set the hyper-parameters. 
