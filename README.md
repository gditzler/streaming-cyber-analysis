# streaming-cyber-analysis

This code repo implements data stream processing algorithms for high volume cybersecurity datastream. Particularly, this code implements online supervised learning and evaluation of data streams related to intrusion detection systems (IDS). We focus on IDS related to network (NSL-KDD and UNSWNB) and WiFi attacks (private dataset). The latter dataset is a private dataset that will be made publicly available. 


# Working Through the Code 

## Reading in the Data

The data for these experiments are publicly available and the code uses versions of the data that are stored on a Microsoft Azure blob. The data are named `UNSWNB-15_training`, `UNSWNB-15_testing`, `NSLKDD_training`, and `NSLKDD_testing`. 
You will need to add the data to your Azure account or modify the code to read the data from a local folder. The tempate for `streamcyber/read_azure.py` should look like 
```
#!/usr/bin/env python 

def read_azure(): 
    subscription_id = '<put id here>'
    resource_group = '<put resource group here>'
    workspace_name = '<put workspace name here>'
    return subscription_id, resource_group, workspace_name
```

## Running the code.

Simply run:  
```
$ python main.py 
```