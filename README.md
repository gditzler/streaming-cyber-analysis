# streaming-cyber-analysis

# Reading in the Data

The data for these experiments are publicly available and the code uses versions of the data that are stored on a Microsoft Azure blob. You will need to add the data to your Azure account or modify the code to read the data from a local folder. The tempate for `streamcyber/read_azure.py` should look like 

```
#!/usr/bin/env python 

def read_azure(): 
    subscription_id = '<put id here>'
    resource_group = '<put resource group here>'
    workspace_name = '<put workspace name here>'
    return subscription_id, resource_group, workspace_name
```