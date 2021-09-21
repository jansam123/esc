import numpy as np
import qexpy as q 
import sys
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import container
import os 



#---------------ODSTRANENIE ERRORBAROV Z LEGENDY-------------------
#handles, labels = ax.get_legend_handles_labels()
#handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
#ax.legend(handles, labels) 
#------------------------------------------------------------------