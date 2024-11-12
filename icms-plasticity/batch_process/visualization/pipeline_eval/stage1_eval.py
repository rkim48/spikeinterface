from pathlib import Path
import time
import matplotlib.pyplot as plt

from batch_process.util.file_util import file_dialog
from util.load_data import load_data
from spikeinterface import full as si
import spikeinterface.preprocessing as sp
from preprocessing.preprocessing_pipelines import new_pipeline3
from batch_process.util.plotting import plot_units_in_batches
import batch_process.util.file_util as file_util
from batch_process.util.curate_util import *
from batch_process.util.misc import *

#%% Evaluate preprocessing of raw data

# Load preproc zarr folder
stage1_path = Path(save_folder) / "stage1"
preproc_zarr_folder = stage1_path / "custom_preproc"
