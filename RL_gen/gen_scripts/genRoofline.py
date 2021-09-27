import sys
import subprocess as sp
import numpy as np
import pandas as pd
from io import StringIO
import os
import re
import shutil
import sqlite3
from IPython.display import display
import matplotlib.pyplot as plt

cudadir = "/usr/common/software/sles15_cgpu/cuda/10.1.243"
homedir = os.path.dirname(os.getcwd())

### For each kernel invocation, parse cycles and cycles/s to calculate the execution time
def parse_time(df_metrics):
    selectkeys = ["ID", "Name"]
    resultkeys = ["ID", "Name"]
    metricdf   = df_metrics.copy()
    profiledf  = pd.DataFrame(columns=selectkeys)
    # get cycles
    metricname = "CUDA Cycles"
    cyclesdf   = metricdf.loc[(metricdf["Metric Name"]=="sm__cycles_elapsed") & (metricdf["Metric Type"]=="total"),
                           selectkeys+["Metric Unit", "Metric Value"]].reset_index(drop=True).sort_values(by=selectkeys).rename(columns={"Metric Value": metricname}).copy()
    # get rates
    metricname = "CUDA Rates"
    ratesdf = metricdf.loc[(metricdf["Metric Name"]=="sm__cycles_elapsed") & (metricdf["Metric Type"]=="rate"),
                           selectkeys+["Metric Unit", "Metric Value"]].reset_index(drop=True).sort_values(by=selectkeys).rename(columns={"Metric Value": metricname}).copy()
    # check consistency
    if not cyclesdf[['ID', 'Name']].equals(ratesdf[['ID', 'Name']]):
        raise ValueError("CUDA Time data not consistent")
    # adjust metric unit
    if(ratesdf.size >0 and cyclesdf.size >0):
        ratesdf.loc[ratesdf["Metric Unit"].str.contains("cycle/nsecond"), ["CUDA Rates"]] *= 1e9
        ratesdf.loc[ratesdf["Metric Unit"].str.contains("cycle/usecond"), ["CUDA Rates"]] *= 1e6
    # manual merge and compute CUDA Time
        cyclesdf["CUDA Rates"] = list(ratesdf["CUDA Rates"])
        cyclesdf["CUDA Time"] = cyclesdf["CUDA Cycles"] / cyclesdf["CUDA Rates"]
    # merge with output
        profiledf = cyclesdf[selectkeys+['CUDA Time']].copy()
    ### Combine
        profiledf['Invocations'] = 1
        profiledf = profiledf.groupby(resultkeys).sum().reset_index()
        profiledf.sort_values(by=resultkeys).reset_index(drop=True, inplace=True)
    return profiledf

### FP32 FLOPs
def parse_fp32(df_metrics):
    selectkeys = ["ID", "Name"]
    resultkeys = ["ID", "Name"]
    metricdf   = df_metrics.copy()
    profiledf  = pd.DataFrame(columns=selectkeys)
    ### FMA FLOPs = number of FMA instructions x 2
    metricdf.loc[metricdf["Metric Name"].str.contains("ffma"), ["Metric Value"]] *= 2
    metrics = ['sm__sass_thread_inst_executed_op_fadd_pred_on',
               'sm__sass_thread_inst_executed_op_ffma_pred_on',
               'sm__sass_thread_inst_executed_op_fmul_pred_on']
    tmpdf = metricdf.loc[ metricdf["Metric Name"].isin(metrics), resultkeys+["Metric Value"] ].copy()
    tmpdf = tmpdf.groupby(resultkeys).sum().reset_index().rename(columns={"Metric Value": "FP32 FLOPs"})
    if (tmpdf.size >0):
        profiledf = tmpdf[resultkeys+["FP32 FLOPs"]]
        profiledf.sort_values(by=resultkeys).reset_index(drop=True, inplace=True)
    del metricdf['ID']
    return profiledf

### FP16 FLOPs
def parse_fp16(df_metrics):
    selectkeys = ["ID", "Name"]
    resultkeys = ["ID", "Name"]
    metricdf   = df_metrics.copy()
    profiledf  = pd.DataFrame(columns=selectkeys)
    ### FMA FLOPs = number of FMA instructions x 2
    metricdf.loc[metricdf["Metric Name"].str.contains("hfma"), ["Metric Value"]] *= 2
    metrics = ['sm__sass_thread_inst_executed_op_hadd_pred_on',
               'sm__sass_thread_inst_executed_op_hfma_pred_on',
               'sm__sass_thread_inst_executed_op_hmul_pred_on']
    tmpdf = metricdf.loc[ metricdf["Metric Name"].isin(metrics), resultkeys+["Metric Value"] ].copy()
    tmpdf = tmpdf.groupby(resultkeys).sum().reset_index().rename(columns={"Metric Value": "FP16 FLOPs"})
    if (tmpdf.size >0):
        profiledf = tmpdf[resultkeys+["FP16 FLOPs"]]
        profiledf.sort_values(by=resultkeys).reset_index(drop=True, inplace=True)
    del metricdf['ID']
    return profiledf

### TC FLOPs
def parse_tc(df_metrics):
    selectkeys = ["ID", "Name"]
    resultkeys = ["ID", "Name"]
    metricdf   = df_metrics.copy()
    profiledf  = pd.DataFrame(columns=selectkeys)
    tmpdf = metricdf.loc[ metricdf["Metric Name"].str.contains("sm__inst_executed_pipe_tensor"), resultkeys+["Metric Value"] ].copy()
    tmpdf = tmpdf.groupby(resultkeys).sum().reset_index().rename(columns={"Metric Value": "TC FLOPs"})
    tmpdf["TC FLOPs"] = 512 * tmpdf["TC FLOPs"]
    if (tmpdf.size >0):
        profiledf = tmpdf[resultkeys+["TC FLOPs"]]
        profiledf.sort_values(by=resultkeys).reset_index(drop=True, inplace=True)
    del metricdf['ID']
    return profiledf


def parse_dram(df_metrics):
    selectkeys = ["ID", "Name"]
    resultkeys = ["ID", "Name"]
    metricdf   = df_metrics.copy()
    profiledf  = pd.DataFrame(columns=selectkeys)
    profiledf  = profiledf.fillna(0.)
    if (metricdf.size >0):
        metricdf.loc[metricdf["Metric Unit"].str.contains("Gbyte"), ["Metric Value"]] *= 1e9
        metricdf.loc[metricdf["Metric Unit"].str.contains("Mbyte"), ["Metric Value"]] *= 1e6
        metricdf.loc[metricdf["Metric Unit"].str.contains("Kbyte"), ["Metric Value"]] *= 1e3

    #project out
    dramdf = metricdf.loc[metricdf["Metric Name"].str.contains("dram__bytes"), resultkeys+["Metric Value"] ].copy()
    dramdf = dramdf.groupby(resultkeys).sum().reset_index().rename(columns={"Metric Value": "DRAM Bytes"})
    # merge
    if (dramdf.size >0):
        profiledf = dramdf[resultkeys+["DRAM Bytes"]]
        profiledf.sort_values(by=resultkeys).reset_index(drop=True, inplace=True)
    del metricdf['ID']
    return profiledf

def import_nsight_metric(filename, cuda_dir='/usr/local/cuda'):
    args = [os.path.join(cuda_dir, "bin/nv-nsight-cu-cli"),"--csv","-i",filename]
    #open subprocess and communicate
    p = sp.Popen(args, stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = p.communicate()
    #get timeline from csv
    profiledf = pd.read_csv(StringIO(stdout.decode("utf-8")),skiprows=0) #.dropna(how="all").rename(columns={"Kernel": "Name"})
    #clean up
    del profiledf["Process ID"]
    del profiledf["Process Name"]
    del profiledf["Host Name"]
    del profiledf["Kernel Time"]
    del profiledf["Context"]
    del profiledf["Section Name"]
    profiledf.rename(columns={"Kernel Name": "Name"}, inplace=True)
    return profiledf

try:
    directory_name=sys.argv[1]
    profile_dir = sys.argv[1]
except:
    profile_dir = "./profilingData"
print("Generate roofline analysis for data in", profile_dir)

outputdir = ["."]
datadirs = []
for file in os.listdir(profile_dir):
 datadir = os.path.join(profile_dir, file)
 if os.path.isdir(datadir):
    datadirs += [datadir]

#combination of markers and colors (8x3=24 for now)
color_list=  ['r', 'g', 'b']
marker_list= ['o', 'v', '*', 's', 'p', '*', 'h', 'd']
plt.figure()
dirCnt=0
for datadir in datadirs:
 #datadir = os.path.join(rootdir, file)
 if os.path.isdir(datadir):
  #get all the files
  files = []
  for d in os.listdir(datadir):
    files += [ os.path.join(datadir,x) for x in os.listdir(datadir) if ((os.path.splitext(x)[-1] == ".ncu-rep"))]

  #recs
  records = []
  #build feature list:
  for path in files:
    file = os.path.basename(path)
    #path
    path = os.path.dirname(path)
    #splitup
    splt = file.split(".")
    prefix = ".".join(splt[0:-1])
    #append to records
    records.append({"prefix": prefix, "file": os.path.join(path, file)})
#put in df
  recorddf = pd.DataFrame(records).sort_values(["prefix"])
  resultkeys          = ["ID", "Name"]
  profiledf_time      = pd.DataFrame(columns=resultkeys)
  profiledf_fp16      = pd.DataFrame(columns=resultkeys)
  profiledf_fp32      = pd.DataFrame(columns=resultkeys)
  profiledf_tc        = pd.DataFrame(columns=resultkeys)
  profiled_allFP16    = pd.DataFrame(columns=resultkeys)
  profiled_allFP32    = pd.DataFrame(columns=resultkeys)
  profiled_allFPTC    = pd.DataFrame(columns=resultkeys)
  profiledf_DRAM      = pd.DataFrame(columns=resultkeys)
  profiled_allDRAM    = pd.DataFrame(columns=resultkeys)
  aggregatedKernelName   = ""

  for pref in recorddf["prefix"]:
    file = os.path.basename(path)
    #set empty lists
    df_times = []
    df_timeline = []
    df_summary = []
    df_metrics = []

    #project frame
    files = recorddf.loc[ recorddf["prefix"] == pref, "file" ].values
    #project the invididual files
    metricfile = [x for x in files if x.endswith(".ncu-rep")][0]
    splt= pref.split(".")
    kernelName= splt[1]
            
    #metrics
    #open subprocess and communicate
    metricdf = import_nsight_metric(metricfile, cuda_dir=cudadir)

    #fuse read/write metrics together:
    unique_metrics = metricdf["Metric Name"].unique()
        
    unique_metrics = set([x.replace(".sum","").replace(".per_second","").replace(".avg","").replace("_write","").replace("_read","").replace("_ld","").replace("_st","") for x in unique_metrics])
    unique_metrics = set([x.replace(".sum","").replace(".per_second","").replace(".avg","") for x in unique_metrics])
    unique_units = metricdf["Metric Unit"].unique()
    #add the metric type
    metricdf["Metric Type"] = "total"
    #read
    metricdf.loc[ metricdf[ "Metric Name" ].str.contains("_read"), "Metric Type" ] = "read"
    metricdf.loc[ metricdf[ "Metric Name" ].str.contains("_ld"), "Metric Type" ] = "read"
    #write
    metricdf.loc[ metricdf[ "Metric Name" ].str.contains("_write"), "Metric Type" ] = "write"
    metricdf.loc[ metricdf[ "Metric Name" ].str.contains("_st"), "Metric Type" ] = "write"
    #rate
    metricdf.loc[ metricdf[ "Metric Name" ].str.contains(".per_second"), "Metric Type" ] = "rate"
                
    for metric in unique_metrics:
        metricdf.loc[ metricdf[ "Metric Name"].str.startswith(metric), "Metric Name" ] = metric
                
    #append to DF:
    df_metrics.append(metricdf)
    
    metricdf = pd.concat(df_metrics)
    
        #compute the profile
    parsedTime          = parse_time(metricdf)
    if parsedTime.size>0:
        profiledf_time      =  parsedTime
        aggregatedKernelName  = kernelName[len("kernel"):]

    parsedFP16          =  parse_fp16(metricdf)
    if parsedFP16.size >0 :
        profiledf_fp16  =  parsedFP16

    parsedFP32          =  parse_fp32(metricdf)
    if parsedFP32.size >0 :
        profiledf_fp32       =  parsedFP32

    parsedTC            = parse_tc(metricdf)
    if parsedTC.size >0 :
        profiledf_tc        =  parsedTC

    parsedDRAM          =  parse_dram(metricdf)
    if parsedDRAM.size >0 :
        profiledf_DRAM  =  parsedDRAM
        aggregatedKernelName = kernelName[len("kernel"):]
  print("parsing file", aggregatedKernelName)
  profiled_allFP32 = profiledf_time
  profiled_allFP32 = profiled_allFP32.merge(profiledf_fp16[resultkeys+["FP16 FLOPs"]], on=resultkeys, how="inner")
  profiled_allFP32 = profiled_allFP32.merge(profiledf_fp32[resultkeys+["FP32 FLOPs"]], on=resultkeys, how="inner")
  profiled_allFP32 = profiled_allFP32.merge(profiledf_tc[resultkeys+["TC FLOPs"]], on=resultkeys, how="inner")
  profiled_allFP32 = profiled_allFP32.merge(profiledf_DRAM[resultkeys+["DRAM Bytes"]], on=resultkeys, how="inner")

  profiled_allFP32["FP32/s"]   = profiled_allFP32["FP32 FLOPs"] / profiled_allFP32["CUDA Time"]
  profiled_allFP16["FP16/s"]   = (profiled_allFP32["FP16 FLOPs"]+profiled_allFP32["TC FLOPs"]) / profiled_allFP32["CUDA Time"]
  profiled_allFP32["FP32/B"]   = profiled_allFP32["FP32 FLOPs"] / profiled_allFP32["DRAM Bytes"]
  profiled_allFP32["FP16/B"]   = (profiled_allFP32["FP16 FLOPs"]+profiled_allFP32["TC FLOPs"]) / profiled_allFP32["DRAM Bytes"]

  AI_FP32= profiled_allFP32["FP32/B"]
  AI_FP16= profiled_allFP32["FP16/B"]
  FP32RateCol=  profiled_allFP32["FP32/s"]
  FP16RateCol=  profiled_allFP16["FP16/s"]

  #remove the underscore
  myColor=  color_list[dirCnt % len(color_list)]
  myMarker= marker_list[dirCnt // len(color_list)]
  lbl= aggregatedKernelName[1:]
  lbl1= lbl+"(FP32)"
  lbl2= lbl+"(FP16)"
  if(np.count_nonzero(FP32RateCol) > 0):  
     roofline= plt.scatter(AI_FP32, FP32RateCol, marker=myMarker, color=myColor, label=lbl1, s=15)
  if(np.count_nonzero(FP16RateCol) > 0):
     roofline1= plt.scatter(AI_FP16, FP16RateCol, marker=myMarker, color=myColor, label=lbl2, s=3)
  plt.legend(numpoints=1, loc='upper left', prop={'size': 4})
  dirCnt = dirCnt+1


peakFP32 = 15e12
peakFP16 = 125e12
membw = 900e9
ai = np.geomspace(1e-3, 1e4, 1000)
roofline_flopFP16 = np.minimum(ai * membw, peakFP16)
roofline_flopFP32 = np.minimum(ai * membw, peakFP32)

plt.plot(ai, roofline_flopFP16, color='black',linestyle='-',linewidth=2.0)  
plt.plot(ai, roofline_flopFP32, color='black',linestyle='-',linewidth=2.0)  
plt.xlabel("Arithmetic Intensity (FLOP/Byte)")
plt.ylabel("Flop Rate (FLOP/s)")
plt.yscale('log')
plt.xscale('log')
#plt.show()
foldername = os.path.basename(os.getcwd())
output= foldername+"_roofline.pdf"
plt.savefig(output)
