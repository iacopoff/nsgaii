import glob
import os
import pandas as pd
import netCDF4 as nc
import numpy as np
from shutil import copyfile,rmtree
import subprocess



#     # routing method = ['KWTroutedRunoff', 'IRFroutedRunoff']
# routing_method = ['KWTroutedRunoff']
# routing_f="/projects/home/iff/mizuRoute-v1.0/NCAR-mizuRoute-21609e0/route"
# VIC_folder="/projects/mas1261/wp3/VIC/ba_bsn_025/run_190704_rmse"
# startTime="2004-05-01"
# endTime="2004-07-31"
# script_dir="/projects/home/iff/Dmoss/vic_calibration/research/vic_calibration"
# SEGID = [138] # should be Cungson station
# output_routing_nc = "/projects/mas1261/wp3/VIC/ba_bsn_025/run_190704_rmse/q_out_new_routing.nc"



def getNetCDFData(fn, varname):
    """Read <varname> variables from NetCDF <fn> """
    f = nc.Dataset(fn,'r')
    f.set_auto_mask(False)
    data = f.variables[varname][:]
    f.close()
    return data


def modify_mizuRoute_control(start_date,end_date,file, outfile=None):
    # overwrite (because 'w' mode) the file with our modified lines
    if not outfile:
        outfile = file
    with open(file, 'r') as oldfile:
        filedata = oldfile.read().splitlines(True)
    # overwrite (because 'w' mode) the file with our modified lines
    with open(outfile, 'w', newline='\r\n') as newfile:
        for line in filedata:
            if line[:12] == '<sim_start> ': newfile.write('<sim_start> ' + start_date + '   !\n')
            elif line[:10] == '<sim_end> ': newfile.write('<sim_end> ' + end_date + '   !\n')
            else:
                newfile.write(line)
            with open(file, 'r') as oldfile:
                filedata = oldfile.read().splitlines(True)


def run_routing(VIC_folder, routing_f, SEGID,script_dir,startTime, endTime,routing_method):


    # output data formatting
    try:
        fluxes_output_name = glob.glob(VIC_folder + "/fluxes*")[0]
    except:
        print("file not loaded")

    try:
        runoff_extraction_Command = f"{script_dir}/runoff_extraction.sh {fluxes_output_name} {routing_f} {startTime} {endTime}"
        subprocess.check_output(runoff_extraction_Command,shell=True)
    except subprocess.CalledProcessError:
        print("runoff extraction process failed")
    # calc_hru_command="python calc_hru_wgtavg_nc.py "+runoff_preprocessing_folder + "runoff_intermediate_mizu.nc " + routing_f + "route/ancillary_data/hru.nc 'runoff' "\
    #                  + routing_f +"route/input/runoff_input_mizu_route.nc"
    # os.system(calc_hru_command)
    try:
        Upscale_ncCommand = script_dir+"/Upscale_nc.sh " + routing_f + " " + script_dir
        print(Upscale_ncCommand)
        subprocess.check_output(Upscale_ncCommand, shell=True)
    except subprocess.CalledProcessError:
        print("Upscale process failed")

    try:
        routingCommand = routing_f + "/route_runoff.exe " + routing_f + "/settings/routing_cal.control"
        subprocess.check_output(routingCommand, shell=True)
    except subprocess.CalledProcessError:
        print("routing process failed")

    output_routing_nc = routing_f + "/output/q_out_cal.nc"
    copyfile(output_routing_nc,VIC_folder+"/out_routing.nc")
    # constants from routing_model
    reachID = getNetCDFData(output_routing_nc, 'reachID')  # Get data value from hru netCDF
    # the output is a dictionary with columns shape equal to seg_ID and row shape equal to time
    ro = dict()
    simTimes = pd.date_range(startTime, endTime, freq='D')
    simSeries_pd = pd.DataFrame(index=simTimes)
    for v in routing_method:
        ro[v] = getNetCDFData(output_routing_nc, v)
        for i in SEGID:
            idx = np.where(reachID == i)
            value = ro[v][:, idx[0]]
            simSeries_pd[i] = value

    simSeries_pd.to_csv(VIC_folder + "/sim_discharge.csv")
    return simSeries_pd





def run_routing_dev(VIC_folder, routing_f, SEGID,startTime, endTime,routing_method):
    fluxes_output_name = glob.glob(VIC_folder + "/fluxes*")[0]
    runoff_file=f"{routing_f}/input/runoff.nc"
    #clean output folder
    for dirpath, dirnames, filenames in os.walk(routing_f+ "/output"):
        for file in filenames:
            os.remove(dirpath + "/" + file)

    #sun baseflow and runoff from VIC, runoff_file is the routing input
    try:
        subprocess.check_output('cdo expr,"runoff=OUT_RUNOFF+OUT_BASEFLOW"'+' ' + fluxes_output_name + ' '+ runoff_file[:-3]+"_1.nc",shell=True)

        subprocess.check_output('cdo -O setcalendar,gregorian '  + runoff_file[:-3]+"_1.nc" + ' '+ runoff_file,shell=True )
    except:
        print("cdo failed")
    #remember some path in the control file are HARDCODED you have to change those if you have moved folders

    control_file=glob.glob(f"{routing_f}/settings/*.control")[0]

    modify_mizuRoute_control(startTime, endTime, control_file, outfile=None)
    try:
        subprocess.check_output(f"{routing_f}/mizuroute.exe {control_file}",shell=True)
    except:
        print("routing failed")
    #merging output
    try:
        subprocess.check_output(f"cdo -O mergetime {routing_f}/output/out* {VIC_folder}/out_routing.nc", shell=True)
    except:
        print("cdo merging failed")

    output_routing_nc=f"{VIC_folder}/out_routing.nc"
    reachID = getNetCDFData(output_routing_nc, 'reachID')  # Get data value from hru netCDF
    # the output is a dictionary with columns shape equal to seg_ID and row shape equal to time
    ro = dict()
    simTimes = pd.date_range(startTime, endTime, freq='D')
    simSeries_pd = pd.DataFrame(index=simTimes)
    for v in routing_method:
        ro[v] = getNetCDFData(output_routing_nc, v)
        for i in SEGID:
            idx = np.where(reachID == i)
            value = ro[v][:,idx[0]]
            simSeries_pd[i] = value

    simSeries_pd.to_csv(VIC_folder + "/sim_discharge.csv")

    # husekeeping
    for dirpath, dirnames, filenames in os.walk(routing_f+ "/input"):
        for file in filenames:
            os.remove(dirpath + "/" + file)

    return simSeries_pd


# if __name__ == "__main__":
#
#     # routing method = ['KWTroutedRunoff', 'IRFroutedRunoff']
#     routing_method = ['IRFroutedRunoff']
#     routing_f="/projects/mas1261/wa_software/mizuRoute/route"
#     VIC_folder="/projects/mas1261/wp3/VIC/ba_bsn_025/run_190816_nse"
#     startTime="2008-01-01"
#     endTime="2008-12-31"
#     script_dir="/projects/home/iff/Dmoss/vic_calibration/research/vic_calibration"
#     SEGID=[27,96, 101,155,174]
#     SEGID = [138] # should be Cungson station
#
#     run_routing(VIC_folder, routing_f, SEGID, script_dir, startTime, endTime, routing_method)

