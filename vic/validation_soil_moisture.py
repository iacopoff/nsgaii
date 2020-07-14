import os, sys

script_dir = "/projects/home/fmo/dmossgit/wa_data_analysis_branch/research"
sys.path.append(f"{script_dir}")

import spotpy
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import glob
import matplotlib.pyplot as plt
import itertools
from matplotlib.ticker import MaxNLocator
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.ion()
import time

from vic_calibration.format_soil_params import format_soil_params_distributed, format_soil_params
from vic_calibration.run_routing import *
from vic_calibration.cal_spotpy_functions import aggregate, read_clusters, set_interaction
from wa_data_analysis.FDC_plot import flow_duration_curve
from vic_calibration.eval_functions import *
from vic_calibration.cal_spotpy_functions import read_clusters, set_interaction, pickling, _parseConfig, _readFromFile


def reshape_dict(dict_in):
    for keys in dict_in:  # keys nrunf
        df = dict_in[keys]
        # useless if columns are string you might want to transform the out_dict keys in numbers again
        dict_in[keys].columns = dict_in[keys].columns.astype(str)
        new_keys = df.columns  # new keys reach id

    out_dict = dict.fromkeys(new_keys)

    for k in out_dict:
        df_empty = pd.DataFrame()
        for i in dict_in:
            df_extract = dict_in[i][k]
            df_extract = df_extract.rename(level=0, index=i)
            df_empty = df_empty.append(df_extract)
        out_dict[k] = df_empty.T

    return out_dict


def modify_vic_global_param_file(parfile, start_date, end_date, outfile=None):
    # overwrite (because 'w' mode) the file with our modified lines
    if not outfile:
        outfile = parfile

    with open(parfile, 'r') as oldfile:
        filedata = oldfile.read().splitlines(True)
    # overwrite (because 'w' mode) the file with our modified lines
    with open(outfile, 'w', newline='\r\n') as newfile:
        for line in filedata:
            if line[:10] == 'STARTYEAR ':
                newfile.write('STARTYEAR ' + start_date[:4] + '\n')
            elif line[:11] == 'STARTMONTH ':
                newfile.write('STARTMONTH ' + start_date[5:7] + '\n')
            elif line[:9] == 'STARTDAY ':
                newfile.write('STARTDAY ' + start_date[-2:] + '\n')
            elif line[:8] == 'ENDYEAR ':
                newfile.write('ENDYEAR ' + end_date[:4] + '\n')
            elif line[:9] == 'ENDMONTH ':
                newfile.write('ENDMONTH ' + end_date[5:7] + '\n')
            elif line[:7] == 'ENDDAY ':
                newfile.write('ENDDAY ' + end_date[-2:] + '\n')
            else:
                newfile.write(line)


def evaluation_coef(obsSeries_df, simCsv_df, cal_start, cal_end, val_start, val_end):
    # obsSeries_df=obs_df
    # simCsv_df=sim_df
    df_final = pd.DataFrame(index=["Calibration", "Validation"])

    obsSeries_df = obsSeries_df[min(val_start, cal_start):max(cal_end, val_end)]
    simCsv_df = simCsv_df[min(val_start, cal_start):max(cal_end, val_end)]
    # , "Monthly Cal", "Monthly Val"])
    df = pd.concat([obsSeries_df, simCsv_df], axis=1)
    df.columns = ["Obs", "Sim"]
    df_cal = df.loc[cal_start:cal_end, :]
    df_val = df.loc[val_start:val_end, :]
    # df_cal_month = df_cal.resample("M").sum()
    # df_val_month = df_val.resample("M").sum()

    df_final.loc["Calibration", "NSE"] = nse(df_cal['Sim'].values[:], df_cal['Obs'].values[:])
    df_final.loc["Validation", "NSE"] = nse(df_val['Sim'].values[:], df_val['Obs'].values[:])
    df_final.loc["Calibration", "logNSE"] = nse(np.log(df_cal['Sim'].values[:]), np.log(df_cal['Obs'].values[:]))
    df_final.loc["Validation", "logNSE"] = nse(np.log(df_val['Sim'].values[:]), np.log(df_val['Obs'].values[:]))

    df_final.loc["Calibration", "KGE"] = kge(df_cal['Sim'].values[:], df_cal['Obs'].values[:])[0]
    df_final.loc["Validation", "KGE"] = kge(df_val['Sim'].values[:], df_val['Obs'].values[:])[0]
    df_final.loc["Calibration", "logKGE"] = kge(np.log(df_cal['Sim'].values[:]), np.log(df_cal['Obs'].values[:]))[0]
    df_final.loc["Validation", "logKGE"] = kge(np.log(df_val['Sim'].values[:]), np.log(df_val['Obs'].values[:]))[0]

    df_final.loc["Calibration", "RMSE"] = rmse(np.log(df_cal['Sim'].values[:]), np.log(df_cal['Obs'].values[:]))
    df_final.loc["Validation", "RMSE"] = rmse(np.log(df_val['Sim'].values[:]), np.log(df_val['Obs'].values[:]))

    df_final.loc["Calibration", "MARE"] = mare(np.log(df_cal['Sim'].values[:]), np.log(df_cal['Obs'].values[:]))
    df_final.loc["Validation", "MARE"] = mare(np.log(df_val['Sim'].values[:]), np.log(df_val['Obs'].values[:]))

    df_final.loc["Calibration", "P Bias"] = pbias(df_cal['Sim'].values[:], df_cal['Obs'].values[:]) / 100
    df_final.loc["Validation", "P Bias"] = pbias(df_val['Sim'].values[:], df_val['Obs'].values[:]) / 100

    # df_final.loc["Monthly Cal", "NSE"] = nse(df_cal_month['Sim'].values[:], df_cal_month['Obs'].values[:])
    # df_final.loc["Monthly Val", "NSE"] = nse(df_val_month['Sim'].values[:], df_val_month['Obs'].values[:])
    # df_final.loc["Monthly Cal", "logNSE"] = nse(np.log(df_cal_month['Sim'].values[:]),
    #                                             np.log(df_cal_month['Obs'].values[:]))
    # df_final.loc["Monthly Val", "logNSE"] = nse(np.log(df_val_month['Sim'].values[:]),
    #                                             np.log(df_val_month['Obs'].values[:]))
    #
    # df_final.loc["Monthly Cal", "KGE"] = kge(df_cal_month['Sim'].values[:], df_cal_month['Obs'].values[:])[0]
    # df_final.loc["Monthly Val", "KGE"] = kge(df_val_month['Sim'].values[:], df_val_month['Obs'].values[:])[0]
    # df_final.loc["Monthly Cal", "logKGE"] = \
    #     kge(np.log(df_cal_month['Sim'].values[:]), np.log(df_cal_month['Obs'].values[:]))[0]
    # df_final.loc["Monthly Val", "logKGE"] = \
    #     kge(np.log(df_val_month['Sim'].values[:]), np.log(df_val_month['Obs'].values[:]))[0]
    #
    # df_final.loc["Monthly Cal", "RMSE"] = rmse(np.log(df_cal_month['Sim'].values[:]),
    #                                            np.log(df_cal_month['Obs'].values[:]))
    # df_final.loc["Monthly Val", "RMSE"] = rmse(np.log(df_val_month['Sim'].values[:]),
    #                                            np.log(df_val_month['Obs'].values[:]))
    #
    # df_final.loc["Monthly Cal", "MARE"] = mare(np.log(df_cal_month['Sim'].values[:]),
    #                                            np.log(df_cal_month['Obs'].values[:]))
    # df_final.loc["Monthly Val", "MARE"] = mare(np.log(df_val_month['Sim'].values[:]),
    #                                            np.log(df_val_month['Obs'].values[:]))
    #
    # df_final.loc["Monthly Cal", "P Bias"] = pbias(np.log(df_cal_month['Sim'].values[:]),
    #                                               np.log(df_cal_month['Obs'].values[:]))
    # df_final.loc["Monthly Val", "P Bias"] = pbias(np.log(df_val_month['Sim'].values[:]),
    #                                               np.log(df_val_month['Obs'].values[:]))
    df_final = df_final.round(2)
    return df_final


def plot_validation_results(obs_df, sim_df, cal_start, cal_end, val_start, val_end, param_df):
    """
    :param obs_df: pandas series of observation
    :param sim_df: pandas series of simulation
    :param cal_start: calibration start date "DD-MM-YYYY"
    :param cal_end:  calibration end date "DD-MM-YYYY"
    :param val_start: validation start date
    :param val_end: validation end date
    :return: fig with daily and monthly discharge, flow duration curve, table with objective functions
    """

    # obs_df=obs_in[i]
    # sim_df=sim_df_in
    if val_start > cal_start:
        line = cal_end

    else:
        line = cal_start
    fig1, bx = plt.subplots(2, 1, sharex="col", sharey="row", figsize=(17, 9.7))
    fig2 = plt.figure(figsize=(17, 9.7))
    ax0 = plt.subplot2grid((6, 2), (0, 0), rowspan=2)
    ax1 = plt.subplot2grid((6, 2), (2, 0), rowspan=2)
    ax2 = plt.subplot2grid((6, 2), (4, 0), rowspan=2)
    ax3 = plt.subplot2grid((6, 2), (0, 1), rowspan=2)
    ax4 = plt.subplot2grid((6, 2), (2, 1), rowspan=2)
    ax5 = plt.subplot2grid((6, 2), (4, 1))
    ax6 = plt.subplot2grid((6, 2), (5, 1))

    fig_n = str(obs_df.name)
    #slicing obs and sim to same start and end data to avoid Nan creation when concatenating
    obs_df=obs_df[min(val_start,cal_start):max(cal_end,val_end)]
    sim_df = sim_df[min(val_start, cal_start):max(cal_end, val_end)]
    df = pd.concat([obs_df, sim_df], axis=1)
    df.columns = ["Obs", "Sim"]  # todo put this first/outside the function
    df.plot(ax=bx[0], color=["r", "black"], linewidth=0.8)
    bx[0].grid(b=True, which='major', axis='y', color="k", linestyle='--',
               linewidth=0.5)
    bx[0].axvline(x=line, linewidth=1, color='black', linestyle='--')
    bx[0].set_ylabel('Daily Discharge [cms]', multialignment='center')

    # df_month = df.resample("M").sum()
    df_log = np.log(df)
    df_log.plot(ax=bx[1], color=["r", "black"], linewidth=0.8)
    bx[1].axvline(x=line, linewidth=1, color='black', linestyle='--')
    bx[1].grid(b=True, which='major', axis='y', color="k", linestyle='--',
               linewidth=0.5)
    bx[1].set_ylabel('Log Discharge [cms]', multialignment='center')
    df_cal = df.loc[cal_start:cal_end, :]
    df_val = df.loc[val_start:val_end, :]

    if val_start < cal_start:
        fig1.text(0.15, 0.05, "validation", fontsize=12, horizontalalignment='center', verticalalignment='center',
                  transform=bx[0].transAxes)
        fig1.text(0.88, 0.05, "calibration", fontsize=12, horizontalalignment='center', verticalalignment='center',
                  transform=bx[0].transAxes)
    else:
        fig1.text(0.15, 0.05, "calibration", fontsize=12, horizontalalignment='center', verticalalignment='center',
                  transform=bx[0].transAxes)
        fig1.text(0.88, 0.05, "validation", fontsize=12, horizontalalignment='center', verticalalignment='center',
                  transform=bx[0].transAxes)

    flow_duration_curve(df_cal['Obs'].values[:], ax=ax0, plot=True, log=True,
                        fdc_kwargs={"color": "red", "label": "Obs"})
    flow_duration_curve(df_cal['Sim'].values[:], ax=ax0, plot=True, log=True,
                        fdc_kwargs={"color": "black", "label": "Sim"})
    ax0.set_title("Log flow duration curve ")
    ax0.legend()
    # ax0.text(10, -4, "calibration", fontsize=12, rotation=90))
    ax0.set_ylabel('calibration\n[cms]', multialignment='center', fontsize=12)
    flow_duration_curve(df_val['Obs'].values[:], ax=ax1, plot=True, log=True,
                        fdc_kwargs={"color": "red", "label": "Obs"})
    flow_duration_curve(df_val['Sim'].values[:], ax=ax1, plot=True, log=True,
                        fdc_kwargs={"color": "black", "label": "Sim"})
    # ax1.text(85, 6000, "validation", fontsize=12)
    ax1.set_ylabel("validation\n[cms]", multialignment='center', fontsize=12)
    ax1.set_xlabel('% of exceedence', multialignment='center')
    ax1.legend()

    flow_duration_curve(df_cal['Obs'].values[:], ax=ax3, plot=True, log=False,
                        fdc_kwargs={"color": "red", "label": "Obs"})
    flow_duration_curve(df_cal['Sim'].values[:], ax=ax3, plot=True, log=False,
                        fdc_kwargs={"color": "black", "label": "Sim"})
    ax3.set_title("Flow duration curve ")
    ax3.legend()
    # ax3.text(85, 6000, "calibration", fontsize=12)
    ax3.set_ylabel('[cms]', multialignment='center')
    flow_duration_curve(df_val['Obs'].values[:], ax=ax4, plot=True, log=False,
                        fdc_kwargs={"color": "red", "label": "Obs"})
    flow_duration_curve(df_val['Sim'].values[:], ax=ax4, plot=True, log=False,
                        fdc_kwargs={"color": "black", "label": "Sim"})
    # ax4.text(85, 6000, "validation", fontsize=12)
    ax4.set_ylabel('[cms]', multialignment='center')
    ax4.set_xlabel('% of exceedence', multialignment='center')
    ax4.legend()

    eval_table = evaluation_coef(obs_df, sim_df, cal_start, cal_end, val_start, val_end)
    cell_text = []
    for row in range(len(eval_table)):
        cell_text.append(eval_table.iloc[row])

    # plot eval coeff
    eval_table.T.plot.bar(ax=ax2, color=["blue", "green"], rot=45, width=0.4)
    ax2.grid(b=True, which='major', axis='y', color="k", linestyle='--',
             linewidth=0.5)
    # eval_tabledf_final
    table = ax5.table(cellText=cell_text, rowLabels=eval_table.index, colLabels=eval_table.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.9)
    ax5.axis("off")
    # param table

    cell_text_pr = []
    for row in range(len(param_df)):
        cell_text_pr.append(param_df.iloc[row])
    table_p = ax6.table(cellText=cell_text_pr, rowLabels=param_df.index, colLabels=param_df.columns, loc='center')
    table_p.auto_set_font_size(False)
    table_p.set_fontsize(10)
    table_p.scale(1, 1.9)
    ax6.axis("off")

    return fig1, fig2


def plot_validation_results_multi(obs_df, sim_df, cal_start, cal_end, val_start, val_end):
    # obs_in[i], sim_df, cal_start, cal_end, val_start, val_end
    """
    :param obs_df: pandas series of observation
    :param sim_df: data frame with simulation (more than 1)
    :param cal_start: calibration start date "DD-MM-YYYY"
    :param cal_end:  calibration end date "DD-MM-YYYY"
    :param val_start: validation start date
    :param val_end: validation end date
    :return: fig 1, daily and monthly discharge, fig2 calibration and validation duration curves
    """
    # creating fig 1 with discharge data
    # obs_df=obs_in[i]
    # sim_df=sim_df_in
    if val_start > cal_start:
        line = cal_end
    else:
        line = cal_start
    fig1, ax = plt.subplots(2, 1, sharex="col", sharey="row", figsize=(17, 9.7))
    col_sim_id = sim_df.columns.tolist()
    sim_col_name = ["sim {0}".format(i) for i in col_sim_id]
    sim_df.colums = sim_col_name

    sim_df.index = pd.to_datetime(sim_df.index)
    df = pd.concat([obs_df, sim_df], axis=1)
    df.columns = ["Obs"] + sim_col_name  # todo put this first/outside the function
    # generate palette a make it as a list of hex colors
    pal_sim = (sns.color_palette("bright", len(sim_col_name)).as_hex())
    pal_sim_list = []
    for i in range(len(sim_col_name)):
        pal_sim_list.append(pal_sim[i])

    sim_df.plot(ax=ax[0], color=pal_sim_list, linewidth=0.8)
    df["Obs"].plot(ax=ax[0], color="#000000", legend=True, linewidth=1.2, linestyle='--')
    ax[0].grid(b=True, which='major', axis='y', color="k", linestyle='--',
               linewidth=0.5)
    ax[0].axvline(x=line, linewidth=1, color='black', linestyle='--')
    ax[0].set_ylabel('Daily Discharge [cms]', multialignment='center')

    df_log = np.log(df)
    sim_df_log = np.log(sim_df)
    # df_log["Obs"].plot(ax=ax[1], color="#000000",legend=True,linewidth=0.8)
    sim_df_log.plot(ax=ax[1], color=pal_sim_list, linewidth=0.8, legend=False)
    df_log["Obs"].plot(ax=ax[1], color="#000000", legend=False, linewidth=1.2, linestyle='--')
    ax[1].axvline(x=line, linewidth=1, color='black', linestyle='--')
    ax[1].grid(b=True, which='major', axis='y', color="k", linestyle='--',
               linewidth=0.5)
    ax[1].set_ylabel('Log Discharge [cms]', multialignment='center')

    if val_start < cal_start:
        fig1.text(0.15, 0.05, "validation", fontsize=12, horizontalalignment='center', verticalalignment='center',
                  transform=ax[0].transAxes)
        fig1.text(0.88, 0.05, "calibration", fontsize=12, horizontalalignment='center', verticalalignment='center',
                  transform=ax[0].transAxes)
    else:
        fig1.text(0.15, 0.05, "calibration", fontsize=12, horizontalalignment='center', verticalalignment='center',
                  transform=ax[0].transAxes)
        fig1.text(0.88, 0.05, "validation", fontsize=12, horizontalalignment='center', verticalalignment='center',
                  transform=ax[0].transAxes)
    ax[0].legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0)

    df_cal_obs = df['Obs'][cal_start:cal_end]
    df_val_obs = df['Obs'][val_start:val_end]

    df_cal_sim = sim_df.loc[cal_start:cal_end, :]
    df_val_sim = sim_df.loc[val_start:val_end, :]

    # fig2 with flow duration curves
    fig2 = plt.figure(figsize=(17, 9.7))
    ax0 = plt.subplot2grid((2, 2), (0, 0))
    ax1 = plt.subplot2grid((2, 2), (1, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 1))

    for i, col in enumerate(sim_df.columns):
        flow_duration_curve(df_cal_sim[col].values[:], ax=ax0, plot=True, log=True,
                            fdc_kwargs={"color": pal_sim_list[i]})
    flow_duration_curve(df_cal_obs.values[:], ax=ax0, plot=True, log=True,
                        fdc_kwargs={"color": "black", "label": "Obs", "linewidth": "2"})
    # ax0.legend(( np.append(np.asarray(df.columns[1:]),np.asarray(df_cal_obs.name))))
    ax0.set_title("Log Flow duration curve ")
    ax0.set_ylabel('calibration\n [cms]', multialignment='center', fontsize=12)

    for i, col in enumerate(sim_df.columns):
        flow_duration_curve(df_val_sim[col].values[:], ax=ax1, plot=True, log=True,
                            fdc_kwargs={"color": pal_sim_list[i], "label": sim_col_name})
    flow_duration_curve(df_val_obs.values[:], ax=ax1, plot=True, log=True,
                        fdc_kwargs={"color": "black", "label": "Obs", "linewidth": "2"})
    # ax1.legend(( np.append(np.asarray(df.columns[1:]),np.asarray(df_cal_obs.name))))
    ax1.set_title("Log Flow duration curve - validation")
    ax1.set_ylabel('validation\n[cms]', multialignment='center', fontsize=12)
    ax1.set_xlabel('% of exceedence', multialignment='center')

    for i, col in enumerate(sim_df.columns):
        flow_duration_curve(df_cal_sim[col].values[:], ax=ax2, plot=True, log=False,
                            fdc_kwargs={"color": pal_sim_list[i]})
    flow_duration_curve(df_cal_obs.values[:], ax=ax2, plot=True, log=False,
                        fdc_kwargs={"color": "black", "label": "Obs"})
    ax2.legend((np.append(np.asarray(df.columns[1:]), np.asarray(df_cal_obs.name))))
    ax2.set_title("Flow duration curve -calibration")
    ax2.set_ylabel('[cms]', multialignment='center')
    ax2.set_ylabel('calibration\n [cms]', multialignment='center', fontsize=12)
    ax2.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0)

    for i, col in enumerate(sim_df.columns):
        flow_duration_curve(df_val_sim[col].values[:], ax=ax3, plot=True, log=False,
                            fdc_kwargs={"color": pal_sim_list[i], "label": sim_col_name})

    flow_duration_curve(df_val_obs.values[:], ax=ax3, plot=True, log=False,
                        fdc_kwargs={"color": "black", "label": "Obs"})
    # ax3.legend((np.asarray(df.columns)))
    ax3.set_title("Flow duration curve - validation")
    ax3.set_ylabel('[cms]', multialignment='center')
    ax3.set_xlabel('% of exceedence', multialignment='center')

    # eval_table = evaluation_coef(obs_df, sim_df, cal_start, cal_end, val_start, val_end)
    # cell_text = []
    # for row in range(len(eval_table)):
    #     cell_text.append(eval_table.iloc[row])
    # table = ax5.table(cellText=cell_text, rowLabels=eval_table.index, colLabels=eval_table.columns, loc='center')
    # table.auto_set_font_size(False)
    # table.set_fontsize(12)
    # table.scale(1, 1.9)
    # ax5.axis("off")
    return fig1, fig2


def run_vic_and_routing(dir_work, dir_routexe, start_date, end_date, river_reachid):
    # start_date=start_date_VIC
    # end_date=end_date_VIC
    # dir_work=VIC_dir
    '''
    :param dir_work: path of vic folder
    :param dir_routexe: path of routing folder
    :param start_date: start date of simulation  "1987-01-01"
    :param end_date: end date of simulation format "1987-01-01"
    :river_reachid: list with river id to extract from routing output
    :return: csv with simulated discharge save in dir_work for the rivers specified in the river_isd list
    '''
    # start = time.time()
    print("executing vic...")
    # os.system(f'vic_image.exe -g {global_file_new}')
    # run_vic_subprocess(global_file)
    # subprocess.run(f'vic_image.exe -g {global_file}', stderr = subprocess.PIPE,stdout = subprocess.PIPE, shell=True)
    #adding 2 years to VIC, allowing for warm-up
    VIC_start=(pd.Timestamp(start_date) - pd.DateOffset(years=2)).strftime(format="%Y-%m-%d")
    global_file = glob.glob(os.path.join(dir_work, "global*"))[0]
    modify_vic_global_param_file(global_file, VIC_start, end_date, outfile=None)
    subprocess.run(f'vic_image.exe -g {global_file}', stderr=subprocess.PIPE, stdout=subprocess.PIPE,  # mpiexec -np 2
                   shell=True)
    # run routing model
    print("run routing...")


    # run_routing_dev(VIC_folder=dir_work, routing_f=dir_routexe, SEGID=river_reachid,
    #                 startTime=pd.Timestamp(VIC_start).strftime(format="%Y-%m-%d"),
    #                 endTime=pd.Timestamp(end_date).strftime(format="%Y-%m-%d"),
    #                 routing_method=['IRFroutedRunoff'])
    # print("...succesful!")
    # simulated_df = pd.read_csv(os.path.join(dir_work, "sim_discharge.csv"), index_col=0, parse_dates=True)

    return
    #return    simulated_df


def validation(dir_work, dir_routexe, river_reachid, discharge_summary, discharge_folder, cal_start,
               cal_end, n_best_runs,clusters_ranks,cal_output_name,model_type,config_global):
    """
    :param dir_work: path of vic folder
    :param dir_routexe: path of routing folder
    :param river_reachid: list with river id [34,44,99]
    :param discharge_summary: summary excel table with discharge data
    :param discharge_folder: folder where csv discharges are saved
    :param cal_start: calibration start date "DD-MM-YYYY"
    :param cal_end:  calibration end date "DD-MM-YYYY"
    :param val_start: validation start date
    :param val_end: validation end date
    :param n_best_runs: n of best runs to analyze
    :return: figures with calibrated and validated discharges
    """
    #  if not os.path.exists(os.joinpath(dir_work,"validation")):
    #    os.makedirs(os.joinpath(dir_work,"validation"))
    # make validation dir

    sim_all = {}
    eval_table_all = {}
    # check if dir_work/run exist, if it does get param and global parameter file form there
    if os.path.isdir(os.path.join(dir_work, "run")):
        param_file = glob.glob(os.path.join(dir_work, "run", "param*.nc"))[0]
        global_file = glob.glob(os.path.join(dir_work, "run", "global*"))[0]
        VIC_dir = os.path.join(dir_work, "run")
    else:
        param_file = glob.glob(os.path.join(dir_work, "param*.nc"))[0]
        global_file = glob.glob(os.path.join(dir_work, "global*"))[0]
        VIC_dir = dir_work

    val_dir = os.path.join(dir_work, "validation")
    if not os.path.isdir(os.path.join(dir_work, "validation")):
        os.mkdir(val_dir)
    if model_type=="lumped":
        params = []
        for par in config_global['vic_parameters']:
            p = config_global['vic_parameters'][par].split(",")
            minmax = [float(i) for i in p[1:]]
            param_distr = getattr(spotpy.parameter, p[0])
            params.append(param_distr(par, *minmax))
        n_par = len(params)

    else:
        params = []
        for par in config_global['vic_parameters']:
            p = config_global['vic_parameters'][par].split(",")
            minmax = [float(i) for i in p[1:]]
            param_distr = getattr(spotpy.parameter, p[0])
            params.append(param_distr(par, minmax[0], minmax[1]))
        clusters, parnames_initial, relation = read_clusters(clusters_ranks)
        param, parnames = set_interaction(params, clusters, relation)
        n_par = len(parnames)
    cal_file = pd.read_csv(dir_work + "/" + cal_output_name + ".csv")
    # df_discharge = pd.read_excel(discharge_summary)
    # # orig=selection
    # selection = df_discharge.loc[df_discharge['routing_id'].isin(river_reachid)]
    # selection.loc[:, 'routing_id'] = selection.loc[:, 'routing_id'].astype(int)
    # selection["val_start"] = ""
    # selection["val_end"] = ""
    # for r in river_reachid:
    #     start, end = selection[selection.routing_id == r].val_length.tolist()[0].split("/")
    #     selection.loc[selection.routing_id == r, "val_start"] = start
    #     selection.loc[selection.routing_id == r, "val_end"] = end
    #
    # # all_dates together
    # all_dates = selection["val_start"].tolist() + selection["val_end"].tolist() + [cal_start, cal_end]
    # start_date_VIC = min(all_dates)
    # end_date_VIC = max(all_dates)
    start_date_VIC = "1980-01-01"
    end_date_VIC = "2010-12-31"
    # best_values=cal_file.sort_values(by="like1", ascending=True).iloc[0, 0:n_par + 1]
    # clusters, parnames_initial, relation = read_clusters(clusters_ranks)
    eval_table_river = pd.DataFrame()
    for b in range(n_best_runs):
        # getting the values of the parameters for the n best run
        vector = cal_file.sort_values(by="like1", ascending=True).iloc[b, 1:n_par + 1]
        cal_ind=cal_file.sort_values(by="like1", ascending=True).index[b]
        vector_list = vector.tolist()

        param_names = cal_file.columns[1:n_par + 1].tolist()
        if model_type=="lumped":
            format_soil_params(param_file, **{name.name: param for name, param in zip(params, vector)})
            par_nam=[x.replace("par", "") for x in param_names]
            param_df = pd.DataFrame(columns=par_nam, index=[1])
            for p in par_nam:
                vect_id = "par" + p
                param_df.loc[1, p] = vector[vect_id]
            param_df = param_df.astype(float).round(4)

        else:
            clusters = list(set([x.split(".")[1] for x in param_names]))
            par_nam = list(set([x.split(".")[0] for x in parnames]))
            param_df = pd.DataFrame(columns=par_nam, index=clusters)
            for c in clusters:
                for p in par_nam:
                    vect_id = "par" + p + "." + c
                    param_df.loc[c, p] = vector[vect_id]
            param_df = param_df.astype(float).round(4)

        # placing parameters values in VIC parameter file
            clusters, parnames_initial, relation = read_clusters(clusters_ranks)
            format_soil_params_distributed(nc_file=param_file, gridcells=clusters,
                                       **{name: param for name, param in zip(parnames, vector_list)})

        sim_df_out = run_vic_and_routing(VIC_dir, dir_routexe, start_date_VIC,end_date_VIC,river_reachid)
        fluxes_output_name = glob.glob(VIC_dir + "/fluxes*")[0]
        os.rename(fluxes_output_name,fluxes_output_name[-3]+"_run_"+str(b)+".nc")
    #     sim_df_out.to_csv(val_dir+"/simulation_"+str(b+1)+".csv")
    #     sim_df_out = pd.read_csv(val_dir + "/simulation_" + str(b + 1) + ".csv", index_col=0)
    #     if n_best_runs > 1:
    #         sim_all[b + 1] = sim_df_out
    #
    #     sim_df = sim_df_out
    #     sim_df.index.name = "time"
    #     sim_df.columns = [i for i in river_reachid]
    #     # get observation
    #
    #     print(" validating with " + str(selection['station'].tolist()))
    #
    #     obs_in = pd.DataFrame()
    #     l0, l1 = river_reachid, ['Calibration', 'Validation']
    #
    #     for i in river_reachid:
    #         val_start = selection[selection.routing_id == i].val_start.tolist()[0]
    #         val_end = selection[selection.routing_id == i].val_end.tolist()[0]
    #
    #         id = selection['routing_id'] == i
    #         valFile = selection[id]['excel_name'].tolist()[0]
    #         obs_in_t = pd.read_csv(discharge_folder + "/" + valFile + ".csv", usecols=[0, 1])
    #         obs_in_t.columns = ["time", i]
    #         obs_in_t['time'] = pd.to_datetime(obs_in_t.time, infer_datetime_format=True, dayfirst=True)
    #         obs_in_t = obs_in_t[(obs_in_t.time >= pd.Timestamp(min(cal_start, val_start))) & (
    #                     obs_in_t.time <= pd.Timestamp(max(cal_end, val_end)))]
    #         obs_in_t = obs_in_t.set_index("time")
    #         if obs_in.empty:
    #             obs_in = obs_in_t
    #         else:
    #             obs_in = pd.concat([obs_in, obs_in_t], sort=True, axis=1)
    #         # obsSeries = obsSeries_df.values[:,0]
    #         # for i in river_reachid:
    #         # obs_df_in = obs_in[i]
    #         sim_df_in = sim_df[i][min(cal_start, val_start):max(cal_end, val_end)]
    #         print(sim_df_in.head())
    #         station_name = df_discharge[df_discharge.routing_id == i].station.tolist()[0]
    #         eval_table = evaluation_coef(obs_in[i], sim_df_in, cal_start, cal_end, val_start, val_end)
    #         if eval_table_river.empty:
    #             eval_table.loc[:, "reach_id"] = i
    #             eval_table.loc[:, "n_best_run"] = b
    #             eval_table_river = eval_table
    #         else:
    #             eval_table.loc[:, "reach_id"] = i
    #             eval_table.loc[:, "n_best_run"] = b
    #             eval_table_river = eval_table_river.append(eval_table)
    #
    #         sim_all[b + 1] = sim_df_out
    #         fig1, fig2 = plot_validation_results(obs_in[i], sim_df_in, cal_start, cal_end, val_start, val_end, param_df)
    #         fig1.suptitle(dir_work.split("/")[-1] + " at " + station_name + " best run n " + str(b + 1))
    #         fig2.suptitle(dir_work.split("/")[-1] + " at " + station_name + " best run n " + str(b + 1))
    #         fig1.savefig(val_dir + "/validation_plot_flows_" + str(i) + "_bestrun_" + str(b + 1)+"_run_" + str(cal_ind)+".png")
    #         fig2.savefig(val_dir + "/validation_plot_FDC_" + str(i) + "_bestrun_" + str(b + 1)+"_run_" + str(cal_ind)+".png")
    #         plt.close('all')
    #
    # if n_best_runs > 1:
    #     # reshape dictionary
    #     sim_dict_out = reshape_dict(sim_all)
    #     # create plot for all validation
    #     for i in river_reachid:
    #         station_name = df_discharge[df_discharge.routing_id == i].station.tolist()[0]
    #         val_start = selection[selection.routing_id == i].val_start.tolist()[0]
    #         val_end = selection[selection.routing_id == i].val_end.tolist()[0]
    #         sim_df = sim_dict_out[str(i)][min(cal_start, val_start):max(cal_end, val_end)]
    #         fig1, fig2 = plot_validation_results_multi(obs_in[i], sim_df, cal_start, cal_end, val_start, val_end)
    #         fig1.suptitle(dir_work.split("/")[-1] + " at " + station_name)
    #         fig1.savefig(val_dir + "/validation_plot_discharge_combined_runs_" + str(i) + "_" + station_name + ".png")
    #         fig2.suptitle(dir_work.split("/")[-1] + " at " + station_name)
    #         fig2.savefig(val_dir + "/validation_plot_FDC_combined_runs_" + str(i) + "_" + station_name + ".png")
    #     eval_table_river["cal_val"] = eval_table_river.index
    #     eval_table_river_melt = eval_table_river.melt(id_vars=['reach_id', 'n_best_run', 'cal_val'])
    #     eval_table_river_melt["best runs"] = eval_table_river_melt["n_best_run"] + 1
    #     eval_table_river_melt.to_csv(dir_work + "/eval_table.csv")
    #     # obj_dic={"obj1":['NSE','logNSE'],"obj2":["KGE","logKGE"],"obj3":["MARE",]}
    #     add_to_best_runs = np.linspace(-0.2, 0.2, (len(river_reachid) * 2)).tolist() * n_best_runs * 3
    #
    #     data0 = eval_table_river_melt[eval_table_river_melt.variable.isin(['NSE', 'logNSE', "KGE"])]
    #     data1 = eval_table_river_melt[eval_table_river_melt.variable.isin(['RMSE', 'MARE', 'P Bias'])]
    #     data0 = data0.sort_values(by=["reach_id", "n_best_run"])
    #     data1 = data1.sort_values(by=["reach_id", "n_best_run"])
    #     data0["best runs"] = data0["best runs"] + add_to_best_runs
    #     data1["best runs"] = data1["best runs"] + add_to_best_runs
    #
    #     data0["Legend"] = data0["variable"].astype(str) + " " + data0["cal_val"]
    #     data1["Legend"] = data1["variable"].astype(str) + " " + data1["cal_val"]
    #     fig6, bx = plt.subplots(len(river_reachid), 1, figsize=(30, 13))
    #     fig7, cx = plt.subplots(len(river_reachid), 1, figsize=(30, 13))
    #     for n, r in enumerate(river_reachid):
    #         minor_ticks = np.arange(1.5, n_best_runs, 1)
    #         major_ticks = np.arange(1, n_best_runs + 1, 1)
    #         datab = data0[data0.reach_id == r]
    #         datac = data1[data1.reach_id == r]
    #         if len(river_reachid)==1:
    #             bx = sns.scatterplot(x="best runs", y="value", data=datab, hue="Legend", style="cal_val",
    #                                     palette=sns.color_palette("Paired", 6), ax=bx, s=150)
    #             cx = sns.scatterplot(x="best runs", y="value", data=datac, hue="Legend", style="cal_val",
    #                                     palette=sns.color_palette("Paired", 6), ax=cx, s=150)
    #             bx.set_ylabel(ylabel)
    #             cx.set_ylabel(ylabel)
    #             bx.xaxis.set_major_locator(MaxNLocator(integer=True))
    #             bx.xaxis.set_minor_locator(MaxNLocator(integer=True))
    #             cx.xaxis.set_major_locator(MaxNLocator(integer=True))
    #             bx.set_xticks(minor_ticks, minor=True)
    #             cx.set_xticks(minor_ticks, minor=True)
    #             cx.grid(which='minor', alpha=0.7)
    #             bx.grid(which='minor', alpha=0.7)
    #             bx.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0)
    #             cx.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0)
    #         elif n > 0:
    #             bx[n] = sns.scatterplot(x="best runs", y="value", data=datab, hue="Legend", style="cal_val",
    #                                     palette=sns.color_palette("Paired", 6), ax=bx[n], s=150, legend=False)
    #             cx[n] = sns.scatterplot(x="best runs", y="value", data=datac, hue="Legend", style="cal_val",
    #                                     palette=sns.color_palette("Paired", 6), ax=cx[n], s=150, legend=False)
    #         else:
    #             bx[n] = sns.scatterplot(x="best runs", y="value", data=datab, hue="Legend", style="cal_val",
    #                                     palette=sns.color_palette("Paired", 6), ax=bx[n], s=150)
    #             cx[n] = sns.scatterplot(x="best runs", y="value", data=datac, hue="Legend", style="cal_val",
    #                                     palette=sns.color_palette("Paired", 6), ax=cx[n], s=150)
    #
    #         station_name = df_discharge[df_discharge.routing_id == r].station.tolist()[0]
    #         ylabel = "river " + station_name
    #         if len(river_reachid) != 1:
    #             bx[n].set_ylabel(ylabel)
    #             cx[n].set_ylabel(ylabel)
    #             bx[n].xaxis.set_major_locator(MaxNLocator(integer=True))
    #             bx[n].xaxis.set_minor_locator(MaxNLocator(integer=True))
    #             cx[n].xaxis.set_major_locator(MaxNLocator(integer=True))
    #             bx[n].set_xticks(minor_ticks, minor=True)
    #             cx[n].set_xticks(minor_ticks, minor=True)
    #             cx[n].grid(which='minor', alpha=0.7)
    #             bx[n].grid(which='minor', alpha=0.7)
    #
    #     # bx0 = sns.scatterplot(x="best runs", y="value", data=data0, hue="Legend", style="reach_id",palette=sns.color_palette("hls", 8), ax=bx0, s=100)
    #             bx[0].legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0)
    #             cx[0].legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0)
    #
    #     # fig6.suptitle("Objective functions summary")
    #     fig6.suptitle("Objective functions summary NSE log NSE KGE")
    #     fig7.suptitle("Objective functions summary RMSE_MARE_P Bias")
    #     fig6.tight_layout()
    #     fig7.tight_layout()
    #     fig6.savefig(val_dir + "/combined_obj_NSE_logNSE_KGE" + str(n_best_runs) + ".png", bbox_inches='tight')
    #     fig7.savefig(val_dir + "/combined_obj_RMSE_MARE_P Bias" + str(n_best_runs) + ".png", bbox_inches='tight')
    #     plt.close('all')

    return


if __name__ == "__main__":

    n_best_runs = 25
    config_global = _readFromFile("/projects/mas1261/wp3/VIC/mekong_025/run/20200220/config_new.ini")  # _readFromFile(sys.argv[1]) #'_readFromFile("/projects/mas1261/wp3/VIC/machu_025/run/191001/config.ini")  #  #  _readFromFile("/projects/mas1261/wp3/VIC/machu_025/run/191001/config.ini") # #) #

    global_options = _parseConfig(config_global)

    model_type = global_options['vic_config']['model_type']
    cal_mode = global_options['vic_config']['cal_mode']
    cal_alg = global_options['vic_config']['cal_alg']

    obj_funct = global_options['vic_config']['obj_funct']
    obj_f_opt_direction = global_options['vic_config']['obj_f_opt_direction']

    dir_work = global_options['vic_config']['dir_work']
    dir_routexe = global_options['vic_config']['dir_routexe']
    discharge_summary = global_options['vic_config']['discharge_summary']
    discharge_folder = global_options['vic_config']['discharge_folder']
    dir_scripts = global_options['vic_config']['dir_scripts']
    river_reachid = [int(i) for i in global_options['vic_config']['river_reachid'].split(
        ",")]  # rivers and river ID as defined in mizourute
    # print(river_reachid)
    cal_start = global_options['vic_config']['cal_start']  # starting sim time
    cal_end = global_options['vic_config']['cal_end']  # end sim time
    time_step = global_options['vic_config']['time_step']  # calibration timestep (D= daily, M=monthly)
    n_of_runs = int(global_options['vic_config']['n_of_runs'])  # number of runs
    param_file = global_options['vic_config']['param_file']  # TODO dove lo legge
    weights = [int(i) for i in global_options['vic_config']['weights'].split(",")]
    cal_output_name = global_options['vic_config']['cal_output_name']
    clusters_ranks = global_options['vic_config']['clusters_ranks']

    validation(dir_work, dir_routexe, river_reachid, discharge_summary, discharge_folder, cal_start,
           cal_end, n_best_runs,clusters_ranks,cal_output_name,model_type,config_global)

##enter validation time


###copy from you calibration folder

# dir_work = "/projects/mas1261/wp3/VIC/ba_bsn_025/run_190909_nse_daily"  # working directory
# dir_routexe = "/projects/mas1261/wa_software/routing_fmo/route"  # routing directory
# # discharge location
# discharge_summary = "/projects/mas1261/wp3/model_variables/discharge/final_output/Summary_19.08.xlsx"
# discharge_folder = "/projects/mas1261/wp3/model_variables/discharge/final_output/daily"
#
# river_reachid = [138, 28]  # ,"Ayunpa":[89],"Ankhe":[28]}                      # rivers and river ID as defined in mizourute
# cal_start = "1980-01-01"  # starting sim time
# cal_end = "1986-12-31"  # end sim time
#
# time_step = "D"  # calibration timestep (D= daily, M=monthly)
# n_streams = 1
# n_best_runs = 10
# # this is for the multi site run.
# # EX: {"river1":[121],"river2":[324],"river3":[424]} could have weights = [1,0.4,0.8]
# # because you think the river1 is more reliable than the others.
# weights = [1, 1, 1]
#
# # name of the calibration output table
# cal_output_name = "cal_nse_log_distr_correct"
#
# # file containing the parameter clusters
#
# rank_in = "/projects/mas1261/wp3/VIC/template/cluster_parameters/cluster_rank_ba.csv"
# # this is where you define the parameter names, the min and max (for a uniform distribution).
# params = [spotpy.parameter.Uniform('depth2d', 0.2, 0.8),  # 0.1, 1.2),
#           spotpy.parameter.Uniform('depth3d', 0.8, 1.6),  # 0.5, 2.1),
#           spotpy.parameter.Uniform('Ds1d', 0.8, 1.6),  # 0.001, 0.5),
#           spotpy.parameter.Uniform('Dsmax1d', 4, 26),  # 1, 30),
#           spotpy.parameter.Uniform('infilt1d', 0.001, 0.35),  # 0.001, 0.5)
#           spotpy.parameter.Uniform('Ws1d', 0.5, 1)]  #
#
# validation(dir_work, dir_routexe, river_reachid, discharge_summary, discharge_folder, val_start, val_end, cal_start,
#            cal_end, n_best_runs)
