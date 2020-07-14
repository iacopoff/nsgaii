import numpy as np
import csv
import os


def edit_vic_global(file,par_file,parall_dir,globalFile_new):
    with open(file, 'r') as dest_f:
        data_iter = csv.reader(dest_f,
                               delimiter=',',
                               quotechar='"')
        data = [data for data in data_iter]


    data = np.asarray(data, dtype=str)


    index_par = [n for n,i in enumerate(data[:,0]) if "PARAMETERS" in i]
    index_results = [n for n,i in enumerate(data[:,0]) if "RESULT_DIR" in i]

    data[index_par, 0] = "PARAMETERS {}".format(parall_dir + "/" + par_file)

    data[index_results, 0] = "RESULT_DIR {}".format(parall_dir + "/")



    fout = open(parall_dir + "/" + globalFile_new,"w")

    for n,i in enumerate(data):
        if n == len(data)-1:
            fout.write(i[0])
        else:
            fout.write(i[0] + "\n")

    fout.close()




def edit_routing_config(config_file,parall_dir,config_new):
    with open(config_file, 'r') as dest_f:
        data_iter = csv.reader(dest_f,
                               delimiter=',',
                               quotechar='"')
        data = [data for data in data_iter]


    data = np.asarray(data, dtype=str)


    index_ancillary = [n for n,i in enumerate(data[:,0]) if "<ancil_dir>" in i]
    index_input = [n for n, i in enumerate(data[:, 0]) if "<input_dir>" in i]
    index_output = [n for n, i in enumerate(data[:, 0]) if "<output_dir>" in i]
    index_param = [n for n,i in enumerate(data[:,0]) if "<param_nml>" in i]

    data[index_ancillary, 0] = "<ancil_dir> {}".format(os.path.join(parall_dir,"ancillary_data/"))
    data[index_input , 0] = "<input_dir> {}".format(os.path.join(parall_dir,"input/"))
    data[index_output, 0] = "<output_dir> {}".format(os.path.join(parall_dir,"output/"))
    data[index_param, 0] = "<param_nml> {}".format(os.path.join(parall_dir,"settings/param.nml.default"))




    fout = open(os.path.join(parall_dir,"settings",config_new),"w")

    for n,i in enumerate(data):
        if n == len(data)-1:
            fout.write(i[0])
        else:
            fout.write(i[0] + "\n")

    fout.close()

#
# config_file = "/projects/mas1261/wa_software/routing/route/settings/route.control"
# parall_dir = "/projects/mas1261/wa_software/routing/route_4"
#
# config_new = "wathereve"
#
# config_file  = "/projects/home/iff/routing/mizu/settings/routing_cal.control"

def edit_routing_config_dev(config_file,parall_dir,config_new):
    print(f"running  in {parall_dir}")
    with open(config_file, 'r') as dest_f:
        data_iter = csv.reader(dest_f,
                               delimiter=',',
                               quotechar='"')
        data = [data for data in data_iter]
        [type(i[0]) for i in data]

    data = np.asarray(data, dtype=object)#str)


    index_ancillary = [n for n,i in enumerate(data[:]) if "<ancil_dir>" in i[0]]
    index_input = [n for n, i in enumerate(data[:]) if "<input_dir>" in i[0]]
    index_output = [n for n, i in enumerate(data[:]) if "<output_dir>" in i[0]]
    #index_param = [n for n,i in enumerate(data[:,0]) if "<param_nml>" in i]

    data[index_ancillary] = [["<ancil_dir> {}  !".format(os.path.join(parall_dir,"ancillary_data/"))]]
    data[index_input ] = [["<input_dir> {}  !".format(os.path.join(parall_dir,"input/"))]]
    data[index_output] = [["<output_dir> {}  !".format(os.path.join(parall_dir,"output/"))]]
    #data[index_param, 0] = "<param_nml> {}".format(os.path.join(parall_dir,"settings/param.nml.default"))




    fout = open(os.path.join(parall_dir,"settings",config_new),"w")

    for n,i in enumerate(data):
        if n == len(data)-1:
            fout.write(i[0])
        else:
            fout.write(i[0] + "\n")

    fout.close()
