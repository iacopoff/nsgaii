import csv
import numpy as np

class Database:
    def __init__(self,obj,params,param_labels,sim,connection):
        self.obj = obj
        self.obj_labels = [f"obj_{i}" for i in range(self.obj.shape[1])]

        self.params = params
        self.param_labels = param_labels

        self.sim = sim
        self.sim_labels =  [f"sim_{idataset}_{igrid}" for idataset,dataset in enumerate(self.sim) for igrid in range(len(dataset))]

        self.connection = connection


    def _create_header(self):
        self.headers = self.obj_labels + self.param_labels + self.sim_labels 



class TableWriter(Database):


    def init_db(self):

        with open(self.connection +".csv",'w') as conn:
            writer = csv.writer(conn,delimiter=',')
            writer.writerow(self.headers)
            conn.flush()

    def write(self,obj,params,sim):


        self.obj = obj
        self.params = params

        self.sim = self.simulation_adaptor(sim)

        #import pdb; pdb.set_trace()
        
        self.obj = np.round(self.obj,3)
        self.params = np.round(self.params,3)

        self.sim = np.round(self.sim,3)

        with open(self.connection + ".csv",'a') as conn:
            writer = csv.writer(conn,delimiter=',')
            for o,p,s in zip(self.obj,self.params,self.sim):

                writer.writerow(np.concatenate([o,p,s]))
            conn.flush()

    def simulation_adaptor(self,sim):
        return sim

class MultiGridWriter(TableWriter):

    def __init__(self,obj,params,param_labels,sim,connection,gridcells):

        super(MultiGridWriter,self).__init__(obj,params,param_labels,sim,connection)
        #import pdb; pdb.set_trace()

        self.sim_labels = [f"sim_{idataset}_{gridcells[igrid]}" for idataset,dataset in enumerate(self.sim[0]) for igrid in range(len(dataset))]
        
        self.sim_length = len(self.sim[0][0][0])

        self.sim_labels = np.repeat(self.sim_labels,self.sim_length).tolist()

        self._create_header()



    def simulation_adaptor(self,sim):

        sim_stacked = np.array(sim)

        #import pdb; pdb.set_trace()

        if len(sim_stacked.shape)<4:
            sim_stacked = sim_stacked[None]

        sim_converted = sim_stacked.reshape((len(sim),sim_stacked.shape[1]*sim_stacked.shape[2]*sim_stacked.shape[3]))

        return sim_converted
