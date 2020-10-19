import csv
import numpy as np

def listify(l):
    if l is None: return None
    elif isinstance(l,list): return l
    else: return [l]


class Database():

    def __init__(self,driver,obj_function,param,simulation,connection,**kwargs):

        self._driver = driver

        self.obj_func = obj_function
        self.param = param
        self.sim = simulation
        self.connection = connection
        self._create_header()

    def init(self):
        self._driver.init(self.connection,self.header)

    def write(self):
        self._driver.write(self.obj_func,self.param,self.sim)

    def close(self):
        if hasattr(self._driver,'close'):
            self._driver.close()
        else:
            pass


    def _create_header(self):
        if hasattr(self._driver,'create_header'):
            self.header = self._driver.create_header(self.obj_func,self.param,self.sim)
        else:
            obj_func_lab = [f"obj_{i}" for i in range(self.obj_func.shape[1])]
            param_lab = [f"par_{i}" for i in range(self.param.shape[1])]
            sim_lab = [f"sim_{idataset}_{igrid}" for idataset,dataset in enumerate(self.sim) for igrid in range(len(dataset))]
            self.header = obj_func_lab + param_lab + sim_lab





class Driver:
    def __init__(self,attrs):
        self.attrs = listify(attrs) 

    def set(self,cls):
        if self.attrs:
            for attr in self.attrs:
                if hasattr(cls,attr):
                    self.__setattr__(attr,vars(cls)[attr])    


class HymodDriver(Driver):

    def __init__(self,param_lab,attrs=None):
        super().__init__(attrs)
        self.param_lab = param_lab


    def init(self,connection,header):

        self.connection = connection
        with open(connection,'w') as conn:
            writer = csv.writer(conn,delimiter=',')
            writer.writerow(header)
            conn.flush()

    def write(self,obj_func,param,sim):

        with open(self.connection,'a') as conn:
            writer = csv.writer(conn,delimiter=',')
            for o,p,s in zip(obj_func,param,sim):
                writer.writerow(np.concatenate([o,p,s]))
            conn.flush()





class VicDriverMultiGridcell:

    def __init__(self,gridcells,param_lab,connection):
        self.gridcells = gridcells
        self.param_lab = param_lab
        self.connection = connection


    def create_header(self,obj_func,param,sim):

        sim_length = len(sim[0][0][0])

        sim_lab = ["sim_{}_{}".format(idataset,self.gridcells[igrid]) for idataset,dataset in enumerate(sim[0]) for igrid in range(len(dataset))]
        sim_lab = np.repeat(sim_lab,sim_length).tolist()

        obj_func_lab = [f"obj_{i}" for i in range(obj_func.shape[1])]

        return obj_func_lab + self.param_lab + sim_lab



    def init(self,header):

        with open(self.connection +".csv",'w') as conn:
            writer = csv.writer(conn,delimiter=',')
            writer.writerow(header)
            conn.flush()

    def write(self,obj_func,param,sim):

        sim = self.simulation_adaptor(sim)

        #import pdb; pdb.set_trace()

        assert len(sim.shape) == 2, f"check simulation dimension, it is {sim.shape}, it should be (n_pop,sim_length)"

        with open(self.connection + ".csv",'a') as conn:
            writer = csv.writer(conn,delimiter=',')
            for o,p,s in zip(obj_func,param,sim):
                writer.writerow(np.concatenate([o,p,s]))
            conn.flush()

    def simulation_adaptor(self,sim):

        sim_stacked = np.array(sim)

        if len(sim_stacked.shape)<4:
            sim_stacked = sim_stacked[None]

        sim_converted = sim_stacked.reshape((len(sim),sim_stacked.shape[1]*sim_stacked.shape[2]*sim_stacked.shape[3]))

        return sim_converted
