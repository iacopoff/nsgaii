import re
from random import random
from time import sleep
import numpy as np
from callbacks import CallBack
from functools import partial
import multiprocessing
import socketio
import time
import json

class RecordEvolution(CallBack):
    """ Allows to send any data to a web application for live visualization.

    Methods can be added that mirror calls in the algorithm class

    Attributes
    ----------
    alg.problem.config.evalVar : list of str
        List of objective functions' label names
    
    alg.pop.F : ndarray
        Numpy ndarray containing the objective functions from the evolution algorithm

    Methods
    -------
    after_evolution
        Establish a connection with the server application and send data to namespace /test

    Example
    ----

    """    
    def __init__(self):
        self._generation_count = 1

    def after_evolution(self):

        sio = socketio.Client()

        sio.connect('http://localhost:5000', namespaces=['/test'])

        labels = self.alg.problem.config.evalVar
        labels = ["obj1","obj2"]
        msg = self.alg.pop.F *-1 # convert to 1 is best

        payload = obj_function_pop_to_json(msg,labels,self._generation_count)

        #  The data can be of type str, bytes, dict, list or tuple. 
        #  When sending a tuple, the elements in it need to be of any of the other four allowed types.
        #  The elements of the tuple will be passed as multiple arguments to the server-side event handler function
        sio.emit('my message', payload ,"/test")

        time.sleep(0.1) # allows to disconnect without troubles

        self._generation_count += 1

        sio.disconnect()



def obj_function_pop_to_json(data,labels,gen_count):
    # expect a numpy ndarray (n_pop x n_objects)
    # whant a json
    p,o = data.shape
    
    l = {labels[k]:data[:,k].tolist() for k in range(o)}

    l["gen"] = [gen_count] *p

    return json.dumps(l)
#threading.Thread(target=server,args=()).start()
#job = multiprocessing.Process(target=Record(),args=(),daemon=True)
#job.start()
#job.join()