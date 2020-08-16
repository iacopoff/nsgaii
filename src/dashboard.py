import re
from random import random
from time import sleep
import numpy as np
from callbacks import CallBack
from functools import partial
import multiprocessing
import socketio
import time


class RecordEvolution(CallBack):
    


    def after_mutation(self):

        print("sending to server!")

        sio = socketio.Client()

        sio.connect('http://localhost:5000', namespaces=['/test'])


        msg = "pop number is: {}".format(self.alg.pop.n_pop)

        sio.emit('my message', msg ,"/test")

        time.sleep(0.1)
        sio.disconnect()


#threading.Thread(target=server,args=()).start()
#job = multiprocessing.Process(target=Record(),args=(),daemon=True)
#job.start()
#job.join()