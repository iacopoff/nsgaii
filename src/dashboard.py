from time import sleep
from callbacks import CallBack
import socketio
import json

class RecordEvolution(CallBack):
    """ Send data to a web application through web socket for live visualization.

    Methods can be added depending on the logical position in the algorithm.
    The method's name should match the function called in the algorithm class.

    Attributes
    ----------
    alg.problem.config.evalVar : list of str
        List of objective functions' label names
    
    alg.pop.F : ndarray, dims (population,objective functions)
        Numpy ndarray containing the objective functions from the evolution algorithm.

    Methods
    -------
    after_evolution
        Establish a connection with the flask server application and send data to socket namespace
        'conn'.

    Example
    ----

    """    
    def __init__(self):
        self._generation_count = 1

    def after_evolution(self):

        sio = socketio.Client()

        sio.connect('http://localhost:5000', namespaces=['/conn'])

        # this is just for testing the service
        try:
            labels = self.alg.problem.config.evalVar
        except:
            labels = ["obj1","obj2"]
        
        data = self.alg.pop.F #*-1 # convert to 1 is best


        payload = create_json_payload(data,labels,self._generation_count)

        #  The data can be of type str, bytes, dict, list or tuple. 
        #  When sending a tuple, the elements in it need to be of any of the other four allowed types.
        #  The elements of the tuple will be passed as multiple arguments to the server-side event handler function
        sio.emit('message', payload ,"/conn")

        sleep(0.1) # allows to disconnect without troubles

        self._generation_count += 1

        sio.disconnect()



def create_json_payload(data,labels,gen_count):

    p,o = data.shape
    
    l = {labels[k]:data[:,k].tolist() for k in range(o)}

    l["gen"] = [gen_count] *p

    return json.dumps(l)
