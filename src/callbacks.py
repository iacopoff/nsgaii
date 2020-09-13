import re
from flask_socketio import SocketIO, emit
from flask import Flask, render_template, url_for, copy_current_request_context
from random import random
from time import sleep
from threading import Thread, Event
import numpy as np

class CallBack:
    
    """

    Base class to build callbacks.

    Attributes
    ----------

    alg : class 
        Algorithm class object, allows access to attributes and methods of the class

    name : str
        Name of the callback

    Example
    -------

    class PrintMutation(CallBack):
        
        def after_mutation(self):
            
            print(self.alg.pop.F)

            return

    
    """
    _order=0
    
    def set_algorithm(self, alg): self.alg = alg
    
    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return name or 'callback'



