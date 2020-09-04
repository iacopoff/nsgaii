import re
from flask_socketio import SocketIO, emit
from flask import Flask, render_template, url_for, copy_current_request_context,session,g
from random import random
from time import sleep
from threading import Thread, Event
import numpy as np




app = Flask("/home/iff/research/dev/nsgaii/src/app/application")
app.instance_path = "/home/iff/research/dev/nsgaii/src/app"
app.root_path = "/home/iff/research/dev/nsgaii/src/app"

app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True

#turn the flask app into a socketio app
socketio = SocketIO(app, async_mode=None, logger=True, engineio_logger=True)


#@staticmethod
@app.route('/')
def index():
    #only by sending this page first will the client be connected to the socketio instance
    return render_template('index.html')




@socketio.on('message',namespace="/conn")
def handle_message(message):

    print(f"message from dashboard {message}")
    
    send = message

    socketio.emit('tofrontend', send, namespace='/conn')



class Dashboard:

    def __init__(self):
        self.app = app
        self.socketio = socketio





    def _run(self):
        self.socketio.run(self.app)



if __name__ == "__main__":
    rec = Dashboard()
    rec._run()
