import re
from flask_socketio import SocketIO, emit
from flask import Flask, render_template, url_for, copy_current_request_context,session,g
from random import random
from time import sleep
from threading import Thread, Event
import numpy as np




app = Flask("/home/iff/research/dev/nsgaii/src/async_flask/application")
app.instance_path = "/home/iff/research/dev/nsgaii/src/async_flask"
app.root_path = "/home/iff/research/dev/nsgaii/src/async_flask"

app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True

#turn the flask app into a socketio app
socketio = SocketIO(app, async_mode=None, logger=True, engineio_logger=True)

#random number Generator Thread
# thread = Thread()
# thread_stop_event = Event()



#@staticmethod
@app.route('/')
def index():
    #only by sending this page first will the client be connected to the socketio instance
    return render_template('index.html')

# @socketio.on('connect', namespace='/test')
# def test_connect():
    
#     global thread
#     global passaround

#     print('Client connected')
#     #socketio.emit('newnumber', {'number': passaround}, namespace='/test')
#     #Start the random number generator thread only if the thread has not been started before.
#     # if not thread.isAlive():
#     #     print("Starting Thread")
#     #     thread = socketio.start_background_task(after_mutation)



@socketio.on('my message',namespace="/test")
def handle_message(message):

    print("from client")
    msg = message.decode()
    print(msg)
    #global passaround
    #global thread
    
    array = np.fromstring(msg,np.float32)
    #if not thread.isAlive():
    #    print("Starting Thread")
    #     thread = socketio.start_background_task(after_mutation)
    socketio.emit('newnumber', {'number': array}, namespace='/test')




# def after_mutation():

#     global passaround

#     while not thread_stop_event.isSet() and passaround:

#         print(passaround)

#         if passaround == "stop":
#             pass
#         else:
#             socketio.emit('newnumber', {'number': passaround}, namespace='/test')


class Dashboard:

    def __init__(self):
        self.app = app
        self.socketio = socketio
        # self.thread = thread
        # self.thread_stop_event = thread_stop_event




    def _run(self):
        self.socketio.run(self.app)



if __name__ == "__main__":
    rec = Dashboard()
    rec._run()