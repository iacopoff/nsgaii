
$(document).ready(function(){
    //connect to the socket server.
    var socket = io.connect('http://' + document.domain + ':' + location.port + '/test');
    var datastore = Array(); // maybe create an object that reference generation number
    //var newdata;





    //receive details from server
    socket.on('newnumber', function(msg) {

        var received = JSON.parse(msg)

        var obj_func_names =Object.keys(received)// ["obj1","obj2"]//Object.keys(received).pop("gen")
        
        

        var newdata = Array.from(received.obj1,(d,i) => ({"gen":received["gen"][i],"obj1":received[obj_func_names[0]][i],"obj2":received[obj_func_names[1]][i]}))

        datastore.push(...newdata)

        console.log(newdata)
        console.log(datastore)
        console.log(received)


        scatterPlot2d(d3.select('#divPlot'),datastore)
        

    });

    
    
    //scatterPlot3d( d3.select('#divPlot'));
});