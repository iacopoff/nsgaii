
$(document).ready(function(){
    //connect to the socket server.
    var socket = io.connect('http://' + document.domain + ':' + location.port + '/test');
    var datastore = Array(); // maybe create an object that reference generation number
   

    var counter = 1

    // set the zooming
    var zoom = d3.zoom()
    .translateExtent([[ 0, 0 ],[ 800, 500 ]])
    .scaleExtent([ 1, 20 ])
    .on('zoom', (d,i) => {
    
      d3.select( "svg" )
        .select('.plot-container')
        .attr('transform', d3.event.transform)
    
    })


    //receive details from server
    socket.on('newnumber', function(msg) {

        var received = JSON.parse(msg)

        var obj_func_names =["obj1","obj2"]//Object.keys(received)// 
        
        

        var newdata = Array.from(received.obj1,(d,i) => ({"gen":received["gen"][i],"obj1":received[obj_func_names[0]][i],"obj2":received[obj_func_names[1]][i]}))

        datastore.push(...newdata)


        if (counter == 1) {

        scatterPlot2d(d3.select('#divPlot'),datastore,zoom)
        counter = counter + 1

        } else {

        console.log("updating")
        updateScatterPlot2d(datastore)


        }

        //zoom
        d3.select("svg")
        .select('.plot-container')
        .call( zoom )

    });

});