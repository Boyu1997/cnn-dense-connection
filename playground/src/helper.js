import * as d3 from 'd3';

const histWidth = 230;
const histHeight = 360;

export function plotImg(data, dataIdx) {
  const imgCanvas = d3.select('#inputImg')
    .style('top', '316px')
    .style('left', '10px')
    .style('width', '128px')
    .style('height', '128px')
    .append('canvas')
    .attr('width', '128px')
    .attr('height', '128px')

  let context = imgCanvas.node().getContext("2d");

  let size = 32;
  let image = context.createImageData(4*size, 4*size);

  for (let x = 0, p = -1; x < size; ++x) {
    for (let i=0; i<4; i++) {
      for (let y = 0; y < size; ++y) {
        for (let j=0; j<4; j++) {
          image.data[++p] = data['inputs'][dataIdx][0][x][y];
          image.data[++p] = data['inputs'][dataIdx][1][x][y];
          image.data[++p] = data['inputs'][dataIdx][2][x][y];
          image.data[++p] = 255;
        }
      }
    }
  }

  context.putImageData(image, 0, 0);
}

export function calculateModelId(connections) {
  var modelId = 0;
  connections.forEach(c => {
    if (c['active']) {
      modelId += Math.pow(2, c['id']);
    }
  });
  return modelId;
}

function getHistData(data, dataIdx, modelId) {
  const outputs = data['models'].find(model => model['id'] == modelId)['outputs'][dataIdx];
  const histData = [
    {'category': 'truck', 'value': outputs[9]},
    {'category': 'ship', 'value': outputs[8]},
    {'category': 'horse', 'value': outputs[7]},
    {'category': 'frog', 'value': outputs[6]},
    {'category': 'dog', 'value': outputs[5]},
    {'category': 'deer', 'value': outputs[4]},
    {'category': 'cat', 'value': outputs[3]},
    {'category': 'bird', 'value': outputs[2]},
    {'category': 'automobile', 'value': outputs[1]},
    {'category': 'airplane', 'value': outputs[0]}
  ];
  return histData;
}


export function plotHist(svg, data, dataIdx, modelId) {
  const histData = getHistData(data, dataIdx, modelId);

  var x = d3.scaleLinear()
    .range([0, histWidth])
    .domain([0, 1]);

  var y = d3.scaleBand()
    .range([histHeight, 0])
    .padding(0.1)
    .domain(histData.map(function (d) {
      return d.category;
    }));

  var bars = svg.selectAll('.bar')
    .data(histData)
    .enter()
    .append('g');

  bars.append('rect')
    .attr('class', 'bar')
    .attr('x', 0)
    .attr('y', function(d) { return y(d.category); })
    .attr('width', function(d) {return x(d.value); })
    .attr('height', y.bandwidth())
    .attr('fill', '#33F');

  bars.append('text')
    .attr('class', 'text')
    .attr('x', function (d) {
      return x(d.value) + 6;
    })
    .attr("y", function (d) {
        return y(d.category) + y.bandwidth() / 2 + 4;
    })
    .text(function (d) {
        return (d.value*100).toFixed(1)+'%';
    })
    .style('font-size', '14px')
    .style('font-weight', '600');

  svg.append('g')
    .call(d3.axisLeft(y)
      .tickSize(0))
      .style('font-size', '18px')
      .style('font-weight', '800');
}

export function updateHist(svg, data, dataIdx, modelId) {
  const histData = getHistData(data, dataIdx, modelId);

  var x = d3.scaleLinear()
    .range([0, histWidth])
    .domain([0, 1]);

  var y = d3.scaleBand()
    .range([histHeight, 0])
    .padding(0.1)
    .domain(histData.map(function (d) {
      return d.category;
    }));

  svg.selectAll(".bar")
    .data(histData)
    .transition().duration(1000)
    .attr('y', function(d) { return y(d.category); })
    .attr('width', function(d) {return x(d.value); });

  svg.selectAll(".text")
    .data(histData)
    .transition().duration(1000)
    .attr('x', function (d) {
      return x(d.value) + 6;
    })
    .attr("y", function (d) {
        return y(d.category) + y.bandwidth() / 2 + 4;
    })
    .text(function (d) {
        return (d.value*100).toFixed(1)+'%';
    });
}
