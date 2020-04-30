import _ from 'lodash';

import * as d3 from 'd3';

function component() {
  const element = document.createElement('div');

  element.innerHTML = _.join(['Hello', 'webpack'], ' ');

  return element;
}

// document.body.appendChild(component());

var data = require('./data.json');

const width = 800;
const height = 600;


const svg = d3.select('svg');
svg.style('width', width);
svg.style('height', height);
svg.style('background-color', 'rgba(255, 0, 0, 0.2)');

// temp css to center
svg.style('display', 'block');
svg.style('margin', 'auto');


// input image group
const input = svg.append('g')
  .attr('transform', 'translate(20, 300)')

const imgDiv = input.append('rect')
  .attr('width', 100)
  .attr('height', 100)
  .attr('stroke',"blue")
  .attr('fill', '#FFF')

const imgCanvas = d3.select('#inputImg').append('canvas')
  .attr("width", 100)
  .attr("height", 100)


let context = imgCanvas.node().getContext("2d");

let dx = data[0].length;
let dy = data.length;
let image = context.createImageData(dx, dy);

for (let x = 0, p = -1; x < dx; ++x) {
  for (let y = 0; y < dy; ++y) {
    let value = data[x][y];
    image.data[++p] = value;
    image.data[++p] = value;
    image.data[++p] = value;
    image.data[++p] = 160;
  }
}

console.log(image)
context.putImageData(image, 0, 0);







// define the arrowhead marker
const arrowPoints = [[0, 0], [0, 10], [10, 5]];
svg.append('defs')
  .append('marker')
  .attr('id', 'arrow')
  .attr('viewBox', [0, 0, 10, 10])
  .attr('refX', 5)
  .attr('refY', 5)
  .attr('markerWidth', 10)
  .attr('markerHeight', 10)
  .attr('orient', 'auto-start-reverse')
  .attr('markerUnits', 'userSpaceOnUse')
  .append('path')
  .attr('d', d3.line()([[0, 0], [0, 10], [10, 5]]))
  .attr('stroke', 'black');


// define connection lines
const linePoints = [
  [[120, 350], [195, 350]],
  [[210, 350], [225, 350]],
];

linePoints.forEach(linePoint => {
  svg.append('path')
    .attr('d', d3.line()(linePoint))
    .attr('marker-end', 'url(#arrow)')
    .attr('stroke', 'black')
    .attr('fill', 'none')
    .attr('stroke-width', '3px');
});


// define dense-connection curves
const curvePoints = [
  [[160, 350], [180, 100], [500, 350]],
  [[160, 350], [180, 120], [400, 350]],
  [[160, 350], [180, 140], [300, 350]]
];
const lineGenerator = d3.line()
  .curve(d3.curveBasis);

curvePoints.forEach(curvePoint => {
  const pathData = lineGenerator(curvePoint);
  svg.append('path')
  	.attr('d', pathData)
    .attr('fill', 'none')
    .attr('stroke','#999')
    .attr('stroke-width', '3px')
    .attr('marker-end', 'url(#arrow)');
});


// define pooling boxs
const poolings = [
  {'fill':'#48F', 'data':[200, 340]}
];
poolings.forEach(pooling => {
  svg.append('rect')
    .attr('transform', 'translate('+pooling['data'][0]+','+pooling['data'][1]+')')
    .attr('width', 10)
    .attr('height', 20)
    .attr('stroke', 'black')
    .attr('fill', pooling['fill']);
});


// define layer boxs
const layers = [
  {'fill':'#F00', 'data':[230, 325]}
];
layers.forEach(layer => {
  svg.append('rect')
    .attr('transform', 'translate('+layer['data'][0]+','+layer['data'][1]+')')
    .attr('width', 50)
    .attr('height', 50)
    .attr('stroke', 'black')
    .attr('fill', layer['fill']);
});
