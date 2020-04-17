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
  .attr('transform', 'translate(200, 200)')

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
