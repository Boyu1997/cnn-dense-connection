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
const height = 400;


const svg = d3.select('svg');
svg.style('width', width);
svg.style('height', height);
svg.style('background-color', 'rgba(255, 0, 0, 0.2)');

// temp css to center
svg.style('display', 'block');
svg.style('margin', 'auto');


// input image group
svg.append('rect')
  .attr('transform', 'translate(20, 260)')
  .attr('width', 80)
  .attr('height', 80)
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





const plotData = [
  {'fill':'#F62', 'boxs':2},
  {'fill':'#F20', 'boxs':2},
  {'fill':'#0E4', 'boxs':4},
  {'fill':'#0A0', 'boxs':4},
  {'fill':'#FF8', 'boxs':6},
  {'fill':'#FD8', 'boxs':6}
]

const lineGenerator = d3.line()
  .curve(d3.curveBasis);

const l1 = 30;
const l2 = 10;
const d = 4;

let x = 100;
let y = 300;

plotData.forEach(data => {
  svg.append('path')
    .attr('d', d3.line()([[x, y], [x+l1, y]]))
    .attr('marker-end', 'url(#arrow)')
    .attr('stroke', 'black')
    .attr('fill', 'none')
    .attr('stroke-width', '3px');
  x = x + l1 + 5;

  const temp = 580;
  svg.append('path')
  	.attr('d', lineGenerator([[x-20, y], [x+10, 100], [temp, y]]))
    .attr('fill', 'none')
    .attr('stroke','#999')
    .attr('stroke-width', '3px')
    .attr('marker-end', 'url(#arrow)');


  svg.append('rect')
    .attr('transform', 'translate('+(x)+','+(y-10)+')')
    .attr('width', 10)
    .attr('height', 20)
    .attr('stroke', 'black')
    .attr('fill', '#48F');
  x = x + 10;

  svg.append('path')
    .attr('d', d3.line()([[x, y], [x+l2, y]]))
    .attr('marker-end', 'url(#arrow)')
    .attr('stroke', 'black')
    .attr('fill', 'none')
    .attr('stroke-width', '3px');
  x = x + l2 + 5;

  y = y - 2*data['boxs'];
  for (var i = 0; i < data['boxs']; i++) {
    svg.append('rect')
      .attr('transform', 'translate('+(x)+','+(y-20)+')')
      .attr('width', 40)
      .attr('height', 40)
      .attr('stroke', 'black')
      .attr('fill', data['fill']);
    x = x + 4;
    y = y + 4;
  }
  x = x + 40 - 4;
  y = y - 2*data['boxs'];
});

// // define dense-connection curves
// const curvePoints = [
//   [[130, 300], [180, 100], [500, 350]],
//   [[130, 300], [180, 120], [400, 350]],
//   [[130, 300], [180, 140], [300, 350]]
// ];
//
// curvePoints.forEach(curvePoint => {
//   const pathData = lineGenerator(curvePoint);
//   svg.append('path')
//   	.attr('d', pathData)
//     .attr('fill', 'none')
//     .attr('stroke','#999')
//     .attr('stroke-width', '3px')
//     .attr('marker-end', 'url(#arrow)');
// });
