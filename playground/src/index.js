import _ from 'lodash';

import * as d3 from 'd3';
import { showDenseConnections, plotHist } from './helper.js';

var data = require('./data.json');

const width = 1080;
const height = 520;


const svg = d3.select('svg');
svg.style('width', width);
svg.style('height', height);
svg.style('background-color', 'rgba(255, 0, 0, 0.2)');

// temp css to center
svg.style('display', 'block');
svg.style('margin', 'auto');


// input image
svg.append('rect')
  .attr('transform', 'translate(10, 368)')
  .attr('width', 64)
  .attr('height', 64)
  .attr('stroke',"blue")
  .attr('fill', '#FFF')

// const imgCanvas = d3.select('#inputImg').append('canvas')
//   .attr("width", 100)
//   .attr("height", 100)
//
//
// let context = imgCanvas.node().getContext("2d");
//
// let dx = data[0].length;
// let dy = data.length;
// let image = context.createImageData(dx, dy);
//
// for (let x = 0, p = -1; x < dx; ++x) {
//   for (let y = 0; y < dy; ++y) {
//     let value = data[x][y];
//     image.data[++p] = value;
//     image.data[++p] = value;
//     image.data[++p] = value;
//     image.data[++p] = 160;
//   }
// }
//
// console.log(image)
// context.putImageData(image, 0, 0);


// --- initialization ---
const plotData = [
  {'lineColor':'#000', 'layerColor':'#F62', 'boxs':2, 'pooling':['i']},
  {'lineColor':'#F62', 'layerColor':'#F20', 'boxs':2, 'pooling':['i']},
  {'lineColor':'#F20', 'layerColor':'#0E4', 'boxs':4, 'pooling':['2']},
  {'lineColor':'#0E4', 'layerColor':'#0A0', 'boxs':4, 'pooling':['2','i']},
  {'lineColor':'#0A0', 'layerColor':'#FF8', 'boxs':8, 'pooling':['4','2']},
  {'lineColor':'#FF8', 'layerColor':'#FD8', 'boxs':8, 'pooling':['4','2','i']}
];

var denseCurveStarts = [];
var denseCurveEnds = [];

const l1 = 30;
const l2 = 10;
const d = 4;

let x = 74;
let y = 380;

// define arrowhead marker
const arrowPoints = [[0, 0], [0, 10], [10, 5]];
svg.append('defs')
  .append('marker')
  .attr('id', 'arrow')
  .attr('viewBox', [0, 0, 10, 10])
  .attr('refX', 5)
  .attr('refY', 5)
  .attr('markerWidth', 10)
  .attr('markerHeight', 10)
  // .attr('orient', 'auto-start-reverse')
  .attr("orient", "auto")
  .attr('markerUnits', 'userSpaceOnUse')
  .append('path')
  .attr('d', d3.line()([[0, 0], [0, 10], [10, 5]]));


// --- main network structure ---
plotData.forEach((data, i) => {

  const yPooling = y+10*(data['pooling'].length-1);
  denseCurveStarts.push({'layer':i, 'point':[x+10,yPooling], 'color': data['lineColor']});

  // box for each block
  if (i%2 == 0) {
    svg.append('rect')
      .attr('transform', 'translate('+(x+30)+','+(y-45)+')')
      .attr('width', 195 + 4*data['boxs'])
      .attr('height', 90)
      .attr("rx", 10)
      .attr("ry", 10)
      .attr('stroke', 'black')
      .style("stroke-dasharray", "4,4")
      .attr('fill', '#DDD');
  }

  svg.append('path')
    .attr('d', d3.line()([[x, yPooling], [i%2==0 ? x+l1+10 : x+l1, yPooling]]))
    .attr('marker-end', 'url(#arrow)')
    .attr('stroke', data['lineColor'])
    .attr('fill', 'none')
    .attr('stroke-width', '3px');
  x = x + l1 + 5 + (i%2==0 ? 10 : 0);

  // pooling rectangle
  data['pooling'].forEach((type, j) => {
    svg.append('rect')
      .attr('transform', 'translate('+(x)+','+(y-10*(data['pooling'].length-2*j))+')')
      .attr('width', 8)
      .attr('height', 20)
      .attr('stroke', 'black')
      .attr('fill', type=='i' ? '#48F' : (type=='2' ? '#8AF' : '#BBF'));
    denseCurveEnds.push({'layer':i, 'type':type, 'point':[x-5, y-10*(data['pooling'].length-2*j-1)]})
  });
  x = x + 8;

  svg.append('path')
    .attr('d', d3.line()([[x, y], [x+l2, y]]))
    .attr('marker-end', 'url(#arrow)')
    .attr('stroke', 'black')
    .attr('fill', 'none')
    .attr('stroke-width', '3px');
  x = x + l2 + 5;

  // layer rectangle
  y = y - 2*data['boxs'] + 2*Math.floor(i/2);
  for (var j = 0; j < data['boxs']; j++) {
    svg.append('rect')
      .attr('transform', 'translate('+(x)+','+(y-20)+')')
      .attr('width', 40-4*Math.floor(i/2))
      .attr('height', 40-4*Math.floor(i/2))
      .attr('stroke', 'black')
      .attr('fill', data['layerColor']);
    x = x + 4;
    y = y + 4;
  }
  x = x + 40 - 4 - 4*Math.floor(i/2);
  y = y - 2*data['boxs'] - 2*Math.floor(i/2);

});


// --- final global pooling ---
svg.append('path')
  .attr('d', d3.line()([[x, y+20], [x+l1, y+20]]))
  .attr('marker-end', 'url(#arrow)')
  .attr('stroke', '#FD8')
  .attr('fill', 'none')
  .attr('stroke-width', '3px');
x = x + l1 + 5;

['4','2','i'].forEach((type, i) => {
  svg.append('rect')
    .attr('transform', 'translate('+(x)+','+(y-10*(3-2*i))+')')
    .attr('width', 8)
    .attr('height', 20)
    .attr('stroke', 'black')
    .attr('fill', '#FFF');
  denseCurveEnds.push({'layer':6, 'type':type, 'point':[x-5, y-10*(3-2*i-1)]})
});
x = x + 8;


// --- output connection ---
svg.append('path')
  .attr('d', d3.line()([[x, y], [x+20, y]]))
  .attr('marker-end', 'url(#arrow)')
  .attr('stroke', 'black')
  .attr('fill', 'none')
  .attr('stroke-width', '3px');
x = x + l2 + 20 + 5;


// --- calculate dense connection curves ---
var denseConnections = [];
var idxCount = 0;
for (var i=0; i<plotData.length; i++) {
  for (var j=i+1; j<=plotData.length; j++) {
    const diff = Math.floor(j/2) - Math.floor(Math.abs(i-1)/2) - (j==6 ? 1 : 0);
    const type = diff == 0 ? 'i' : (diff == 1 ? '2' : '4');

    var c = {
      'id': idxCount++,
      'active': true,
      'startLayer':i,
      'endLayer': j,
      'type': type
    };
    c['color'] = denseCurveStarts.filter(d => d['layer']==c['startLayer'])[0]['color'];
    c['startPoint'] = denseCurveStarts.filter(d => d['layer']==c['startLayer'])[0]['point'];
    c['midPoint'] = [c['startPoint'][0]+40*(j-i), y-10-(j-i)*50]
    c['endPoint'] = denseCurveEnds.filter(d => d['layer']==c['endLayer'] && d['type']==c['type'])[0]['point'];
    denseConnections.push(c);
  }
}


// show dense connection curves
showDenseConnections(svg, denseConnections);



// for debug
console.log(denseConnections);






var histData = [
  {'category': 'airplane', 'value': 0.032},
  {'category': 'automobile', 'value': 0.009},
  {'category': 'bird', 'value': 0.83},
  {'category': 'cat', 'value': 0.12},
  {'category': 'deer', 'value': 0.3319},
  {'category': 'dog', 'value': 0.0003},
  {'category': 'frog', 'value': 0.0021},
  {'category': 'horse', 'value': 0.832},
  {'category': 'ship', 'value': 0.2},
  {'category': 'truck', 'value': 0.031}
];


const hist = svg.append('g')
  .attr('transform', 'translate(870,50)');

plotHist(hist, histData);
