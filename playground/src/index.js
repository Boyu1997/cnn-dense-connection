import _ from 'lodash';

import * as d3 from 'd3';
import { getRandomId, calculateModelId, updateDenseConnectionCurves,
  plotImg, updateImg, plotHist, updateHist } from './helper.js';

const data = require('./data.json');
var dataIdx = getRandomId(data['inputs'].length);
var modelId = 0;

const width = 1080;
const height = 520;


const svg = d3.select('#d3')
  .style('width', width)
  .style('height', height);

// temp css to center
svg.style('display', 'block');
svg.style('margin', 'auto');


// input image
svg.append('rect')
  .attr('transform', 'translate(10, 316)')
  .attr('width', 128)
  .attr('height', 128)
  .attr('stroke', 'black')
  .attr('fill', '#FFF')

plotImg(data, dataIdx);


// --- initialization ---
const plotData = [
  {'lineColor':'#000', 'layerColor':'#F62', 'boxs':2, 'pooling':['i']},
  {'lineColor':'#F62', 'layerColor':'#F20', 'boxs':2, 'pooling':['i']},
  {'lineColor':'#F20', 'layerColor':'#0E4', 'boxs':4, 'pooling':['2']},
  {'lineColor':'#0E4', 'layerColor':'#0A0', 'boxs':4, 'pooling':['2','i']}
];

var denseCurveStarts = [];
var denseCurveEnds = [];

const l1 = 40;
const l2 = 10;
const d = 4;

let x = 138;
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
      .attr('transform', 'translate('+(x+40)+','+(y-45)+')')
      .attr('width', 212 + 4*data['boxs'])
      .attr('height', 90)
      .attr("rx", 10)
      .attr("ry", 10)
      .attr('stroke', 'black')
      .style("stroke-dasharray", "4,4")
      .attr('fill', '#DDD');
  }

  svg.append('path')
    .attr('d', d3.line()([[x, yPooling], [i%2==0 ? x+l1+20 : x+l1, yPooling]]))
    .attr('marker-end', 'url(#arrow)')
    .attr('stroke', data['lineColor'])
    .attr('fill', 'none')
    .attr('stroke-width', '3px');
  x = x + l1 + 5 + (i%2==0 ? 20 : 0);

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
  .attr('d', d3.line()([[x, y+10], [x+l1, y+10]]))
  .attr('marker-end', 'url(#arrow)')
  .attr('stroke', '#0A0')
  .attr('fill', 'none')
  .attr('stroke-width', '3px');
x = x + l1 + 5;

['2','i'].forEach((type, i) => {
  svg.append('rect')
    .attr('transform', 'translate('+(x)+','+(y-10*(2-2*i))+')')
    .attr('width', 10)
    .attr('height', 20)
    .attr('stroke', 'black')
    .attr('fill', '#FFF');
  denseCurveEnds.push({'layer':4, 'type':type, 'point':[x-5, y-10*(2-2*i-1)]})
});
x = x + 10;


// --- output connection ---
svg.append('path')
  .attr('d', d3.line()([[x, y], [x+45, y]]))
  .attr('marker-end', 'url(#arrow)')
  .attr('stroke', 'black')
  .attr('fill', 'none')
  .attr('stroke-width', '3px');
x = x + l2 + 45 + 5;


// --- calculate dense connection curves ---
var denseConnections = [];
var idxCount = 0;
for (var i=0; i<plotData.length; i++) {
  for (var j=i+1; j<=plotData.length; j++) {
    const diff = Math.floor(j/2) - Math.floor(Math.abs(i-1)/2) - (j==4 ? 1 : 0);
    const type = diff == 0 ? 'i' : '2';

    var c = {
      'id': idxCount++,
      'active': false,
      'startLayer':i,
      'endLayer': j,
      'type': type
    };
    c['color'] = denseCurveStarts.filter(d => d['layer']==c['startLayer'])[0]['color'];
    const startPoint = denseCurveStarts.filter(d => d['layer']==c['startLayer'])[0]['point'];
    const midPoint = [startPoint[0]+40*(j-i), y-10-(j-i)*50]
    const endPoint = denseCurveEnds.filter(d => d['layer']==c['endLayer'] && d['type']==c['type'])[0]['point'];
    c['path'] = [startPoint, midPoint, endPoint];
    denseConnections.push(c);
  }
}

// define dense connection curves
const lineGenerator = d3.line()
  .curve(d3.curveCatmullRom.alpha(0.3));

svg.selectAll()
   .data(denseConnections)
   .enter()
   .append('g')
   .append('path')
   .attr('class', 'denseConnection')
   .attr('d', function(d) {
     return lineGenerator(d.path);
   })
   .attr('fill', 'none')
   .attr('stroke', function(d) {
     return d.active ? d.color : '#CCC';
   })
   .attr('stroke-width', '3px')
   .attr('marker-end', 'url(#arrow)')
   .on('mouseover', function() {
     d3.select(this).attr("stroke-width", '6px');
   })
   .on('mouseout', function() {
     d3.select(this).attr("stroke-width", '3px');
   })
   .on('click', function(d) {
     d.active = !d.active;
     updateDenseConnectionCurves(svg, denseConnections);
     modelId = calculateModelId(denseConnections);
     updateHist(hist, data, dataIdx, modelId);
   });


const hist = svg.append('g')
  .attr('transform', 'translate(800,70)');

plotHist(hist, data, dataIdx, modelId);

document.getElementById('changeImageButton').onclick = function(){
  dataIdx = getRandomId(data['inputs'].length);
  updateImg(data, dataIdx);
  updateHist(hist, data, dataIdx, modelId);
};

document.getElementById('configSequentialButton').onclick = function(){
  denseConnections.forEach(c => c['active'] = false);
  updateDenseConnectionCurves(svg, denseConnections);
  modelId = calculateModelId(denseConnections);
  updateHist(hist, data, dataIdx, modelId);
};

document.getElementById('configDensenetButton').onclick = function(){
  denseConnections.forEach(c => {
    if (c['id']==0 || c['id']==1 || c['id']==4 || c['id']==9) {
      c['active'] = true;
    }
    else {
      c['active'] = false;
    }
  });
  updateDenseConnectionCurves(svg, denseConnections);
  modelId = calculateModelId(denseConnections);
  updateHist(hist, data, dataIdx, modelId);
};

document.getElementById('configCondensenetButton').onclick = function(){
  denseConnections.forEach(c => c['active'] = true);
  updateDenseConnectionCurves(svg, denseConnections);
  modelId = calculateModelId(denseConnections);
  updateHist(hist, data, dataIdx, modelId);
};

document.getElementById('more').onclick = function(){
  window.scrollTo({
    top: 600,
    left: 0,
    behavior: 'smooth'
  });
};
