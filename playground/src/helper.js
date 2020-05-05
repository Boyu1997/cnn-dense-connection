import * as d3 from 'd3';

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
  console.log(data['inputs'][dataIdx])

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

export function showDenseConnections(svg, denseConnections) {
  // define dense connection curves
  const lineGenerator = d3.line()
    .curve(d3.curveCatmullRom.alpha(0.3));

  denseConnections.forEach(c => {
    const path = []
    path.push(c['startPoint'])
    path.push(c['midPoint'])
    path.push(c['endPoint'])
    svg.append('path')
      .attr('d', lineGenerator(path))
      .attr('fill', 'none')
      .attr('stroke', c['active'] ? c['color'] : '#CCC')
      .attr('stroke-width', '3px')
      .attr('marker-end', 'url(#arrow)')
      .on('mouseover', function() {
        d3.select(this).attr("stroke-width", '6px');
      })
      .on('mouseout', function() {
        d3.select(this).attr("stroke-width", '3px');
      })
      .on('click', function(d) {
        const color = c['active'] ? '#CCC' : c['color'];
        c['active'] = !c['active'];
        d3.select(this).attr("stroke", color);
      });
  });
}


export function plotHist(svg, data, dataIdx, modelId) {
  const width = 230;
  const height = 360;
  const outputs = data['models'].find(model => model['id'] == modelId)['outputs'][dataIdx];
  var histData = [
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
  ]

  var x = d3.scaleLinear()
    .range([0, width])
    .domain([0, 1]);

  var y = d3.scaleBand()
    .range([height, 0])
    .padding(0.1)
    .domain(histData.map(function (d) {
      return d.category;
    }));

  var bars = svg.selectAll('.bar')
    .data(histData)
    .enter()
    .append("g")

  bars.append('rect')
    .attr('x', 0)
    .attr('y', function(d) { return y(d.category); })
    .attr('width', function(d) {return x(d.value); } )
    .attr('height', y.bandwidth())
    .attr('fill', '#33F')

  bars.append('text')
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
