import * as d3 from 'd3';

export function plotImg(data) {
  const imgCanvas = d3.select('#inputImg')
    .style('top', '324px')
    .style('left', '10px')
    .style('width', '112px')
    .style('height', '112px')
    .append('canvas')
    .attr('width', '112px')
    .attr('height', '112px')

  let context = imgCanvas.node().getContext("2d");

  let dx = data[0].length;
  let dy = data.length;
  let image = context.createImageData(4*dx, 4*dy);

  for (let x = 0, p = -1; x < dx; ++x) {
    for (let i=0; i<4; i++) {
      for (let y = 0; y < dy; ++y) {
        let value = data[x][y];
        for (let j=0; j<4; j++) {
          image.data[++p] = value;
          image.data[++p] = value;
          image.data[++p] = value;
          image.data[++p] = 180;
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
      .attr('stroke',c['color'])
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


export function plotHist(svg, data) {
  const width = 280;
  const height = 360;

  data = data.sort(function (a, b) {
    return d3.descending(a.category, b.category);
  })

  var x = d3.scaleLinear()
    .range([0, width])
    .domain([0, d3.max(data, function (d) {
      return d.value;
    })]);

  var y = d3.scaleBand()
    .range([height, 0])
    .padding(0.1)
    .domain(data.map(function (d) {
      return d.category;
    }));

  var bars = svg.selectAll('.bar')
    .data(data)
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
