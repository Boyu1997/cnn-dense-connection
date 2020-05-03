import * as d3 from 'd3';

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
  const width = 160;
  const height = 380;

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
      return x(d.value) + 4;
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
      .style('font-size', '14px')
      .style('font-weight', '600');
}
