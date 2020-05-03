import * as d3 from 'd3';

export function showDenseConnections(svg, denseConnections) {
  // define dense connection curves
  const lineGenerator = d3.line()
    .curve(d3.curveCatmullRom.alpha(0.6));

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
      .on("click", function(d) {
        const color = c['active'] ? '#CCC' : c['color'];
        c['active'] = !c['active'];
        d3.select(this).attr("stroke", color);
      });
  });
}
