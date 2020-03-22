
require 'gnuplot'
-- require 'utilities/scatter3'

function insertLabel( multiPackage, label, location )

	local plot   = { torch.Tensor({-666}), torch.Tensor({-666}), 'with dots point pointtype 6 ps 2 lc rgb "white"' }
	local params = { size  = '0.1,0.1',	origin = location , raw = {'unset border', 'unset xtics', 'unset ytics', 'unset label', 'set label "' .. label .. '" at 0,0 left font "Helvetica,24"'} }
	local pkg = { plot = plot, params = params }

	table.insert( multiPackage, pkg ) 

end

function insertTransparency( to, from, transparency )

	for i = 1, transparency:size(1) do 

		local str = from[3]:gsub( 'solid border lc rgb "white"', 'transparent solid ' .. transparency[i] .. ' noborder' )
		table.insert( to, {from[1]:narrow(1,i,1), from[2]:narrow(1,i,1), str } )

	end

end

gnuplot.colors = {'blue_050', 'green_050 ', 'red_050   ', 'brown_050 ', 'blue_025  ', 'green_025 ', 'red_025   ', 'brown_025 ', 'blue_075  ', 'green_075 ', 'red_075   ', 'brown_075 ' , 'blue_100  ', 'green_100 ', 'red_100   ', 'brown_100 '}

function gnuplot.plotPackage( dir, pkg ) 

	local plot = pkg.plot 
	local pkg  = pkg.params 

	local raw = {}
	if pkg.raw then for i = 1, #pkg.raw do table.insert( raw, pkg.raw[i] ) end end

	if pkg.xtics  then table.insert( raw, 'set xtics out nomirror ' .. pkg.xtics )                 end 
	if pkg.ytics  then table.insert( raw, 'set ytics out nomirror ' .. pkg.ytics )                 end 

	if pkg.offset then table.insert( raw, 'set offsets ' .. pkg.offset ) end 


	gnuplot.savePlot( dir, plot, pkg.xlabel, pkg.ylabel, pkg.axis, raw ) 

end 

function gnuplot.multiPackage( dir, pkg, raw ) 
-- function gnuplot.multiPackage( dir, pkg, size ) 

	-- local font = ' font "Helvetica,10"'
	-- local font = ' font "Helvetica,6"'
	local font = ' font "Helvetica,12"'

	-- local size = size or { 6, 4 } 

	if dir:find( '.png' ) then 

		gnuplot.setterm('png')
		gnuplot.pngfigure( dir )

	elseif dir:find( '.eps' ) then 

		-- gnuplot.raw('set terminal postscript eps enhanced solid font "Helvetica, 12"') --enhanced color 
		-- gnuplot.raw('set terminal postscript eps enhanced solid font "Helvetica, 6"') --enhanced color 
		gnuplot.epsfigure( dir ) 
		gnuplot.raw('set terminal postscript eps size 6in,4in' .. font) --enhanced color 
		-- gnuplot.raw('set terminal postscript eps size ' .. size[1] .. 'in,' .. size[2] .. 'in' .. font) --enhanced color 
		-- gnuplot.raw('set terminal postscript eps size 7in,4in' .. font) --enhanced color 

	end
	gnuplot.raw("set output '" .. dir .. "'") -- set output before multiplot
	gnuplot.raw('set multiplot' )

	-- gnuplot.raw('set tics' .. font )

	for i = 1, #pkg do 

		local plot = pkg[i].plot 
		local pkg  = pkg[i].params

		gnuplot.raw('set origin ' .. (pkg.origin or '') )
		gnuplot.raw('set size ' .. (pkg.size or '') )
		gnuplot.raw('set offsets ' .. (pkg.offset or '') )
		gnuplot.raw('set xtics out nomirror ' .. (pkg.xtics or ''))
		gnuplot.raw('set ytics out nomirror ' .. (pkg.ytics or ''))
		gnuplot.raw('set border 3 lw 1' )
		gnuplot.raw('set xlabel "' .. (pkg.xlabel or '') .. '" offset '   .. ( pkg.xlabeloffset or 0 ) .. ',0.5')
		gnuplot.raw('set ylabel "' .. (pkg.ylabel or '') .. '" offset 1,' .. ( pkg.ylabeloffset or 0 ) )

		if pkg.raw then for i = 1, #pkg.raw do gnuplot.raw( pkg.raw[i] ) end end

		if     type( plot ) == 'table'    then 

			gnuplot.plot( plot )

		elseif type( plot ) == 'userdata' then 

			gnuplot.imagesc( plot ) 

		end

	end

	-- if raw then for i = 1, #raw do gnuplot.raw( raw[i] ) end end 

	gnuplot.raw('unset multiplot')

end

function gnuplot.saveMulti( dir, plot, xlabel, ylabel, xtics, ytics, offsets, sizes, origins, raw )

	if dir:find( '.png' ) then 

		gnuplot.setterm('png')
		gnuplot.pngfigure( dir )

	elseif dir:find('.pdf') then 

		gnuplot.raw('set terminal pdfcairo enhanced color font "Helvetica, 12" ') 
		gnuplot.pdffigure( dir ) 

	elseif dir:find( '.eps' ) then 

		gnuplot.raw('set terminal postscript eps enhanced color solid font "Helvetica, 12"') --enhanced color 
		gnuplot.epsfigure( dir ) 

	end
	gnuplot.raw("set output '" .. dir .. "'") -- set output before multiplot
	gnuplot.raw('set multiplot' )

	gnuplot.raw('set origin ' .. origins[1])
	gnuplot.raw('set size ' .. sizes[1] )
	gnuplot.raw('set offsets ' .. offsets[1] .. ', 0, 0, ' .. offsets[1])
	gnuplot.raw('set xtics out nomirror 0,' .. xtics[1] )
	gnuplot.raw('set border 3' )
	gnuplot.raw('set xlabel "' .. xlabel[1] .. '"')
	gnuplot.raw('set ylabel "' .. ylabel[1] .. '"')
	gnuplot.raw('set ytics out nomirror 0,' .. ytics[1] )
	for i = 1, #raw[1] do gnuplot.raw( raw[3][i] ) end 
	gnuplot.plot( plot[1] ) 

	gnuplot.raw('set origin ' .. origins[2])
	gnuplot.raw('set size ' .. sizes[2] )
	gnuplot.raw('set offsets ' .. offsets[2] .. ', 0, 0, ' .. offsets[2])
	gnuplot.raw('set xtics out nomirror 0,' .. xtics[2] )
	gnuplot.raw('set border 3' )
	gnuplot.raw('set xlabel "' .. xlabel[2] .. '"')
	gnuplot.raw('set ylabel "' .. ylabel[2] .. '"')
	gnuplot.raw('set ytics out nomirror 0,' .. ytics[2] )
	for i = 1, #raw[2] do gnuplot.raw( raw[3][i] ) end 
	gnuplot.plot( plot[2] ) 

	gnuplot.raw('set origin ' .. origins[3] )
	gnuplot.raw('set size ' .. sizes[3] )
	gnuplot.raw('set offsets ' .. offsets[3] .. ', 0, 0, ' .. offsets[3])
	gnuplot.raw('set xtics out nomirror 0,' .. xtics[3] )
	gnuplot.raw('set border 3' )
	gnuplot.raw('set xlabel "' .. xlabel[3] .. '"')
	gnuplot.raw('set ylabel "' .. ylabel[3] .. '"')
	gnuplot.raw('set ytics out nomirror 0,' .. ytics[3] )
	gnuplot.raw('set style circle radius 0.02' )
	for i = 1, #raw[3] do gnuplot.raw( raw[3][i] ) end 
	gnuplot.plot( plot[3] ) 

	gnuplot.raw('unset multiplot')

end
--[[
function gnuplot.saveMulti( dir, plot, xlabel, ylabel, axis, xtics, ytics )

	local xoffset = 0.05
	local yoffset = 0.2
	local ySize1 =   2             
	local ySize2 = ( 2 + yoffset + 0.25 ) 
	local ySizeTotal = ySize1 + ySize2 
	local ySize1 = ySize1 / ySizeTotal
	local ySize2 = ySize2 / ySizeTotal

	if dir:find( '.png' ) then 

		gnuplot.setterm('png')
		gnuplot.pngfigure( dir )

	elseif dir:find( '.eps' ) then 

		gnuplot.raw('set terminal postscript eps enhanced solid font "Helvetica, 12"') --enhanced color 
		gnuplot.epsfigure( dir ) 

	end
	gnuplot.raw("set output '" .. dir .. "'") -- set output before multiplot
	gnuplot.raw('set multiplot' )

	gnuplot.raw('set xrange [' .. axis[1][1] .. ':' .. axis[1][2] .. ']')
	gnuplot.raw('set yrange [' .. axis[1][3] .. ':' .. axis[1][4] .. ']')
	gnuplot.raw('set origin 0,0.55' )
	gnuplot.raw('set size 1,' .. ySize1 )
	gnuplot.raw('set offsets graph ' .. xoffset .. ', 0, 0, 0')
	gnuplot.raw('unset xtics')
	gnuplot.raw('set border 2' )
	gnuplot.raw('set ylabel "' .. ylabel[1] .. '"')
	gnuplot.raw('set ytics out nomirror ' .. ytics[1] )

	gnuplot.plot( plot[1] ) 

	gnuplot.raw('set xrange [' .. axis[2][1] .. ':' .. axis[2][2] .. ']')
	gnuplot.raw('set yrange [' .. axis[2][3] .. ':' .. axis[2][4] .. ']')
	gnuplot.raw('set origin 0,0' )
	gnuplot.raw('set size 1,' .. ySize2 )
	gnuplot.raw('set offsets graph ' .. xoffset .. ', 0, 0, ' .. yoffset )
	gnuplot.raw('set xtics out nomirror ' .. xtics )
	gnuplot.raw('set border 3' )
	gnuplot.raw('set xlabel "' .. xlabel .. '"')
	gnuplot.raw('set ylabel "' .. ylabel[2] .. '"')
	gnuplot.raw('set ytics out nomirror ' .. ytics[2] )

	gnuplot.plot( plot[2] ) 

	gnuplot.raw('unset multiplot')

end
]]
function histRaw( hist )

	local xaxis = torch.Tensor( #hist )
	local yaxis = torch.Tensor( #hist )

	for i = 1, #hist do 

		xaxis[i] = hist[i].val
		yaxis[i] = hist[i].nb

	end

	return xaxis, yaxis 

end 

function nuFig( reset )

	if reset or not figCounter then 

		figCounter = reset 

	else 

		figCounter = figCounter + 1 

	end 

	gnuplot.figure( figCounter )
end 

function gnuplot.axisScat( xr )

	local m1 = xr[1]:min(); M1 = xr[1]:max()
	local m2 = xr[2]:min(); M2 = xr[2]:max()
	local r1 = ( M1 - m1 ) / 2; 
	local r2 = ( M2 - m2 ) / 2;
	local range = 1.1*math.max(r1,r2)
	local x1 = (M1+m1)/2; x2 = (M2+m2)/2

	gnuplot.axis{x1-range, x1+range, x2-range, x2+range}
end

function gnuplot.scatter( x, y, fig )

	local aux1 = torch.DoubleTensor( x:nElement() ):typeAs( x ):copy( x )
	local aux2 = torch.DoubleTensor( y:nElement() ):typeAs( y ):copy( y )

	if fig then 

		gnuplot.figure( fig ) 

	else

		nuFig()

	end 

	gnuplot.plot( aux1, aux2, '+' )

end

function gnuplot.savePlot( dir, plot, xlabel, ylabel, axis, raw, title )

	gnuplot.raw('set border 3' )
	gnuplot.raw('set ytics out nomirror') 
	gnuplot.raw('set xtics out nomirror')

	if axis   then gnuplot.axis(   axis   ) end
	if xlabel then gnuplot.xlabel( xlabel ) end 
	if ylabel then gnuplot.ylabel( ylabel ) end 
	if title  then gnuplot.title(  title  ) end 

	if     dir:find( '.eps' ) then 

		gnuplot.raw('set terminal postscript eps enhanced solid font "Helvetica, 20"') --enhanced color 
		-- gnuplot.raw('set terminal postscript eps enhanced solid font "Helvetica, 6"') --enhanced color 
		-- gnuplot.raw('set terminal postscript eps enhanced solid font "Helvetica, 12"') --enhanced color 
		-- gnuplot.raw('set terminal postscript eps solid font "Helvetica, 12"') --enhanced color 

	elseif dir:find('.pdf') then 

		gnuplot.raw('set term pdfcairo enhanced color font "Helvetica, 12" ') 
		
	elseif dir:find( '.png' ) then 

		gnuplot.raw('set term pngcairo font "Helvetica-Italic, 12" ')

	end 

	if raw then for i = 1, #raw do gnuplot.raw( raw[i] ) end end

	gnuplot.raw('set output "' .. dir  ..'"') 
	gnuplot.plot( plot )
	gnuplot.plotflush()
	
end 

function gnuplot.plotHist( data, mode, nbins, min, max )

	local h = gnuplot.hist( data, nbins, min, max ) 
	local x,y = histRaw( h ) 

	if     mode == 'counts'  then 

	elseif mode == 'density' then 

		y:div( (x[2]-x[1]) * y:sum() ) 

	else

		error( 'unknown histogram mode' )

	end

	return { x, y, '|' }

end 

function gnuplot.saveHist( dir, data, mode, xlabel, ylabel, axis, nbins, min, max )

	-- local h = gnuplot.hist( data, nbins, min, max )
	-- local x,y = histRaw( h ) 

	-- gnuplot.savePlot( dir, { x, y, '|' }, xlabel, ylabel, axis )
	gnuplot.savePlot( dir, gnuplot.plotHist( data, mode, nbins, min, max ), xlabel, ylabel, axis )
	
end 

function gnuplot.savePlot3Dscatter( dir, plot, xlabel, ylabel, zlabel, axis, raw, title )

	-- -- gnuplot.raw('set border '..1+4+16 )
	gnuplot.myRaw('set ztics out nomirror') 
	gnuplot.myRaw('set ytics out nomirror') 
	gnuplot.myRaw('set xtics out nomirror')

	gnuplot.myRaw('set size ratio -1')


	if axis   then gnuplot.myAxis(   axis   ) end
	-- if title  then gnuplot.title(  title  ) end 
	-- if xlabel then gnuplot.xlabel( xlabel ) end 
	-- if ylabel then gnuplot.ylabel( ylabel ) end 
	-- if zlabel then gnuplot.zlabel( zlabel ) end 

	-- if     dir:find( '.eps' ) then 
	-- 	gnuplot.raw('set terminal postscript eps enhanced solid font "Helvetica, 12"') 
	-- elseif dir:find( '.png' ) then 
	-- 	gnuplot.raw('set term pngcairo font "Helvetica-Italic, 12" ')
	-- end 
	-- -- if raw then for i = 1, #raw do gnuplot.raw( raw[i] ) end end
	-- gnuplot.raw('set output "' .. dir  ..'"') 

	-- gnuplot.figure()
	-- gnuplot.scatter3( plot )	
	-- gnuplot.plotflush()

	if     dir:find( '.eps' ) then 
		gnuplot.myRaw('set terminal postscript eps enhanced solid font "Helvetica, 12"') 
	elseif dir:find( '.png' ) then 
		gnuplot.myRaw('set term pngcairo font "Helvetica-Italic, 12" ')
	end 
	-- if raw then for i = 1, #raw do gnuplot.raw( raw[i] ) end end
	gnuplot.myRaw('set output "' .. dir  ..'"') 


	gnuplot.myScatter3( plot )	
	gnuplot.myPlotflush()
	
end 

function plot3Ddemo()

	z = torch.linspace(-2 * math.pi, 2 * math.pi)
	x = z:clone():cos()
	y = z:clone():sin()

	gnuplot.myFigure()

	gnuplot.savePlot3D( 'test.eps', {x,y,z}, nil, nil, nil, 'equal' )

end
-- plot3Ddemo()

function gnuplot.savePlot3Dcurve( dir, plot, xlabel, ylabel, zlabel, axis, raw, title )

	-- -- gnuplot.raw('set border '..1+4+16 )
	gnuplot.raw('set ztics out nomirror') 
	gnuplot.raw('set ytics out nomirror') 
	gnuplot.raw('set xtics out nomirror')

	if title  then gnuplot.title(  title  ) end 
	if xlabel then gnuplot.xlabel( xlabel ) end 
	if ylabel then gnuplot.ylabel( ylabel ) end 
	if zlabel then gnuplot.zlabel( zlabel ) end 

	if     dir:find( '.eps' ) then 
		gnuplot.raw('set terminal postscript eps enhanced solid font "Helvetica, 12"') 
	elseif dir:find( '.png' ) then 
		gnuplot.raw('set term pngcairo font "Helvetica-Italic, 12" ')
	end 

	if raw then for i = 1, #raw do gnuplot.raw( raw[i] ) end end

	gnuplot.raw('set output "' .. dir  ..'"') 

	local m, M = math.huge, -math.huge 

	for i = 1, #plot do 

		if type(plot[i]) == 'userdata' then 

			plot[i] = plot[i]:view(plot[i]:size(1),1)
			m = math.min(m, plot[i]:min())
			M = math.max(M, plot[i]:max())

		elseif type(plot[i]) == 'table' then 

			for j = 1, #plot[i] do 

				if type(plot[i][j]) == 'userdata' then 

					plot[i][j] = plot[i][j]:view(plot[i][j]:size(1),1)
					m = math.min(m, plot[i][j]:min())
					M = math.max(M, plot[i][j]:max())

				end 

			end 

		end 

	end

	local m = torch.Tensor(1,1):fill(m)-- math.min( x:min(), y:min(), z:min() ) )
	local M = torch.Tensor(1,1):fill(M)-- math.max( x:max(), y:max(), z:max() ) )
	local p = plot --{plot}
	table.insert( p, {m,m,m} )
	table.insert( p, {M,M,M} )


	-- gnuplot.splot( plot ) 
	gnuplot.splot( p ) 

	gnuplot.plotflush()

end

function splotDemo()

	local dir = 'splot.png'

	-- gnuplot.axis( 'equal' )
	-- gnuplot.raw( 'set view equal xyz' )

	n = 100 

	z = torch.linspace( -1, 1, n):mul( 10 ) --:view(n,1)
	x = z:clone():cos()
	y = z:clone():sin()
	local plot = { {x, y, z }, {x, y, z:clone():add(1)} } 

	-- local m = torch.Tensor(1,1):fill( math.min( x:min(), y:min(), z:min() ) )
	-- local M = torch.Tensor(1,1):fill( math.max( x:max(), y:max(), z:max() ) )
	-- table.insert( plot, {m,m,m} )
	-- table.insert( plot, {M,M,M} )

	gnuplot.figure(); gnuplot.savePlot3Dcurve( dir, plot, 'PC1', 'PC2', 'PC3')  --, nil, {'unset clabel', 'unset contour'} 

	-- gnuplot.splot( plot ) 

	-- gnuplot.plotflush()

end






