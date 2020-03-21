require 'nn'
require 'nngraph'
require 'common-torch/utilities/gnuplot' 

function multiscaleBlurKernel( p ) 

	local x = torch.Tensor( 2*p + 1 ):zero(); x[p+1] = 1
	x = x:view( 2*p+1, 1 )

	local k = torch.Tensor({ 1, 1 }):view( 2, 1 ) 

	for i = 1, p do 

		x = torch.conv2( x, k, 'F' )
		x = torch.conv2( x, k, 'V' )

	end

	x:div( x:sum() ) 
	x = x:squeeze()

	return x 

end

function multiscaleConv( p, nsmpl ) 

	local blur = nn.SpatialConvolution( 1, 1, 2*p+1, 1, 2, 1 )
	blur.weight:copy( multiscaleBlurKernel( p ) ) 
	blur.bias:zero()

	local network = nn.Sequential()
					:add( nn.View( nsmpl, -1 ) ) 
					:add( nn.Transpose( {1,2} ) )
					:add( nn.Contiguous() ) 
					:add( nn.Unsqueeze( 2 ) ) 
					:add( nn.Unsqueeze( 2 ) ) 
					:add( nn.SpatialReplicationPadding(  1,   1, 0, 0) )
					:add( nn.SpatialReflectionPadding( p-1, p-1, 0, 0) )
					:add( blur ) 
					:add( nn.Select( 2, 1 ) ) 
					:add( nn.Select( 2, 1 ) ) 
					:add( nn.Transpose( {1,2} ) )
					:add( nn.Contiguous() )

	return network 

end

function multiscaleMatrix( p, nsmpl ) 

	local dim  = 1 
	local network = multiscaleConv( p, nsmpl )

	local x = torch.randn( nsmpl, dim )
	local y = network:updateOutput( x )
	local z = torch.Tensor( y:size(1), nsmpl ) 

	for i = 1, y:size(1) do 

		y:zero(); y[i][1] = 1 
		z[i]:copy( network:updateGradInput( x, y ) )

	end

	return z 

end

function multiscaleFwd( p, nsmpl ) 

	local blurMatrix = multiscaleMatrix( p, nsmpl ) 
	local blur = nn.Linear( blurMatrix:size(2), blurMatrix:size(1) ) 
	blur.weight:copy( blurMatrix ) 
	blur.bias:zero()

	local network = blur 

	-- local network = nn.Sequential()
	-- 				:add( nn.View( nsmpl, -1 ) ) 
	-- 				:add( nn.Transpose( {1,2} ) )
	-- 				:add( nn.Contiguous() ) 
	-- 				:add( blur ) 
	-- 				:add( nn.Transpose( {1,2} ) )
	-- 				:add( nn.Contiguous() )

	return network 

end

function multiscaleBkw( p, nsmpl ) 

	local blurMatrix = multiscaleMatrix( p, nsmpl ) 
	local blur       = nn.Linear( blurMatrix:size(1), blurMatrix:size(2) ) 
	blur.weight:copy( blurMatrix:t() )
	blur.bias:zero()

	local network = blur 

	-- local network = nn.Sequential()
	-- 				:add( nn.View( math.ceil(nsmpl/2), -1 ) ) 
	-- 				:add( nn.Transpose( {1,2} ) )
	-- 				:add( nn.Contiguous() ) 
	-- 				:add( blur ) 
	-- 				:add( nn.Transpose( {1,2} ) )
	-- 				:add( nn.Contiguous() )

	return network 

end

-- function multiscaleAnalysis( p, nsmpl ) 

-- 	local inode = nn.Identity()()

-- 	local lp = multiscaleFwd( p, nsmpl )( inode ) 
-- 	local hp = nn.CSubTable()({ inode, multiscaleBkw( p, nsmpl )( lp )})

-- 	local output = { lp, hp } 

-- 	local network = nn.gModule({ inode }, output ) 

-- 	return network

-- end

-- local z = multiscaleMatrix( 3, 11 ) 
-- for i =1, 6 do print( z[i]:sum() ) end 
-- print( z ) 

-- function multiscaleBkw( p, nsmpl ) 

-- 	local blur = nn.SpatialFullConvolution( 1, 1, 2*p+1, 1, 2, 1 )
-- 	blur.weight:copy( multiscaleBlurKernel( p ) ) 
-- 	blur.bias:zero()

-- 	local inode = nn.Identity()()

-- 	local output = nn.View( math.ceil(nsmpl/2), -1 )( inode ) 
-- 	output = nn.Transpose( {1,2} )( output )
-- 	output = nn.Contiguous()( output ) 
-- 	output = nn.Unsqueeze( 2 )( output ) 
-- 	output = nn.Unsqueeze( 2 )( output ) 
-- 	output = blur( output ) 
-- 	output = nn.SpatialReplicationPadding(  1,   1, 0, 0 )( output )
-- 	output = nn.SpatialReflectionPadding( p-1, p-1, 0, 0 )( output ) 
-- 	output = nn.Select( 2, 1 )( output ) 
-- 	output = nn.Select( 2, 1 )( output ) 

-- 	local o1 = nn.Narrow( -1, 1 ,   p            )( output ) 
-- 	local o2 = nn.Narrow( -1, 1 + 2*p , nsmpl    )( output )  
-- 	local o3 = nn.Narrow( -1, 1 + 3*p + nsmpl, p )( output )  
-- 	o1 = nn.CAddTable()({ o1, nn.Narrow( -1,         1, p )( o2 ) })
-- 	o3 = nn.CAddTable()({ o3, nn.Narrow( -1, 1+nsmpl-p, p )( o2 ) })
-- 	o2 = nn.Narrow( -1, 1+p, nsmpl-2*p )( o2 ) 
-- 	output = nn.JoinTable(-1)({ o1, o2, o3 })
-- 	output = nn.Transpose( {1,2} )( output )
-- 	output = nn.Contiguous()( output ) 

-- 	local network = nn.gModule( { inode }, { output } )

-- 	return network 

-- end

function testMultiscale()

	local nsmpl = 11 
	local dim   = 1 

	local p     = 3 
	local netF = multiscaleFwd(  p, nsmpl )
	local netB = multiscaleBkw( p, nsmpl )

	local x  = torch.randn( dim, nsmpl )
	local y  = netF:updateOutput( x )
	local z1 = netF:updateGradInput( x, y ) 
	local z2 = netB:updateOutput( y )

	print( 'transpose error', z1:dist( z2 ) ) 

end
-- testMultiscale()