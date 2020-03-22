function computeSpeed( x ) 

	local nsmpl = x:size( 1 ) 
	local v = x:narrow( 1, 2, nsmpl-1 ):clone():add( -1, x:narrow( 1, 1, nsmpl-1 ) )
	local d = torch.Tensor( nsmpl-1 )
	for i = 1, nsmpl-1 do 

		d[i] = v[i]:norm() + 1e-8

		v[i]:div( d[i] ) 

	end

	return v, d 

end 

function computeDistCurvature( x ) 

	local nsmpl = x:size( 1 ) 
	local v, d  = computeSpeed( x ) 
	local c     = torch.Tensor( nsmpl - 2 )
	local aux   = torch.Tensor( v[1]:size() ):typeAs( v )

	for i = 1, nsmpl-2 do

		aux:copy( v[i] ):cmul( v[i+1] )
		local cos = aux:sum(); cos = math.min( cos, 1 ); cos = math.max( cos, -1 ) 
		c[i] = math.acos( cos ) 
		-- print( i, 180 * c[i] / math.pi ); 
	end 

	local e = x[1]:dist( x[nsmpl] ) / ( nsmpl-1 ) 

	return d, c, e  

end 