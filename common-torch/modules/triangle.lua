function lowerTriangle( dim ) 

	local cmul = nn.CMul( 1, dim, dim ) 
	cmul.weight:copy( torch.tril( torch.ones(dim,dim), -1 ) )

	local cadd = nn.CAdd( 1, dim, dim )
	cadd.bias:copy( torch.eye( dim ) ) 

	local inode = nn.Identity()()
	local output = cadd( cmul( inode ) )

	local network = nn.gModule({inode}, {output})

	return network 

end

--[[
function blockDiagonal( n )

	local cmul = nn.CMul( 1, n:sum(), n:sum() ) 
	cmul.weight:zero() 
	local ind = 1 

	for i = 1, n:size( 1 ) do 

		cmul.weight:narrow( 2, ind, n[i] ):narrow( 3, ind, n[i] ):fill( 1 ) 
		ind = ind + n[i] 

	end

	return cmul 

end

function lowerTriangleBlockwise( nsmpl, dim ) 

	local cmul = nn.CMul( 1, dim*nsmpl, dim*nsmpl ) 

	cmul.weight:zero()
	for i = 1, nsmpl do 

		local ind = 1 + (i-1)*dim
		local block = cmul.weight:narrow( 2, ind, dim ):narrow( 3, ind, dim )
		block:copy( torch.tril( torch.ones(dim,dim), -1 ) )

	end

	local cadd = nn.CAdd( 1, dim*nsmpl, dim*nsmpl )
	cadd.bias:copy( torch.eye( dim*nsmpl ) ) 

	local inode = nn.Identity()()
	local output = cadd( cmul( inode ) )

	local network = nn.gModule({inode}, {output})

	return network 

end

function lowerTriangleBlockwiseSub( nsmpl, dim ) 

	local cmul = nn.CMul( 1, dim*nsmpl, dim*nsmpl ) 

	cmul.weight:zero()
	for i = 1, nsmpl do 

		local ind = 1 + (i-1)*dim
		local block = cmul.weight:narrow( 2, ind, dim ):narrow( 3, ind, dim )
		block:copy( torch.tril( torch.ones(dim,dim), -1 ) )

		if i < nsmpl then 

			local block = cmul.weight:narrow( 2, ind+dim, dim ):narrow( 3, ind, dim )
			block:fill(1)

		end

	end

	local cadd = nn.CAdd( 1, dim*nsmpl, dim*nsmpl )
	cadd.bias:copy( torch.eye( dim*nsmpl ) ) 

	local inode = nn.Identity()()
	local output = cadd( cmul( inode ) )

	local network = nn.gModule({inode}, {output})

	return network 

end

function initializeAccRotationDim( polarInit ) 

	local a = polarInit[3] 
	local nNodes = a:size( 2 )
	local dim    = a:size( 3 ) 

	local unmaskedRotations = torch.Tensor( nNodes, dim, dim ):zero()

	table.insert( polarInit, unmaskedRotations )

	return polarInit 

end

function initializeAccRotationAll( polarInit ) 

	local a = polarInit[3] 
	local nNodes = a:size( 2 )
	local dim    = a:size( 3 ) 

	local unmaskedRotation = torch.Tensor( nNodes*dim, nNodes*dim ):zero()

	table.insert( polarInit, unmaskedRotation )

	return polarInit 

end

function initializeRotation( polarInit, ind ) 

	local z  = polarInit[ind] 
	local nZ = z:size( 2 )

	local unmaskedRotation = torch.Tensor( nZ, nZ ):zero()

	table.insert( polarInit, unmaskedRotation )

	return polarInit 

end

--- tests ---------

function testLowerTriangleBlockwiseSub()

	local nsmpl = 9 
	local dim = 10 

	local network = lowerTriangleBlockwiseSub( nsmpl, dim ) 

	local r = torch.rand( nsmpl*dim, nsmpl*dim )
	local l = network:updateOutput( r )

	print( l ) 

end 
]]