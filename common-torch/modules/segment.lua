function Segment( tensors, dim ) 

	local dim = dim or 1 
	local inode = nn.Identity()() 
	local output = {} 
	local ind = 1 

	for i, tensor in pairs( tensors ) do 

		local len = tensor:nElement()/tensor:size(1)

		output[i] = nn.Narrow( dim, ind, len )( inode )
		output[i] = nn.View( tensor:size() )( output[i] ) 

		ind = ind + len 

	end 

	local network = nn.gModule( {inode}, output )

	return network

end 

function testSegment() 

	local t1 = torch.Tensor( 2, 3 )
	local t2 = torch.Tensor( 3, 4, 5 )
	local t3 = torch.Tensor( 4, 1 )

	local tensors = {t1, t2, t3}
	local ind = 1 

	for _, tensor in pairs( tensors ) do 

		local s = tensor:storage() 

		for i = 1, tensor:nElement() do 

			s[i] = ind 
			ind = ind + 1 

		end 

	end 

	local input = torch.rand( t1:nElement() + t2:nElement() + t3:nElement() ) 

	local segment = Segment( tensors ) 
	segmented = segment:updateOutput( input ) 

	local fused = segment:updateGradInput( input, tensors )

	print( fused:dist( torch.range( 1, fused:size(1) ) ) )

end 
-- testSegment() 
