require 'nngraph'

require 'common-torch/modules/constTensor'
require 'common-torch/modules/Cos'

function grahamSchmidtMatrix( mb, ncells, dim )

	local inode = nn.Identity()() 

	local e1 = nn.Narrow( 3, 1, 1 )( inode )

	local output = { grahamSchmidt( mb, ncells )({ e1, nn.MulConstant(0)( e1 ) }) }

	for i = 2, dim do 

		local new = nn.Narrow( 3, i, 1 )( inode )

		for j = 1, i-1 do 

			new = grahamSchmidt( mb, ncells )({ new, output[j] })

		end

		table.insert( output, new ) 

	end

	output = nn.JoinTable( 3 )( output ) 

	local network = nn.gModule( {inode}, {output} )

	return network 

end

function grahamSchmidt( mb, dim )

	local inode = nn.Identity()()

	local a = nn.SelectTable(1)( inode ) -- mb, dim, 1 
	local v = nn.SelectTable(2)( inode ) -- mb, dim, 1  

	local dot = nn.Sum(2)( nn.CMulTable()({ a, v }) )
	dot = nn.View( mb, 1, 1 )( dot )
	dot = nn.Replicate( dim, 2 )( dot )

	local aHat = nn.CSubTable()({ a, nn.CMulTable()({dot, v}) })

	local norm = nn.Sqrt()( nn.Sum(2)( nn.Square()( aHat ) ) ) 
	norm = nn.View( mb, 1, 1 )( norm )
	norm = nn.Replicate( dim, 2 )( norm ) 

	aHat = nn.CDivTable()({ aHat, norm })

	local network = nn.gModule( {inode}, {aHat} )

	return network 

end

function accCurvatureToVhat( mb, dim, nsmpl, v0 )

	local inode = nn.Identity()()

	local t = nn.SelectTable(1)( inode ):annotate{ name = 'theta' } -- mb,   1, nsmpl-2
	local a = nn.SelectTable(2)( inode ):annotate{ name = 'accel' } -- mb, dim, nsmpl-2

	local e1 = torch.Tensor( mb, dim, 1 ):zero(); e1:select(2,1):fill(1)

	local vHat = {} 

	if v0 then 
		vHat[1] = grahamSchmidt( mb, dim )({ nn.SelectTable(3)( inode ), nn.ConstTensor(e1:zero())(t) })
	else
		vHat[1] = nn.ConstTensor(e1)( t ):annotate{ name = 'vHat'..1 }
	end

	local cosT = nn.Replicate( dim, 2 )( nn.Cos()( t ) ):annotate{ name = 'cosTheta' }
	local sinT = nn.Replicate( dim, 2 )( nn.Sin()( t ) ):annotate{ name = 'sinTheta' }

	for i = 1, nsmpl-2 do

		local aHat = grahamSchmidt( mb, dim )({ nn.Narrow( 3, i, 1 )( a ), vHat[i] })

		local cI = nn.Narrow( 3, i, 1 )( cosT ):annotate{ name = 'cosTheta'..i }
		local sI = nn.Narrow( 3, i, 1 )( sinT ):annotate{ name = 'sinTheta'..i } 

		local cosTvHat = nn.CMulTable()({ cI, vHat[i] })
		local sinTaHat = nn.CMulTable()({ sI, aHat    })

		vHat[i+1] = nn.CAddTable()({ cosTvHat, sinTaHat }):annotate{ name = 'vHat'..i+1 }

	end

	vHat = nn.JoinTable( 3 )( vHat )

	local network = nn.gModule( {inode}, {vHat} )

	return network 

end 



function distCurvAccToTraj( mb, dim, nsmpl, v0, z0 )

	local inode = nn.Identity()()

	local d = nn.SelectTable(1)( inode ) -- mb,   1, nsmpl-1
	local t = nn.SelectTable(2)( inode ) 
	local a = nn.SelectTable(3)( inode ) 

	local vHat = accCurvatureToVhat( mb, dim, nsmpl, v0 )({ t, a, v0 and nn.SelectTable(4)(inode) or nil })

	local d = nn.Replicate( dim, 2 )( d ) 
	local v = nn.CMulTable()({ d, vHat } ) -- mb, dim, nsmpl-1

	local z 
	if z0 then

		z = { nn.SelectTable(5)( inode ) }

	else

		local z0 = torch.Tensor( mb, dim, 1 ):zero()
		z = { nn.ConstTensor(z0)( v ) } 

	end

	for i = 2, nsmpl do 

		z[i] = nn.CAddTable()({ z[i-1], nn.Narrow( 3, i-1, 1)( v ) })

	end 

	z = nn.JoinTable( 3 )( z ) 

	local network = nn.gModule( {inode}, {z} )

	return network

end 



function testGrahamSchmidt() 

	local mb = 10
	local dim   = 5 

	local a = torch.randn( mb, dim, 1 )
	local v = torch.randn( mb, dim, 1 )

	for i = 1, mb do v[i]:div( v[i]:norm() ) end 

	local gs   = grahamSchmidt( mb, dim ) 
	local aHat = gs:updateOutput({ a, v })

	for i = 1, mb do 

		print( aHat[i]:norm(), v:dot( aHat ) ) 

	end	

end
-- testGrahamSchmidt() 

function testGrahamSchmidtMatrix()

	local mb = 3
	local dim = 5 
	local ncells = 36 

	local gs = grahamSchmidtMatrix( mb, ncells, dim )

	local x = torch.rand( mb, ncells, dim ) 
	local y = gs:updateOutput( x ) 

	g1 = torch.bmm( x:transpose(2,3), x )
	g2 = torch.bmm( y:transpose(2,3), y )

	-- print( g:size() )
	print( g1 ) 
	print( g2 ) 

	error('stop her')

	-- for i = 1, mb do print( g[i] ) end  

end
-- testGrahamSchmidtMatrix()



