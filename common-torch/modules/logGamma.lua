require 'cephes'
require 'nn'

local logGamma, parent = torch.class('nn.LogGamma', 'nn.Module')

function logGamma:updateOutput( input ) 

	self.output:resizeAs( input ) 

	cephes.lgam( self.output, input )

	return self.output 

end 

function logGamma:updateGradInput( input, gradOutput )

	self.gradInput:resizeAs( input )

	cephes.psi( self.gradInput, input ) 

	self.gradInput:cmul( gradOutput ) 

	return self.gradInput

end 

function testLogGamma()

	local net = nn.LogGamma() 
	local  x  = torch.linspace( -2*math.pi, 2*math.pi, 100 )

	local err = nn.Jacobian.testJacobian( net, x, 1, 2 )
	print( 'jacobian error', err ) 	

end
-- testLogGamma() 

function logMultivariateGamma( p, dim ) 

	local dim = dim or 2 

	local inode nn.Identity()()

	local offset = nn.CAdd( p )
	offset.bias:copy( torch.range(1,p) ):mul(-1):add(1):div(2)

	local output = nn.Replicate( p, dim )( inode ) 
	output = offset( output ) 
	output = nn.LogGamma()( output )
	output = nn.Sum()( output ) 
	output = nn.AddConstant( p*(p-1)*math.log(math.pi)/2 )( output ) 

	local network = nn.gModule({ inode }, { output })

	return network 

end

-- logMultivariateGamma( 33 ) 