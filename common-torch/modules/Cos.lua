require 'nn'

local cos, parent = torch.class('nn.Cos', 'nn.Module')

function cos:updateOutput( input ) 

	self.output:resizeAs(input):copy(input):cos()

	return self.output 

end 

function cos:updateGradInput( input, gradOutput )

	self.gradInput:resizeAs( input ):copy( input ):sin():mul(-1):cmul( gradOutput ) 

	return self.gradInput 

end 

function testCos()

	local net = nn.Cos()
	local x = torch.linspace( -2*math.pi, 2*math.pi, 100 )
	local y = net:updateOutput( x ) 

	local dy = y:clone():fill(1)
	local dx = net:updateGradInput( x, dy ) 

	gnuplot.figure(); gnuplot.savePlot( 'cos.png', {{'fx',x,y,'-'}, {'df/dx',x,dx,'-'}} )

	local err = nn.Jacobian.testJacobian( net, x, -2*math.pi, 2*math.pi )
	print( 'jacobian error', err ) 	

end
-- testCos()

local sin, parent = torch.class('nn.Sin', 'nn.Module')

function sin:updateOutput( input ) 

	self.output:resizeAs(input):copy(input):sin()

	return self.output 

end 

function sin:updateGradInput( input, gradOutput )

	self.gradInput:resizeAs( input ):copy( input ):cos():cmul( gradOutput ) 

	return self.gradInput 

end 

function testSin()

	local net = nn.Sin()
	local x = torch.linspace( -2*math.pi, 2*math.pi, 100 )
	local y = net:updateOutput( x ) 

	local dy = y:clone():fill(1)
	local dx = net:updateGradInput( x, dy ) 

	gnuplot.figure(); gnuplot.savePlot( 'sin.png', {{'fx',x,y,'-'}, {'df/dx',x,dx,'-'}} )

	local err = nn.Jacobian.testJacobian( net, x, -2*math.pi, 2*math.pi )
	print( 'jacobian error', err ) 	

end
-- testSin()
