require 'nn'

local constTensor, parent = torch.class('nn.ConstTensor', 'nn.Module')

function constTensor:__init( tensor )

	parent.__init(self)

	self.output = tensor

end

function constTensor:updateOutput( input ) 

	return self.output 

end

function constTensor:updateGradInput( input, gradOutput )

	self.gradInput:resizeAs( input ):zero()

	return self.gradInput 

end

function testConstTensor()

	local ncond = 2 
	local nsmpl = 11 

	local mask = torch.ByteTensor(ncond, nsmpl, nsmpl):zero()
	for i = 1, nsmpl-1 do
		mask:select(2,i):select(2,i+1):fill(1)
	end

	local x = torch.rand( ncond, nsmpl, nsmpl )
	local n = nn.ConstTensor( mask )
	local y = n:updateOutput( x ) 
	local z = n:updateGradInput( x, y ) 

	print( x, y, z )

end
--testConstTensor()