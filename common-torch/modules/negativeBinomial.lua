require 'nngraph'
require 'cephes'
require 'randomkit'

require 'common-torch/utilities/gnuplot'

require 'common-torch/modules/constTensor'
require 'common-torch/modules/logGamma'

local negativeBinomial, parent = torch.class('nn.NegativeBinomialNLL', 'nn.Module')

-- function negativeBinomial:__init( data ) -- data is SUM of spike counts, and number of trials 
function negativeBinomial:__init( data ) -- data is all spike counts, with trials along first dimension

	parent.__init(self)

	local data = nn.Unsqueeze(1):updateOutput( data ) 

	self.numTrials    = nn.ConstTensor( data[1]:clone() )
	self.mulSumTrials = nn.CMul( data[1]:size() ) 
	self.addSumLogGam = nn.Add(  data[1]:size() ) 
	self.addData      = nn.Add(  data:size() ) 

	local inode = nn.Identity()()
	local r     = nn.SelectTable( 1 )( inode ) 
	local p     = nn.SelectTable( 2 )( inode ) 

	local numTrials = self.numTrials( r ) 
	numTrials = nn.Replicate( mb, 1 )( numTrials )


	local term1 = nn.CMulTable()({ numTrials, nn.LogGamma()( r ) })
	-- local term1 = self.mulNumTrials( nn.LogGamma()( r ) )

	term1 = self.addSumLogGam( term1 ) 

	local logP         = nn.Log()( p ) 
	local logOneMinusP = nn.Log()( nn.AddConstant( 1 )( nn.MulConstant( -1 )( p ) ) )

	local term2 = nn.CMulTable()({ numTrials, r, logOneMinusP })
	term2 = nn.MulConstant( -1 )( term2 ) 

	local term3 = self.mulSumTrials( logP ) 
	term3 = nn.MulConstant( -1 )( term3 ) 

	-- local dataDim = 1
	local dataDim = 2

	local term4 = nn.Replicate( data:size(1), dataDim )( r ) -- WHAT ABOUT MB??? 
	term4 = nn.LogGamma()( self.addData( term4 ) )
	term4 = nn.Sum( dataDim )( term4 ) 
	term4 = nn.MulConstant( -1 )( term4 ) 

	local nll = nn.CAddTable()({ term1, term2, term3, term4 })

	self.network = nn.gModule({ inode }, { nll })

	self:loadData( data )

end

function negativeBinomial:updateOutput( input ) 

	self.output = self.network:updateOutput( input ) 

	return self.output 

end 

function negativeBinomial:updateGradInput( input, gradOutput )

	self.gradInput = self.network:updateGradInput( input, gradOutput ) 

	return self.gradInput

end 

function negativeBinomial:loadData( data )

	local data = nn.Unsqueeze(1):updateOutput( data ) 

	self.numTrials.output:fill( data:size(1) )

	self.mulSumTrials.weight:copy( data:sum(1) )

	self.addSumLogGam.bias:copy( cephes.lgam(data:clone():add(1)):viewAs(data):sum(1) )

	self.addData.bias:copy( data )

	collectgarbage()

end

function testNegativeBinomial() 

	-- local r = torch.randn( 1 ):exp() 
	local r = torch.Tensor{ 4 } --randn( 1 ):exp() 
	local p = torch.rand(  1 ) 

	local nTrials = 10000

	local samples = randomkit.negative_binomial( torch.Tensor(nTrials), r:squeeze(), 1-p:squeeze() )

	local data = torch.Tensor( 1, 1 ) 
	local nll = nn.NegativeBinomialNLL( data )

	local x = torch.range( 0, samples:max(), 1 ) 
	local y = x:clone()

	for i = 1, x:size(1) do 

		data:fill( x[i] ) 
		nll:loadData( data )
		y[i] = nll:updateOutput( {r,p} ):squeeze() 

	end

	y:mul( -1 ):exp()
	y:mul( nTrials )

	local plot = {} 
	table.insert( plot, gnuplot.plotHist( samples, 'counts' ) )
	table.insert( plot, { x, y, '+' } )

	gnuplot.savePlot( 'negativeBinomial.png', plot ) 

end
-- testNegativeBinomial() 
