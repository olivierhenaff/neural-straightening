require 'nngraph'
require 'cephes'
require 'randomkit'

require 'common-torch/utilities/gnuplot'




local poisson, parent = torch.class('nn.PoissonNLL_normalized_masked', 'nn.Module')

function poisson:__init( mb, data, dataMask ) -- data is spike counts, with trials along first dimension

	parent.__init(self)

	self.poissonNLL = nn.PoissonNLL_normalized( nn.Unsqueeze(1):updateOutput(data) )
	self.mask       = nn.CMul( dataMask:size() ) 

	local rate = nn.Identity()()

	local nll  = self.poissonNLL( rate )
	nll = self.mask( nll ) 
	nll = nn.View( mb, -1 )( nll ) 
	nll = nn.Sum(2)( nll )

	self.network = nn.gModule({ rate }, { nll })

	self:loadData( data, dataMask )

end

function poisson:updateOutput( input ) 

	self.output = self.network:updateOutput( input ) 

	return self.output 

end 

function poisson:updateGradInput( input, gradOutput )

	self.gradInput = self.network:updateGradInput( input, gradOutput ) 

	return self.gradInput

end 

function poisson:loadData( data, dataMask )

	self.poissonNLL:loadData( nn.Unsqueeze(1):updateOutput(data) ) 
	self.mask.weight:copy( dataMask ) 

end




local poisson, parent = torch.class('nn.PoissonNLL_unnormalized', 'nn.Module')

function poisson:__init( data ) -- data is SUM of spike counts, and number of trials 

	parent.__init(self)

	self.sum     = nn.CMul( data.sum:size() )
	self.nTrials = nn.MulConstant( 1 )

	local rates = nn.Identity()() 
	local nll   = nn.CSubTable()({ self.nTrials(rates), self.sum( nn.Log()(rates) ) })

	self.network = nn.gModule({ rates }, { nll })

	self:loadData( data )

end

function poisson:updateOutput( input ) 

	self.output = self.network:updateOutput( input ) 

	return self.output 

end 

function poisson:updateGradInput( input, gradOutput )

	self.gradInput = self.network:updateGradInput( input, gradOutput ) 

	return self.gradInput

end 

function poisson:loadData( data )

	self.sum.weight:copy( data.sum ) 
	self.nTrials.constant_scalar = data.nTrials

end





local poisson, parent = torch.class('nn.PoissonNLL_normalized', 'nn.Module')

function poisson:__init( data ) -- data is spike counts, with trials along first dimension

	parent.__init(self)

	-- local data = data:transpose( 1, 5 )

	-- self.sum     = nn.CMul( data[1]:size() )
	-- self.nTrials = nn.MulConstant( 1 )
	-- self.const   = nn.Add( data[1]:size() )

	self.sum     = nn.CMul( data:select( 5, 1 ):size() )
	self.nTrials = nn.MulConstant( 1 )
	self.const   = nn.Add( data:select( 5, 1 ):size() )

	local rates = nn.Identity()() 
	local nll   = nn.CSubTable()({ self.nTrials(rates), self.sum( nn.Log()(rates) ) })
	nll = self.const( nll ) 

	self.network = nn.gModule({ rates }, { nll })

	self:loadData( data )

end

function poisson:updateOutput( input ) 

	self.output = self.network:updateOutput( input ) 

	return self.output 

end 

function poisson:updateGradInput( input, gradOutput )

	self.gradInput = self.network:updateGradInput( input, gradOutput ) 

	return self.gradInput

end 

function poisson:loadData( data )

	-- print( 'load data poisson' )

	-- local sum     = data:sum(1)
	-- local nTrials = data:size(1)

	local sum     = data:sum(5)
	local nTrials = data:size(5)

	-- print( data:size() ) 
	-- print( sum:size())
	-- print( self.sum.weight:size() ) 

	self.sum.weight:copy(         sum ) 

	-- print( data:size() ) 
	-- print( self.sum.weight:size() ) 
	-- error('self.sum.weight')

	self.nTrials.constant_scalar = nTrials 

	-- self.const.bias:copy( cephes.lgam(data:clone():add(1)):viewAs(data):sum(1) )
	self.const.bias:copy( cephes.lgam(data:clone():add(1)):viewAs(data):sum(5) )

	collectgarbage()

end

function testPoissonAgainstHist()

	local rate    = torch.Tensor{ 6 }
	local nTrials = 1
	local data    = torch.rand( nTrials, 1 ) 
	local nll = nn.PoissonNLL_normalized( data )

	local poisson_emp = randomkit.poisson( torch.Tensor(10000), rate[1] )

	local plot = {}
	table.insert( plot, gnuplot.plotHist( poisson_emp, 'counts' ) )

	local x = torch.range( 0, poisson_emp:max(), 1 ) 
	local y = x:clone()

	for i = 1, x:size(1) do 

		data:fill( x[i] ) 
		nll:loadData( data )
		y[i] = nll:updateOutput( rate ):squeeze() 

	end

	y:mul( -1 ):exp()
	y:mul( poisson_emp:size(1) )

	table.insert( plot, { x, y, '+' } )

	gnuplot.savePlot( 'poisson.png', plot ) 

end	
-- testPoissonAgainstHist() 

function testPoissonAgainstUnnormalized() 

	local nTrials = 100
	local nSample = 10 

	local rates  = torch.rand( nSample ) 
	local spikes = torch.Tensor( nTrials, nSample ) 
	for i = 1, nTrials do randomkit.poisson( spikes[i], rates ) end 

	local poisson1 = nn.PoissonNLL_normalized( spikes ) 
	local poisson2 = nn.PoissonNLL_unnormalized{ sum = spikes:sum(1), nTrials = nTrials } 	

	local rates  = torch.rand( nSample ) 
	local p1 = poisson1:updateOutput( rates ):clone()
	local p2 = poisson2:updateOutput( rates ):clone() 

	local gradOutput = p1:clone():fill( 1 ) 

	local g1 = poisson1:updateGradInput( rates, gradOutput ) 
	local g2 = poisson2:updateGradInput( rates, gradOutput ) 

	print( 'difference between gradients of normalized and unnormalized Poisson likelihoods: ', g1:dist( g2 ) ) 

end
-- testPoissonAgainstUnnormalized()

function testPoissonLikelihood()

	local rate_true = 4 
	local nTrials   = 100
	local nPoints   = 100000

	local sum   = torch.Tensor( nPoints-1 ):fill( randomkit.poisson( rate_true*nTrials ) ) 
	local nll   = nn.PoissonNLL_unnormalized{ sum = sum, nTrials = nTrials }
	local rates = torch.linspace( 0, 10, nPoints )
	rates = rates:narrow( 1, 2, rates:size(1)-1 )

	local l     = nll:updateOutput( rates ) 

	local plot = {} 
	table.insert( plot, { rates, l, '-'} )
	table.insert( plot, {torch.Tensor(2):fill( sum[1]/nTrials ), torch.Tensor{l:min(),l:max()}, '-'} )

	s, i = torch.sort( l ) 

	print( sum[1]/nTrials, rates[i[1]] ) 

	gnuplot.savePlot( 'poisson-likelihood.png', plot ) 


end
-- testPoissonLikelihood()

function testPoissonConjugate()

	local rate_true = 4 
	local nTrials   = 10
	local nPoints   = 10000

	local sum   = torch.Tensor( nPoints-1 ):fill( randomkit.poisson( rate_true*nTrials ) ) 
	local nll   = nn.PoissonNLL_unnormalized{ sum = sum, nTrials = nTrials }
	local rates = torch.linspace( 0, 10, nPoints )
	rates = rates:narrow( 1, 2, rates:size(1)-1 )

	local l     = nll:updateOutput( rates ) 
	l:mul( -1 ):exp() 
	l:div( (rates[2] - rates[1]) * l:sum() ) 
	-- l:mul( 10000 / l:sum() ) 

	local plot = {} 

	local gamma_samples = randomkit.gamma( torch.Tensor(10000), sum[1]+1, 1/nTrials )

	print( 'gamma mean', gamma_samples:mean() ) 

	table.insert( plot, gnuplot.plotHist( gamma_samples, 'density' ) ) 

	table.insert( plot, { rates, l, '-'} )
	gnuplot.savePlot( 'poisson-conjugate.png', plot ) 


end
-- testPoissonConjugate()

