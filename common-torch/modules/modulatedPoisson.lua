require 'common-torch/modules/negativeBinomial'

function neuralToNegativeBinomialParams()

	local inode = nn.Identity()()

	local rates   = nn.SelectTable( 1 )( inode ) 
	local logSigG = nn.SelectTable( 2 )( inode ) 

	local sigG2    = nn.Exp()( nn.MulConstant(  2 )( logSigG ) ) 
	local r        = nn.Exp()( nn.MulConstant( -2 )( logSigG ) ) 

	local s = nn.CMulTable()({ sigG2, rates })
	local p = nn.CDivTable()({ s, nn.AddConstant(1)(s) })

	local network = nn.gModule({ inode }, { r, p } )

	return network 

end












local modulatedPoisson, parent = torch.class('nn.ModulatedPoissonNLL_masked', 'nn.Module')

function modulatedPoisson:__init( mb, data, dataMask ) -- data is spike counts, with trials along first dimension

	parent.__init(self)

	self.modulatedPoissonNLL = nn.ModulatedPoissonNLL( nn.Unsqueeze(1):updateOutput(data) )
	self.mask       = nn.CMul( dataMask:size() ) 

	local rate = nn.Identity()()

	local nll  = self.modulatedPoissonNLL( rate )
	nll = self.mask( nll ) 
	nll = nn.View( mb, -1 )( nll ) 
	nll = nn.Sum(2)( nll )

	self.network = nn.gModule({ rate }, { nll })

	self:loadData( data, dataMask )

end

function modulatedPoisson:updateOutput( input ) 

	self.output = self.network:updateOutput( input ) 

	return self.output 

end 

function modulatedPoisson:updateGradInput( input, gradOutput )

	self.gradInput = self.network:updateGradInput( input, gradOutput ) 

	return self.gradInput

end 

function modulatedPoisson:loadData( data, dataMask )

	self.modulatedPoissonNLL:loadData( nn.Unsqueeze(1):updateOutput(data) ) 
	self.mask.weight:copy( dataMask ) 

end



















local modulatedPoisson, parent = torch.class('nn.ModulatedPoissonNLL', 'nn.Module')

-- function modulatedPoisson:__init( data ) -- data is SUM of spike counts, and number of trials 
function modulatedPoisson:__init( data ) -- data is all spike counts, with trials along the first dimension  

	parent.__init(self)

	self.negativeBinomialNLL = nn.NegativeBinomialNLL( data )

	local ratesLogSigG = nn.Identity()()

	local rp  = neuralToNegativeBinomialParams()( ratesLogSigG )

	local nll = self.negativeBinomialNLL( rp ) 

	self.network = nn.gModule({ ratesLogSigG }, { nll })

end

function modulatedPoisson:updateOutput( input ) 

	self.output = self.network:updateOutput( input ) 

	return self.output 

end 

function modulatedPoisson:updateGradInput( input, gradOutput )

	self.gradInput = self.network:updateGradInput( input, gradOutput ) 

	return self.gradInput

end 

function modulatedPoisson:loadData( data )

	self.negativeBinomialNLL:loadData( data ) 

end

function testConvergenceModulatedPoisson()

	require 'common-torch/modules/poisson'

	local nTrials  = 100 
	local nSamples =  10 

	local rates = torch.rand( 10 )
	local data = torch.Tensor( nTrials, nSamples )
	for i = 1, nTrials do randomkit.poisson( data[i], rates ) end 

	local poisson = nn.PoissonNLL_normalized( data ) 
	local modPois = nn.ModulatedPoissonNLL( data ) 

	local logSigG = rates:clone():fill( -10 ) 

	local o1 = poisson:updateOutput( rates ) 
	local o2 = modPois:updateOutput({rates, logSigG})

	print( 'maximum difference between Poisson NLL and Modulated Poisson with low sigG2', o1:clone():add( -1, o2 ):abs():max() )

end
-- testConvergenceModulatedPoisson()













local sampler, parent = torch.class('nn.ModulatedPoissonSampler', 'nn.Module')

function sampler:__init( params )

	parent.__init(self)

	self.converter = neuralToNegativeBinomialParams()

	self:loadParams( params ) 

end

function sampler:loadParams( params ) 

	self.params = self.converter:updateOutput( params ) 

	self.params[2]:mul( -1 ):add( 1 ) 

end

function sampler:sample()

	self.output:resizeAs( self.params[1] )

	randomkit.negative_binomial( self.output, self.params[1], self.params[2] )

	return self.output 

end 


function testModulatedPoissonSampler( nTrials ) 

	-- local nTrials = 10000

	local rate    = torch.Tensor( 1 ):fill( 4.0 )
	local logSigG = torch.Tensor( 1 ):fill( 0.5 ):log()

	local modulatedPoissonSampler = nn.ModulatedPoissonSampler{ rate, logSigG }
	local samples = torch.Tensor( nTrials, 1 ) 
	for n = 1, nTrials do samples[n]:copy( modulatedPoissonSampler:sample() ) end 

	local plot = {} 
	local plotHist = gnuplot.plotHist( samples, 'counts', samples:max()*10+1, 0, samples:max() ); table.insert( plotHist, 1, 'modulated' )
	table.insert( plot, plotHist )

	local x = torch.range( 0, samples:max(), 1 ) 
	local y = x:clone()
	local data = torch.Tensor( 1, 1, 1 ) 
	mb = 1 
	local nll = nn.ModulatedPoissonNLL( data ) 

	for i = 1, x:size(1) do 

		nll:loadData( data:fill( x[i] ) ) 
		y[i] = nll:updateOutput{ rate, logSigG }:squeeze()

	end

	y:mul( -1 ):exp():mul( nTrials ) 

	table.insert( plot, { x, y, '+' } )

	local samplesPoisson = randomkit.poisson( torch.Tensor( nTrials, 1 ), rate:squeeze() ) 
	local plotHist = gnuplot.plotHist( samplesPoisson, 'counts', samples:max()*10+1, 0, samples:max() ); table.insert( plotHist, 1, 'poisson' )
	table.insert( plot, plotHist )

	gnuplot.savePlot( 'modulatedPoissonHist.png', plot ) 

--[[

	local nll = nn.ModulatedPoissonNLL( samples )

	local rates    = torch.linspace(  1, 10, 100 ) 
	local logSigGs = torch.logspace( -2,  0, 100 ):log()
	local loss     = rates:clone()

	local params   = { torch.Tensor(1), torch.Tensor(1) }

	for i = 1, rates:size(1) do 

		params[1]:fill( rates[i] )
		params[2]:copy( logSigG )
		loss[i] = nll:updateOutput( params ):squeeze()

	end
	loss:add( - loss:min() ):mul( -1 ):exp() 

	local plot = {} 
	table.insert( plot, { rates, loss, '-' } )
	table.insert( plot, { torch.Tensor( 2 ):fill( rate:squeeze() ), torch.Tensor{ loss:min(), loss:max() } } )

	gnuplot.savePlot( 'modulatedPoissonNLL-rate-' .. nTrials .. 'trials.png', plot ) 



	for i = 1, rates:size(1) do 

		params[1]:copy( rate ) 
		params[2]:fill( logSigGs[i] ) 
		loss[i] = nll:updateOutput( params ):squeeze()

	end
	loss:add( - loss:min() ):mul( -1 ):exp() 

	local plot = {} 
	table.insert( plot, { logSigGs, loss, '-' } )
	table.insert( plot, { torch.Tensor( 2 ):fill( logSigG:squeeze() ), torch.Tensor{ loss:min(), loss:max() } } )

	gnuplot.savePlot( 'modulatedPoissonNLL-logSigG-' .. nTrials .. 'trials.png', plot ) 
]]
end
-- testModulatedPoissonSampler( 10000 )


-- for _, nTrials in pairs{ 2, 10, 100, 1000, 10000, 100000, 1000000 } do testModulatedPoissonSampler( nTrials ) end


