require 'common-torch/modules/SampleGaussian'

require 'randomkit'

require 'common-torch/utilities/gnuplot'
require 'common-torch/modules/modulatedPoisson'
require 'common-torch/modules/poisson'
require 'common-torch/modules/GaussianKL'


local likelihood, parent = torch.class( 'nn.OverdisperedPoissonNLL_VB_singleSample_shared', 'nn.Module' )

function likelihood:__init( mb, muSigDim, data, dataMask ) 

	self.poissonNLL  = nn.PoissonNLL_normalized( nn.Unsqueeze(1):updateOutput(data):transpose(1,5) )
	-- self.poissonNLL  = nn.PoissonNLL_normalized( nn.Unsqueeze(5):updateOutput(data) )

	local inode = nn.Identity()()

	local membRate = nn.SelectTable( 1 )( inode )
	local qParams  = nn.SelectTable( 2 )( inode )
	local tParams  = nn.SelectTable( 3 )( inode )
	local coupling = nn.SelectTable( 4 )( inode ) 
	local sParams  = nn.SelectTable( 5 )( inode ) 

	-- membRate = nn.Replicate( data:size(1), 2 )( membRate ) 
	tParams  = nn.Replicate( data:size(1), 3 )( tParams  ) 

	coupling = nn.Replicate( data:size(1), 2 )( coupling ) 
	coupling = nn.Replicate( data:size(2), 3 )( coupling ) 
	coupling = nn.Copy( nil, nil, true )( coupling ) 
	coupling = nn.View( mb*data:size(1)*data:size(2), data:size(3), nModulatorsModel )( coupling ) 

	local pSample = SampleGaussianMuLogStd( muSigDim )( qParams ):annotate{ name = 'pSample' }
	local sSample = SampleGaussianMuLogStd( muSigDim )( sParams ):annotate{ name = 'sSample' }
	sSample = nn.View( mb*data:size(1)*data:size(2), nModulatorsModel, 1 )( sSample ) 
	sSample = nn.MM()({ coupling, sSample })
	sSample = nn.View( mb, data:size(1), data:size(2), data:size(3) )( sSample ) 

	local poissonRate = nn.CAddTable()({ membRate, pSample, sSample }):annotate{ name = 'prePoissonRate' } 
	poissonRate = nn.Exp()( poissonRate ):annotate{ name = 'poissonRate' }

	local dataMask = nn.Unsqueeze(1):updateOutput(dataMask):transpose(1,5):squeeze()
	local mask = nn.CMul( dataMask:size() )
	mask.weight:copy( dataMask )

	local pKL = GaussianKLdiagonalLearnedMuStd( mb, muSigDim )({ qParams, tParams })
	local sKL = GaussianKLdiagonalLearnedMuStd( mb, muSigDim )({ sParams, nn.MulConstant(0)( sParams ) })

	local nll = self.poissonNLL( poissonRate ):annotate{ name = 'perSample' }
	nll = mask( nll )
	nll = nn.View( mb, -1 )( nll )
	nll = nn.Sum(2)( nll )

	local loss = nn.CAddTable()({ nll, pKL, sKL })

	self.network = nn.gModule({ inode }, { loss })

end

function likelihood:updateOutput( input )  

	self.output    = self.network:updateOutput( input ) 

	return self.output 

end

function likelihood:updateGradInput( input, gradOutput ) 

	self.gradInput = self.network:updateGradInput( input, gradOutput ) 

	return self.gradInput 

end 

function likelihood:loadData( data ) 

	-- print( 'load data overdispersed' )

	self.poissonNLL:loadData( nn.Unsqueeze(1):updateOutput(data):transpose(1,5) ) 
	-- self.poissonNLL:loadData( nn.Unsqueeze(5):updateOutput(data) ) 

end




--[[
local likelihood, parent = torch.class( 'nn.OverdisperedPoissonLikelihoodVB', 'nn.Module' )

function likelihood:__init( nsamplesNLL, mb, muSigDim, data ) 

	self.poissonNLL  = nn.PoissonNLL_normalized( data )
	self.poissonRate = torch.Tensor( nsamplesNLL ) 
	-- self.poissonRate = torch.Tensor( data:size() ) 
	self.kl 		 = GaussianKLdiagonalLearnedMuStd( mb, muSigDim )

end

function likelihood:updateOutput( rate, qParams, tParams ) 

	-- self.qMu  = self.qMu  or torch.Tensor()
	-- self.qSig = self.qSig or torch.Tensor()   

	-- self.qMu:resizeAs(  qParams[1][1] ):copy( qParams[1][1] )
	-- self.qSig:resizeAs( qParams[1][2] ):copy( qParams[1][2] ):exp()

	-- self.qMu:resize(  qParams:size(3), 1 ):copy( qParams[1][1] )
	-- self.qSig:resize( qParams:size(3), 1 ):copy( qParams[1][2] ):exp()

	local qMu  = 		   qParams[1][1]
	local qSig = math.exp( qParams[1][2] ) 

	self.poissonRate:randn( self.poissonRate:size() ):mul( qSig ):add( qMu + math.log( rate ) ):exp()
	-- self.poissonRate:randn( self.poissonRate:size() )
	-- 	:cmul( self.qSig:expandAs( self.poissonRate ) )
	-- 	:add(   self.qMu:expandAs( self.poissonRate ) )
	-- 	:add( math.log( rate ) )
	-- 	:exp()




	-- local nll  = self.poissonNLL:updateOutput( self.poissonRate ):sum(1):mean(2) 
	-- local kl   = self.kl:updateOutput{ qParams, tParams }
	-- self.output = - nll - kl 

	local nll  = self.poissonNLL:updateOutput( self.poissonRate )
	self.output = - nll:mean() - self.kl:updateOutput{ qParams, tParams }

	return self.output 

end

function likelihood:loadData( data ) 

	self.poissonNLL:loadData( data ) 

end
]]
--[[
local likelihood, parent = torch.class( 'nn.OverdisperedPoissonLikelihoodExact', 'nn.Module' )

function likelihood:__init( nsamplesNLL, data ) 

	self.poissonNLL  = nn.PoissonNLL_normalized( data )
	-- self.poissonRate = torch.Tensor( nsamplesNLL ) 
	self.poissonRate = torch.Tensor( data:size() ) 

end

function likelihood:updateOutput( rate, sigN ) 

	self.poissonRate:randn( self.poissonRate:size() ):mul( sigN ):add( math.log( rate ) ):exp()
	local nll = self.poissonNLL:updateOutput( self.poissonRate ) 

	self.output = nll:mul( - 1 ):exp():mean(2)

	return self.output 

end

function likelihood:loadData( data ) 

	self.poissonNLL:loadData( data ) 

end
]]





local sampler, parent = torch.class('nn.OverdispersedPoissonSamplerShared', 'nn.Module')

function sampler:__init( muSigDim, paramsPrivate, coupling ) -- params contain [ membMean, sigN ] 

	parent.__init(self)

	local nModulators = coupling:size(1)
	local nCells      = coupling:size(2)

	self.sampleGaussian       = nn.SampleGaussianMuStd( muSigDim )
	self.coupling             = nn.Linear( nModulators, nCells ) 

	self:loadParams( paramsPrivate, coupling ) 

end

function sampler:loadParams( paramsPrivate, coupling ) 

	self.paramsPrivate = self.paramsPrivate or torch.Tensor() 
	self.paramsPrivate:resizeAs( paramsPrivate ):copy( paramsPrivate ) 
	self.paramsPrivate[1]:log()

	self.coupling.bias:zero()
	self.coupling.weight:copy( coupling ) 

end

function sampler:sample()

	self.noiseWhite = self.noiseWhite or torch.Tensor() 
	self.noiseWhite:randn( self.paramsPrivate:size(2), self.coupling.weight:size(2) ) 

	local noiseShared = self.coupling:updateOutput( self.noiseWhite )

	local memb = self.sampleGaussian:updateOutput( self.paramsPrivate )
	memb:add( noiseShared )

	local rate = memb:exp() 

	randomkit.poisson( self.output:resizeAs( self.paramsPrivate[1] ), rate )

	return self.output 

end 

--[[
function testOverdispersedMeanVsVar() 

	local rate  = 4 
	local sigN  = 0.5
	local sigG2 = math.exp(sigN^2) - 1

	local muSigDim = 1 
	local nsamples = 10000

	local overdispParams   = torch.Tensor( 2, nsamples )
	local overdispSampler  = nn.OverdispersedPoissonSampler( muSigDim, overdispParams ) 

	local modulatedParams  = torch.Tensor( 2, nsamples ) 
	local modulatedSampler = nn.ModulatedPoissonSampler( modulatedParams )

	local npoints = 21 
	local rates = torch.logspace( -1, 1, npoints ) 

	local overdispMean  = torch.Tensor( npoints ) 
	local overdispVar   = torch.Tensor( npoints )
	local modulatedMean = torch.Tensor( npoints ) 
	local modulatedVar  = torch.Tensor( npoints )
	local poissonMean   = torch.Tensor( npoints ) 
	local poissonVar    = torch.Tensor( npoints )

	for i = 1, npoints do 

		local rate = rates[i] 
		overdispParams[1]:fill( rate ) 
		overdispParams[2]:fill( sigN )
		overdispSampler:loadParams( overdispParams ) 

		local sample = overdispSampler:sample()
		overdispMean[i] = sample:mean() 
		 overdispVar[i] = sample:var() 

		modulatedParams[1]:fill( rate ) 
		modulatedParams[2]:fill( sigG2 ):log():div(2)
		modulatedSampler:loadParams( modulatedParams ) 

		local sample = modulatedSampler:sample()
		modulatedMean[i] = sample:mean()
		modulatedVar[i]  = sample:var() 

		randomkit.poisson( sample, rate ) 
		poissonMean[i] = sample:mean()
		poissonVar[i]  = sample:var() 

	end

	local plot = {} 
	table.insert( plot, {'overdispered, empirical',  overdispMean,  overdispVar, '+' } )
	table.insert( plot, { 'modulated  , empirical', modulatedMean, modulatedVar, '+' } )
	table.insert( plot, { 'poisson    , empirical',   poissonMean,   poissonVar, '+' } )

	overdispVarTheory = rates:clone():pow(2):mul( sigG2 ):add( rates ) 
	table.insert( plot, { 'theory', rates, overdispVarTheory, '-' } )
	table.insert( plot, {rates, rates, 'with lines ls 1 lc rgb "grey"'} )

	gnuplot.figure(1)
	gnuplot.savePlot( 'mean-vs-var.eps', plot, 'mean', 'variance', 'equal', {'set logscale xy'} ) 

end 
-- testOverdispersedMeanVsVar()

function testOverdispersedProbability()

	local rate  = 4 
	-- local sigN  = 0.5
	local sigN  = 1

	local nsamples = 100000

	local overdispParams  = torch.Tensor( 2, nsamples )
	overdispParams[1]:fill( rate ) 
	overdispParams[2]:fill( sigN )
	local overdispSampler = nn.OverdispersedPoissonSampler( muSigDim, overdispParams ) 
	local overdispSample  = overdispSampler:sample()


	local poissonSample = torch.Tensor( nsamples )
	randomkit.poisson( poissonSample, rate ) 

	local M = 20 --math.max( sample1:max(), sample2:max() ) 
	-- local M = 50 --math.max( sample1:max(), sample2:max() ) 

	local plot1 = gnuplot.plotHist( overdispSample, 'counts', 100, 0, M )
	table.insert( plot1, 1, 'overdispersed' ) 

	local plot2 = gnuplot.plotHist( poissonSample , 'counts', 100, 0, M )
	table.insert( plot2, 1, 'pure poisson'  )

	local plot  = {} 
	table.insert( plot, plot1 )
	table.insert( plot, plot2 )

	local nsamplesNLL = 10000 
	local data        = torch.Tensor( 1, nsamplesNLL ) 
	local poissonNLL  = nn.PoissonNLL_normalized( data )
	local poissonRate = torch.Tensor( nsamplesNLL ) 
	local counts      = torch.range( 0, M, 1 )
	local likelihood = torch.Tensor( M+1 ) 

	local likelihoodModule = nn.OverdisperedPoissonLikelihoodExact( nsamplesNLL, data) 

	for i = 1, M+1 do 
		 
		likelihoodModule:loadData( data:fill( counts[i] ) )
		likelihood[i] = likelihoodModule:updateOutput( rate, sigN ) 
		 
	end
	likelihood:mul( nsamples ) 

	table.insert( plot, { 'overdispered likelihood, exact', counts, likelihood, '+' } )
	gnuplot.figure(2); gnuplot.savePlot( 'overdispersedProbability.png', plot )



	local count = 4 
	data:fill( count ) 
	likelihoodModule:loadData( data ) 

	local rates = torch.linspace( 0, 10, 101 )
	local likelihood = torch.Tensor( rates:size(1) )
	for i = 1, rates:size(1) do likelihood[i] = likelihoodModule:updateOutput( rates[i], sigN ) end

	local plot = {} 
	table.insert( plot, {'overdispersed LL, exact', rates, likelihood:clone():log(), '-' } )



	local qMu  = 0 
	local qSig = sigN 
	local mb   = 1  
	local muSigDim = 2 

	local tParams = torch.Tensor( mb, 2 ) 
	tParams:select( 2, 1 ):fill( 0 )
	tParams:select( 2, 2 ):fill( sigN ):log()

	local qParams = torch.Tensor( mb, 2 )
	qParams:select( 2, 1 ):fill( qMu  ) 
	qParams:select( 2, 2 ):fill( qSig ):log() 

	local likelihoodModule = nn.OverdisperedPoissonLikelihoodVB( nsamplesNLL, mb, muSigDim, data ) 
	for i = 1, rates:size(1) do likelihood[i] = likelihoodModule:updateOutput( rates[i], qParams, tParams ) end
	table.insert( plot, {'overdispered LL, variational', rates, likelihood, '-' } )

	gnuplot.figure(3)
	gnuplot.savePlot( 'overdispersedLogLikelihood.png', plot)

end
-- testOverdispersedProbability()

function testOverdispersedLikelihood()

	nngraph.setDebug(true)

	local sigN    = 1
 	local ncounts = 10
 	-- local ncounts = 1

	local overdispParams  = torch.Tensor( 2, ncounts )
	local overdispSampler = nn.OverdispersedPoissonSampler( muSigDim, overdispParams ) 

	local nsamplesNLL        = 10000
	local data               = torch.Tensor( ncounts, nsamplesNLL ) 
	local dataVB			 = torch.Tensor( 1, ncounts )
	-- local dataVB			 = torch.Tensor( 1, ncounts, 1 )
	local likelihoodModuleEx = nn.OverdisperedPoissonLikelihoodExact( nsamplesNLL, data) 

 	local qMu      = 0 
	local qSig     = sigN 
	local mb       = 100  
	local muSigDim = 2 

	-- local likelihoodModuleVB = nn.OverdisperedPoissonLikelihoodVB( nsamplesNLL, mb, muSigDim, data ) 
	local likelihoodModuleVB = nn.OverdisperedPoissonNLL_VB_singleSample( mb, muSigDim, dataVB ) 

	local ratesTrue = torch.linspace( 1, 5, 6 ) 
	-- local ratesTrue = torch.linspace( 1, 5, 21 ) 
	local ratesEstm = torch.Tensor( ratesTrue:size(1) )
	local ratesEsVB = torch.Tensor( ratesTrue:size(1) )
	local rates     = torch.linspace( 0, 10, mb+1 )
	rates = rates:narrow( 1, 2, rates:size(1)-1 ) 

	-- local logRate = torch.Tensor( mb, 1 )
	local logRate = rates:clone():log()

	local plot = {} 

	local m = - math.huge

	for n = 1, ratesTrue:size(1) do 

		overdispParams[1]:fill( ratesTrue[n] ) 
		overdispParams[2]:fill( sigN )
		overdispSampler:loadParams( overdispParams )

		local overdispSample  = overdispSampler:sample()
		for i = 1, ncounts do data[i]:fill( overdispSample[i] ) end 
		dataVB:copy( overdispSample )

		likelihoodModuleEx:loadData( data ) 
		local likelihood = torch.Tensor( rates:size(1) )
		for i = 1, rates:size(1) do likelihood[i] = likelihoodModuleEx:updateOutput( rates[i], sigN ):log():sum() end
		likelihood:add( - likelihood:max() ) 
		m = math.max( m, likelihood:min() ) 
		table.insert( plot, { n == 1 and 'overdispersed LL, exact' or '', rates, likelihood, 'with lines ls 1' } )
		local _, ind = torch.sort( likelihood, true )
		ratesEstm[n] = rates[ ind[1] ]


		local tParams = torch.Tensor( mb, 2, ncounts ) 
		tParams:select( 2, 1 ):fill( 0 )
		tParams:select( 2, 2 ):fill( sigN ):log()

		local qParams = torch.Tensor( mb, 2, ncounts )
		qParams:select( 2, 1 ):fill( qMu  ) 
		qParams:select( 2, 2 ):fill( qSig ):log() 

		local likelihood = torch.Tensor( rates:size(1) ):zero()
		local nsamples   = 10000
		likelihoodModuleVB:loadData( dataVB ) 
		for j = 1, nsamples do likelihood:add( likelihoodModuleVB:updateOutput({logRate, qParams, tParams}) ) end
		likelihood:div( - nsamples ) 
		likelihood:add( - likelihood:max() ) 
		local _, ind = torch.sort( likelihood, true )
		ratesEsVB[n] = rates[ ind[1] ]

		table.insert( plot, { n == 1 and 'overdispered LL, variational' or '', rates, likelihood, 'with lines ls 2' } )

		print( n, ratesTrue:size(1) ) 

	end

	m = -100 

	gnuplot.figure(3)
	gnuplot.savePlot( 'overdispersedLogLikelihood.png', plot, 'rate', 'LL', {'','',m,''} ) 

	local plot = {} 
	table.insert( plot, { 'exact likelihood'  , ratesTrue, ratesEstm, '+' } )
	table.insert( plot, { 'variational approx', ratesTrue, ratesEsVB, '+' } )

	gnuplot.figure(4)
	gnuplot.savePlot('overdispersedRecovery.png', plot, 'true rate', 'estimated rate', 'equal' )


end
-- testOverdispersedLikelihood()


function testOverdispersedLikelihoodVB()

	local fixedPrior = true 
	local fixedPost  = true 

	nngraph.setDebug(true)

	local sigN    = 1
 	local nTrials = 50
 	local nCells  =  21 

	local overdispParams  = torch.Tensor( 2, nTrials, nCells )
	local overdispSampler = nn.OverdispersedPoissonSampler( muSigDim, overdispParams ) 

	local ratesTrue = torch.linspace( 1, 5, nCells ) 
	for i = 1, nTrials do
		overdispParams[1][i]:copy( ratesTrue ) 
		overdispParams[2][i]:fill( sigN )
	end
	overdispSampler:loadParams( overdispParams )
	local overdispSample  = overdispSampler:sample()

	local nsamplesNLL        = 10000
	local dataVB			 = torch.Tensor( 1, nTrials, nCells ):copy( overdispSample )

 	local qMu      = 0 
	local qSig     = sigN 
	local mb       = 1
	local muSigDim = 2 

	local likelihoodModuleVB = nn.OverdisperedPoissonNLL_VB_singleSample( mb, muSigDim, dataVB ) 

	require 'common-torch/modules/segment'

	local ratesEstm = torch.randn( mb, nCells ) 

	local plot = {} 

	local m = - math.huge

	local tParams = torch.Tensor( mb, 2, nCells ) 
	tParams:select( 2, 1 ):fill( 0 )
	tParams:select( 2, 2 ):fill( sigN ):log()

	local qParams = torch.Tensor( mb, 2, nTrials, nCells )
	qParams:select( 2, 1 ):fill( qMu  ) 
	qParams:select( 2, 2 ):fill( qSig ):log() 

	local init = { ratesEstm, qParams, tParams } 

	local nparams = 0; for i = 1, #init do nparams = nparams + init[i]:nElement()/mb end 
	local fused = torch.Tensor( mb, nparams ) 
	local segm = Segment( init, 2 ); segm:updateOutput( fused ) 
	local z = segm:updateGradInput( fused, init ):clone()
	segm:updateOutput( z ) 

	likelihoodModuleVB:loadData( dataVB ) 
	local maxiter = 10000
	local gradOutput = torch.Tensor{ 1 } 

	local function opfunc( z )

		local t = segm:updateOutput( z ) 

		local loss = likelihoodModuleVB:updateOutput(    t )
		local dt   = likelihoodModuleVB:updateGradInput( t, gradOutput )

		if fixedPost  then dt[2]:zero() end 
		if fixedPrior then dt[3]:zero() end 

		local dz   = segm:updateGradInput( z, dt ) 

		return loss[1], dz

	end
	require 'optim'
	local config = { learningRate = 0.01 }
	local state  = {} 

	for i = 1, maxiter do 

		-- optim.adam( opfunc, ratesEstm, config, state )
		optim.adam( opfunc, z, config, state )
		print( i, likelihoodModuleVB.output[1] ) 

	end
	ratesEstm = segm:updateOutput( z )[1]:exp()

	local plot = {} 
	table.insert( plot, { ratesTrue, ratesTrue   , 'with  lines ls 1 lc rgb "gray"'} )
	table.insert( plot, { ratesTrue, ratesEstm[1], 'with points ls 1 lc rgb "black"' } ) --'variational approx', 

	local savedir = 'overdispersedRecovery'
	savedir = savedir .. '_nTrials' .. nTrials
	savedir = savedir .. '_maxiter' .. maxiter

	if fixedPrior then savedir = savedir .. '_fixedPrior' end 
	if fixedPost  then savedir = savedir .. '_fixedPost'  end 

	gnuplot.figure(4)
	gnuplot.savePlot( savedir .. '.eps', plot, 'true rate', 'estimated rate', 'equal', {'set yrange [0:10]', 'set xtics 1', 'set ytics 1'} )


end
-- testOverdispersedLikelihoodVB()
]]


