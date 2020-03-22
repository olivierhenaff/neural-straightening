require 'common-torch/modules/SampleGaussian'

require 'randomkit'

require 'common-torch/utilities/gnuplot'
require 'common-torch/modules/modulatedPoisson'
require 'common-torch/modules/poisson'
require 'common-torch/modules/GaussianKL'

require 'common-torch/modules/segment'
require 'optim'
require 'common-torch/utilities'

nngraph.setDebug(true)

local likelihood, parent = torch.class( 'nn.OverdisperedPoissonNLL_VB_singleSample', 'nn.Module' )

function likelihood:__init( mb, muSigDim, data, dataMask ) 

	self.poissonNLL  = nn.PoissonNLL_normalized( nn.Unsqueeze(1):updateOutput(data) )

	local inode = nn.Identity()()

	local rate    = nn.SelectTable( 1 )( inode )
	local qParams = nn.SelectTable( 2 )( inode )
	local tParams = nn.SelectTable( 3 )( inode )

	-- membRate = nn.Replicate( data:size(1), 2 )( membRate ) 
	tParams  = nn.Replicate( data:size(1), 3 )( tParams  ) 












	local logGain = SampleGaussianMuLogStd( muSigDim )( qParams ):annotate{ name = 'sampleGaussian' } 
	local    gain = nn.Exp()( logGain ) 
	local poissonRate = nn.CMulTable()({ gain, rate })

	local mask = nn.CMul( dataMask:size() ) 
	mask.weight:copy( dataMask ) 




	local nll = self.poissonNLL( poissonRate )
	nll = mask( nll ) 
	nll = nn.View( mb, -1 )( nll ) 
	nll = nn.Sum(2)( nll ) 

	-- self.network = nn.gModule({ inode }, { nll })

	local  kl = GaussianKLdiagonalLearnedMuStd( mb, muSigDim )({ qParams, tParams })
	local loss = nn.CAddTable()({ nll, kl })
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

	self.poissonNLL:loadData( nn.Unsqueeze(1):updateOutput(data) ) 

end




local sampler, parent = torch.class('nn.OverdispersedPoissonSampler', 'nn.Module')

function sampler:__init( muSigDim, params ) -- params contain { rate, gParams }

	parent.__init(self)

	self.sampleLogGain = nn.SampleGaussianMuStd( muSigDim )

	self:loadParams( params ) 

end

function sampler:loadParams( params ) 

	self.rate = self.rate or torch.Tensor()
	self.rate:resizeAs( params[1] ):copy( params[1] )

	self.gParams = self.gParams or torch.Tensor() 
	self.gParams:resizeAs( params[2] ):copy( params[2] )

end

function sampler:sample()

	local gain = self.sampleLogGain:updateOutput( self.gParams ):exp()
	local rate = gain:cmul( self.rate )

	randomkit.poisson( self.output:resizeAs( self.rate ), rate )

	return self.output 

end 


function testOverdispersedMeanVsVar() 

	local rate  = 4 
	local sigN  = 0.5
	local sigG2 = math.exp(sigN^2) - 1

	local muSigDim = 1 
	local nsamples = 10000

	local overdispParams   = { torch.Tensor( nsamples ), torch.Tensor( 2, nsamples ) }
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
		overdispParams[2][1]:fill(   0  )
		overdispParams[2][2]:fill( sigN )
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

function OverdispersedNLL_wrapper( mb, data, dataMask, init )

	local muSigDim = 2 

	local inode = nn.Identity()()

	local params = Segment( init, 2 )( inode ):annotate{ name = 'segment'  }

	local preRate = nn.SelectTable( 1 )( params )
	local qParams = nn.SelectTable( 2 )( params )
	-- local tParams = nn.SelectTable( 3 )( params )

	local logSigN = nn.SelectTable( 3 )( params ):annotate{ name = 'logSigN'  }
	local     muN = nn.MulConstant( -0.5 )( nn.Square()( nn.Exp()( logSigN ) ) )
	local tParams = nn.JoinTable( 2 )({ muN, logSigN })

	local rate = nn.SoftPlus()( preRate ):annotate{ name = 'rate' } 
	rate = nn.Replicate( data:size(1), 3 )( rate ) 

	tParams = nn.Replicate( data:size(2), 3 )( tParams )

	local nll = nn.OverdisperedPoissonNLL_VB_singleSample( mb, muSigDim, data, dataMask )({ rate, qParams, tParams })

	local network = nn.gModule({inode}, {nll})

	return network

end

function testOverdispersedProbability()

	local rate  = 10 
	-- local sigN  = 0.5
	local sigN  = 1
	local muSigDim = 1 

	local nTrials = 50
	local nsmpl   = 11 
	local dim     =  1 

	local overdispParams   = { torch.Tensor( nTrials, nsmpl ), torch.Tensor( 2, nTrials, nsmpl ) }
	overdispParams[1]:fill( rate )
	overdispParams[2][1]:fill( sigN ):pow(2):div( - 2 ) 
	overdispParams[2][2]:fill( sigN )
	local overdispSampler = nn.OverdispersedPoissonSampler( muSigDim, overdispParams ) 
	local overdispSample  = overdispSampler:sample()

	local mb = 1
	local muSigDim = 2 

	local sigNinit = 0.1 

	local initRate    = torch.Tensor( mb,             nsmpl, dim ):fill( overdispSample:mean() ) 
	local initQparams = torch.Tensor( mb, 2, nTrials, nsmpl, dim ) 
	local initLogSigN = torch.Tensor( mb, 1 ):fill( sigNinit ):log()
	initQparams:select( 2, 1 ):fill( - 0.5 * sigNinit^2 ) 
	initQparams:select( 2, 2 ):fill( math.log( sigNinit ) )

	local init = { initRate, initQparams, initLogSigN } 

	local nparams = 0; for i = 1, #init do nparams = nparams + init[i]:nElement()/mb end 
	local fused = torch.Tensor( mb, nparams ) 
	local segm = Segment( init, 2 ); segm:updateOutput( fused ) 
	local z = segm:updateGradInput( fused, init )

	nll = OverdispersedNLL_wrapper( mb, overdispSample, overdispSample:clone():fill(1), init )
	gradOutput = torch.Tensor{ 1 } 

	nll:updateOutput( z ) 

	local predRate = findNode( nll, 'rate'    ).output
	local predSigN = findNode( nll, 'logSigN' ).output

	local function opfunc( z )

		local loss = nll:updateOutput( z )
		local dz   = nll:updateGradInput( z, gradOutput )
		return loss[1], dz

	end

	local config = { learningRate = 0.01 }
	local state  = {}
	-- local maxiter = 10000
	local maxiter = 5000

	for i = 1, maxiter do 

		-- if i == 5000 then config.learningRate = config.learningRate / 10 end

		optim.adam( opfunc, z, config, state )

		-- print( rate ) 
		print( 'optimizing', i, nll.output:squeeze(), round( math.exp(predSigN:squeeze()), 4 ) ) --round( predRate:squeeze(), 4 ), 
	end

	local function modulatedVariance( rate, sigN )

		return rate + (math.exp(sigN^2)-1)*rate^2

	end

	local result = segm:updateOutput( z )

	local truRate, truVar = rate 				 , modulatedVariance( rate, sigN )
	local empRate, empVar = overdispSample:mean(), overdispSample:var()

	local infRate = nn.SoftPlus():updateOutput( result[1] ):squeeze()
	local infSigN = math.exp( result[3][1][2] )
	local infVar  = modulatedVariance( infRate, infSigN )

	print( 'true mean', round( truRate, 3), 'variance', round( truVar, 3) )
	print( 'emp. mean', round( empRate, 3), 'variance', round( empVar, 3) )
	print( 'inf. mean', round( infRate, 3), 'variance', round( infVar, 3) )

	print( 'true sigN', sigN, 'inf. sigN', infSigN )


end
-- testOverdispersedProbability()










--[[
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
-- testOverdisperse7dLikelihood()


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

