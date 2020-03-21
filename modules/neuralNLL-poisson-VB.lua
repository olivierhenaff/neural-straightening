require 'cephes'
require 'randomkit'
require 'nngraph'

require 'common-torch/modules/segment'
require 'common-torch/modules/distCurvAcc-to-Traj'

require 'common-torch/modules/poisson'
require 'common-torch/modules/modulatedPoisson'
require 'common-torch/modules/overdispersedPoisson'
require 'common-torch/modules/overdispersedPoissonShared'

require 'common-torch/modules/SampleGaussian'
require 'common-torch/modules/GaussianKL'

require 'common-torch/modules/triangle'

require 'modules/spikeNL'

local likelihood, parent = torch.class('nn.Neural_NLL_poisson_VB', 'nn.Module')

function narrowTable( ind1, ind2 )

	local inode = nn.Identity()()

	local output = {} 

	for i = ind1, ind2 do table.insert( output, nn.SelectTable(i)( inode ) ) end

	local network = nn.gModule( { inode }, output ) 

	return network 

end

function trajectoryNetwork( nsmpl, deterministic )

	local muSigDim = 2 

	local losses = {} 

	local params = nn.Identity()()

	local      d    			  = nn.SelectTable( 1 )( params ):annotate{ name =  'preDistParams' }
	local      t    			  = nn.SelectTable( 2 )( params ):annotate{ name = 'preThetaParams' }
	local      a    			  = nn.SelectTable( 3 )( params ):annotate{ name =   'preAccParams' }
	local      v0   			  = nn.SelectTable( 4 )( params ):annotate{ name = 'v0' }
	local      z0   			  = nn.SelectTable( 5 )( params ):annotate{ name = 'z0' }
	local priorD    			  = nn.SelectTable( 6 )( params ):annotate{ name = 'priorDist'  }
	local priorT    			  = nn.SelectTable( 7 )( params ):annotate{ name = 'priorTheta' }
	local priorA    			  = nn.SelectTable( 8 )( params ):annotate{ name = 'priorAcc'   }
	local priorAccRotationInverse = nn.SelectTable( 9 )( params ):annotate{ name = 'priorAccRotationInverse'  }

	if ditherSamplesJoint then 

		priorD = nn.Narrow( 1, 1, mb/ditherSamplesJoint )( priorD )
		priorD = nn.Replicate( ditherSamplesJoint, 1 )(    priorD ) 
		priorD = nn.Contiguous()( priorD ) 
		priorD = nn.View( mb, 2 )(                         priorD )

		priorT = nn.Narrow( 1, 1, mb/ditherSamplesJoint )( priorT )
		priorT = nn.Replicate( ditherSamplesJoint, 1 )(    priorT ) 
		priorT = nn.Contiguous()( priorT ) 
		priorT = nn.View( mb, 2 )(                         priorT )

	end

	--- preprocess params 

	local  muT = nn.Narrow( 2, 1, 1 )( t )
	local sigT = nn.Narrow( 2, 2, 1 )( t )
	muT = nn.Sigmoid()(              muT ):annotate{ name = 'thetaParams01' }
	muT = nn.MulConstant( math.pi )( muT ):annotate{ name = 'thetaParamsPi' }
	t = nn.JoinTable( 2 )( { muT, sigT } )

	a = nn.View( mb, 2, dim*(nsmpl-2) )( a )

	priorD = nn.Replicate( nsmpl-1, 3 )( priorD )
	priorT = nn.Replicate( nsmpl-2, 3 )( priorT )
	priorA = nn.Replicate( nsmpl-2, 4 )( priorA )
	priorA = nn.MulConstant(1)( priorA )
	priorA = nn.View( mb, 2, dim*(nsmpl-2) )( priorA )

	if     priorAcc:find('Zero') then

		priorA = nn.MulConstant(0)( priorA )

	elseif priorAcc:find('Sig' ) then 

		local  muA = nn.Narrow( 2, 1, 1 )( priorA )
		local sigA = nn.Narrow( 2, 2, 1 )( priorA )
		muA = nn.MulConstant(0)( muA )
		priorA = nn.JoinTable( 2 )( { muA, sigA } )

	end

	table.insert( losses, GaussianKLdiagonalLearnedMuStd( mb, muSigDim )({ d, priorD }) )
	table.insert( losses, GaussianKLdiagonalLearnedMuStd( mb, muSigDim )({ t, priorT }) )

	if deterministic then 
		t = setLogStdToNegInf( muSigDim )( t ) 
		d = setLogStdToNegInf( muSigDim )( d ) 
	end 
	d = SampleGaussianMuLogStd( muSigDim )( d )
	t = SampleGaussianMuLogStd( muSigDim )( t )


	if priorAcc == 'FullZero' then priorAccRotationInverse = nn.MulConstant(  0 )( priorAccRotationInverse ) end 

	local      a_reshaped = nn.View( mb, 2, dim, nsmpl-2 )(      a ) 
	local priorA_reshaped = nn.View( mb, 2, dim, nsmpl-2 )( priorA ) 
	     a_reshaped = nn.Transpose({3,4})(      a_reshaped ) 
	priorA_reshaped = nn.Transpose({3,4})( priorA_reshaped ) 
	local klA 
	if priorAcc:find('FullWishart') then 

		if priorAcc:find('FullWishartLearned') then 

			local nu = priorAcc:gsub( 'FullWishartLearned', '' )
			klA = GaussianKLdiagonalLearnedMuStdPriorFullWishartLearned( mb, muSigDim, nsmpl-2, dim, tonumber(nu) )({ a_reshaped, priorA_reshaped, priorAccRotationInverse, logSigP })

		else

			local p = priorAcc:gsub( 'FullWishart', '' )

			if priorAcc:find('Sig') then 
				p = p:gsub('Sig', '' )
			end

			klA = GaussianKLdiagonalLearnedMuStdPriorFullWishart(        mb, muSigDim, nsmpl-2, dim, tonumber(p)  )({ a_reshaped, priorA_reshaped, priorAccRotationInverse })

		end

	else
		klA = GaussianKLdiagonalLearnedMuStdPriorFull( mb, muSigDim, nsmpl-2, dim )({ a_reshaped, priorA_reshaped, priorAccRotationInverse })
	end
	table.insert( losses, klA )

	if deterministic then a = setLogStdToNegInf( muSigDim )( a ) end 

	a = SampleGaussianMuLogStd( muSigDim )( a )
	a = nn.View( mb, dim, nsmpl-2 )( a ) 


	--- run samples through transfer functions 

	d = nn.SoftPlus()( d ):annotate{ name = 'distSample' }

	--- construct trajectory and evaluate likelihood 

	local memb = distCurvAccToTraj( mb, dim, nsmpl, true, true )({ d, t, a, v0, z0 })
	-- memb = nn.Transpose({2,3})( memb ):annotate{ name = 'membTrajectory' }


	losses = nn.CAddTable()( losses ) 

	local network = nn.gModule( { params }, { memb, losses } ) 

	return network 

end

function likelihood:__init( data, dataMask, trialLength, dim, mb, init, deterministic )
-- function likelihood:__init( data, dataMask, trialLength, dim, init, deterministic )

	parent.__init(self)

	local muSigDim = 2 
	-- local mb       = init[1]:size(1)

	local inode  = nn.Identity()()
	local params = Segment( init, 2 )( inode ):annotate{ name = 'segment'  }

	local memb, losses, p3  

	if multiscale then 

		local p1 = narrowTable(  1,  9 )( params ) 
		local p2 = narrowTable( 10, 18 )( params ) 
		      p3 = narrowTable( 19, 24 )( params ) 

		local p1 = trajectoryNetwork( math.ceil( nsmpl/2 ), deterministic )( p1 ):annotate{ name = 'trajectoryCoarse' }
		local p2 = trajectoryNetwork(            nsmpl    , deterministic )( p2 ):annotate{ name = 'trajectoryFine'   }

		local t1 = nn.SelectTable( 1 )( p1 ) 
		local t2 = nn.SelectTable( 1 )( p2 ) 

		local l1 = nn.SelectTable( 2 )( p1 ) 
		local l2 = nn.SelectTable( 2 )( p2 ) 
		losses = { l1, l2 } 

		t1 = nn.View( mb * dim, -1 )( t1 ) 
		t1 = multiscaleBkw( multiscale.p[1], nsmpl )( t1 )
		t1 = nn.View( mb , dim, -1 )( t1 ) 
		memb = nn.CAddTable()({ t1, t2 })

	else

		local p1 = narrowTable(  1,  9 )( params ) 
		      -- p3 = narrowTable( 10, 15 )( params ) 
		      p3 = narrowTable( 10, 14 )( params ) 

		local p1 = trajectoryNetwork(            nsmpl, deterministic )( p1 ):annotate{ name = 'trajectoryFine'   }
		local t1 = nn.SelectTable( 1 )( p1 ) 
		local l1 = nn.SelectTable( 2 )( p1 ) 
		losses = { l1 } 

		memb = t1 

	end
	memb = nn.Transpose({2,3})( memb ):annotate{ name = 'membTrajectory' }

	local logSigN   = nn.SelectTable( 1 )( p3 ) 
	local qParams   = nn.SelectTable( 2 )( p3 ):annotate{ name = 'noisePosterior' } 
	local coupling  = nn.SelectTable( 3 )( p3 )
	local sParams   = nn.SelectTable( 4 )( p3 ):annotate{ name = 'sParams' }
	local embedding = nn.SelectTable( 5 )( p3 )
	-- local memb_mean = nn.SelectTable( 6 )( p3 ):annotate{ name = 'memb_mean' }
	-- memb_mean = nn.MulConstant( 0 )( memb_mean ) 

	logSigN  = nn.Select( 1, 1 )( logSigN ) 
	logSigN  = nn.Replicate(    mb, 1 )( logSigN ):annotate{ name = 'logSigN' } 

	coupling = nn.Select( 1, 1 )(     coupling )
	coupling = nn.Replicate( mb, 1 )( coupling ):annotate{ name = 'coupling' } 

	local priorEmbedding = nn.MulConstant( 0 )( embedding ) 
	local klEmbedding    = GaussianKLdiagonalLearnedMuStd( mb, muSigDim )({ embedding, priorEmbedding })
	table.insert( losses, klEmbedding ) 

	if deterministic then embedding = setLogStdToNegInf( muSigDim )( embedding ) end 

	embedding = SampleGaussianMuLogStd( muSigDim )( embedding ) 
	embedding = grahamSchmidtMatrix( mb, data:size(3), dim )( embedding ):annotate{ name = 'embedding' } 
	embedding = nn.Replicate( nsmpl, 2 )( embedding ) 
	embedding = nn.Contiguous()( embedding ) 
	embedding = nn.View( -1, data:size(3), dim )( embedding ) 

	memb = nn.View( -1, dim, 1 )( memb ) 
	memb = nn.MM()({ embedding, memb })
	memb = nn.View( mb, nsmpl, data:size(3) )( memb ) 
	-- memb = nn.CAddTable()({ memb, nn.Replicate( nsmpl, 2 )( memb_mean ) })

	local rateOffset = 1e-3

	local rate
	if spikeNLmodel == 'SquaredSoftPlus' then 
		rate = nn.AddConstant( rateOffset )( nn.SoftPlus()( memb ) ):annotate{ name = 'preRateTrajectory' } 
		rate = nn.Square()( rate ):annotate{ name = 'rateTrajectory' }
	elseif spikeNLmodel == 'SquaredSinhSoftPlus' then 

		local sigG = effectiveSigG( data )({ qParams, sParams, coupling }):annotate{ name = 'sigG' } 
		sigG = nn.Replicate( nsmpl, 2 )( sigG )

		-- error('SquaredSinhSoftPlus')

		rate = nn.AddConstant( rateOffset )( nn.SoftPlus()( memb ) ):annotate{ name = 'preRateTrajectory' }
		rate = nn.MulConstant( 0.5 )( nn.CMulTable()({ sigG, rate }) )
		rate = HyperbolicSine()( rate )
		rate = nn.CDivTable()({ rate, sigG })
		rate = nn.Square()( rate ):annotate{ name = 'rateTrajectory' }

	else
		local spikeNL = nn[spikeNLmodel]()
		rate = spikeNL(       memb ):annotate{ name = 'preRateTrajectory' }
		rate = nn.Identity()( rate ):annotate{ name =    'rateTrajectory' }
	end

	-- rate
	rate = nn.Replicate( data:size(1), 2 )( rate )
	
	-- local mulLength = nn.CMul( trialLength:size() )
	-- mulLength.weight:copy( trialLength )
	-- rate = mulLength( rate )
	rate = nn.MulConstant( trialDuration )( rate )
	-- rate = nn.AddConstant( 1e-3 )( rate )

	-- rate:annotate{ name = 'rateTrajectory' }


	logSigN = nn.Replicate( nsmpl, 2 )( logSigN )
	if spikeCountDistributionModel:find('SharedOnly') then 
		logSigN = nn.MulConstant(  0 )( logSigN )
		logSigN = nn.AddConstant( -9 )( logSigN )
	else
		logSigN = nn.MulConstant( 1 )( logSigN )
	end
	-- logSigN = nn.View( mb, 1, nsmpl, dim )( logSigN ) 
	logSigN = nn.View( mb, 1, nsmpl, data:size(3) )( logSigN )

	local muN = nn.Exp()( nn.MulConstant(2)( logSigN ) )

	local couplingVar = nn.Sum(3)( nn.Square()( coupling ) )
	couplingVar = nn.Replicate( nsmpl, 2 )( couplingVar )
	muN = nn.CAddTable()({ muN, couplingVar })
	muN = nn.MulConstant(-0.5)( muN )

	local tParams = nn.JoinTable(2)({ muN, logSigN })
	tParams = nn.MulConstant(1)( tParams )

	self.nll = nn.OverdisperedPoissonNLL_VB_singleSample_shared( mb, muSigDim, data, dataMask )
	local nll = self.nll({ nn.Log()(rate), qParams, tParams, coupling, sParams }):annotate{ name = 'likelihood' }
	nll = nn.View( mb, -1 )( nll )
	nll = nn.Sum( 2 )( nll )

	table.insert( losses, nll )
	local loss = nn.CAddTable()( losses ):annotate{ name = 'lossPerBatchSample' }
	loss = nn.View( 1, -1 )( loss )
	loss = nn.Sum( 2 )( loss ):annotate{ name = 'totalLoss' }

	self.network = nn.gModule({ inode }, { loss })

	-- self:loadData( data, dataMask )

	-- self.gradOutput = torch.Tensor( mb ):fill( 1 ) 
	self.gradOutput = torch.Tensor( 1 ):fill( 1 ) 

	self:cuda()

end

function likelihood:updateOutput( input ) 

	self.output = self.network:updateOutput( input ) 

	return self.output 

end

function likelihood:updateGradInput( input ) 

	self.gradInput = self.network:updateGradInput( input, self.gradOutput )

	return self.gradInput 

end

function likelihood:loadData( data, dataMask )

	-- print( 'load data likelihood' )

	self.nll:loadData( data, dataMask ) 

end

function likelihood:cuda()

	self.network:float()
	self.network:cuda()

	self.gradOutput = self.gradOutput:cuda()

end

