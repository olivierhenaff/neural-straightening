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

require 'untangling-neural/modules/spikeNL'

local likelihood, parent = torch.class('nn.Neural_NLL_poisson_VB', 'nn.Module')

function likelihood:__init( data, dataMask, trialLength, dim, mb, init, deterministic )
-- function likelihood:__init( data, dataMask, trialLength, dim, init, deterministic )

	parent.__init(self)

	local muSigDim = 2 
	-- local mb       = init[1]:size(1)

	local losses = {} 

	local inode  = nn.Identity()()
	local params = Segment( init, 2 )( inode ):annotate{ name = 'segment'  }

	local      d    = nn.SelectTable( 1 )( params ):annotate{ name =  'preDistParams' }
	local      t    = nn.SelectTable( 2 )( params ):annotate{ name = 'preThetaParams' }
	local      a    = nn.SelectTable( 3 )( params ):annotate{ name =   'preAccParams' }
	local      v0   = nn.SelectTable( 4 )( params ):annotate{ name = 'v0' }
	local      z0   = nn.SelectTable( 5 )( params ):annotate{ name = 'z0' }
	local priorD    = nn.SelectTable( 6 )( params ):annotate{ name = 'priorDist'  }
	local priorT    = nn.SelectTable( 7 )( params ):annotate{ name = 'priorTheta' }
	local priorA    = nn.SelectTable( 8 )( params ):annotate{ name = 'priorAcc'   }

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

	if     priorAcc == 'Zero' then 

		priorA = nn.MulConstant(0)( priorA )

	elseif priorAcc == 'Sig'  then 

		local  muA = nn.Narrow( 2, 1, 1 )( priorA ) 
		local sigA = nn.Narrow( 2, 2, 1 )( priorA ) 
		muA = nn.MulConstant(0)( muA )
		priorA = nn.JoinTable( 2 )( { muA, sigA } )

	end

	if priorAcc:find('Full') then 

		local klD = GaussianKLdiagonalLearnedMuStd( mb, muSigDim )({ d, priorD })
		local klT = GaussianKLdiagonalLearnedMuStd( mb, muSigDim )({ t, priorT })		

		table.insert( losses, klD )
		table.insert( losses, klT )

		if deterministic then 

			d = setLogStdToNegInf( muSigDim )( d ) 
			t = setLogStdToNegInf( muSigDim )( t ) 

		end 

		d = SampleGaussianMuLogStd( muSigDim )( d )
		t = SampleGaussianMuLogStd( muSigDim )( t )

		local priorAccRotationInverse = nn.SelectTable( 9 )( params ):annotate{ name = 'priorAccRotationInverse'  }

		if priorAcc == 'FullZero' then priorAccRotationInverse = nn.MulConstant(  0 )( priorAccRotationInverse ) end 

		local      a_reshaped = nn.View( mb, 2, dim, nsmpl-2 )(      a ) 
		local priorA_reshaped = nn.View( mb, 2, dim, nsmpl-2 )( priorA ) 
		     a_reshaped = nn.Transpose({3,4})(      a_reshaped ) 
		priorA_reshaped = nn.Transpose({3,4})( priorA_reshaped ) 
		local klA 
		if priorAcc:find('FullWishart') then 
			local p = priorAcc:gsub( 'FullWishart', '' )
			klA = GaussianKLdiagonalLearnedMuStdPriorFullWishart( mb, muSigDim, nsmpl-2, dim, tonumber(p) )({ a_reshaped, priorA_reshaped, priorAccRotationInverse })
		else
			klA = GaussianKLdiagonalLearnedMuStdPriorFull( mb, muSigDim, nsmpl-2, dim )({ a_reshaped, priorA_reshaped, priorAccRotationInverse })
		end
		table.insert( losses, klA )

		if deterministic then a = setLogStdToNegInf( muSigDim )( a ) end 

		a = SampleGaussianMuLogStd( muSigDim )( a )
		a = nn.View( mb, dim, nsmpl-2 )( a ) 

	else

		local paramsPost  = nn.JoinTable( 3 )( {      d,      t,      a } )
		local paramsPrior = nn.JoinTable( 3 )( { priorD, priorT, priorA } )
		local kl = GaussianKLdiagonalLearnedMuStd( mb, muSigDim )({ paramsPost, paramsPrior })
		table.insert( losses, kl ) 

		if deterministic then paramsPost = setLogStdToNegInf( muSigDim )( paramsPost ) end 

		local samplePost = SampleGaussianMuLogStd( muSigDim )( paramsPost ) 

		d = nn.Narrow( 2, 1 , nsmpl-1                           )( samplePost ) 
		t = nn.Narrow( 2, 1 + nsmpl-1 , nsmpl-2                 )( samplePost ) 
		a = nn.Narrow( 2, 1 + nsmpl-1 + nsmpl-2 , (nsmpl-2)*dim )( samplePost )
		a = nn.View( mb, dim, nsmpl-2 )( a ) 

	end

	--- run samples through transfer functions 

	d = nn.SoftPlus()( d ):annotate{ name = 'distSample' }

	--- construct trajectory and evaluate likelihood 

	local memb = distCurvAccToTraj( mb, dim, nsmpl, true, true )({ d, t, a, v0, z0 })
	memb = nn.Transpose({2,3})( memb ):annotate{ name = 'membTrajectory' }

	local rate --spikeNL
	if spikeNLmodel == 'SquaredSoftPlus' then 
		rate = nn.SoftPlus()( memb ):annotate{ name = 'preRateTrajectory' } 
		rate = nn.Square()( rate ):annotate{ name = 'rateTrajectory' }
		-- spikeNL = nn.Sequential():add( nn.SoftPlus() ):add( nn.Square() ) 
	elseif spikeNLmodel == 'WhitenModulatedPoisson' then 

		local logSigN = nn.SelectTable( 10 )( params )--:annotate{ name = 'logSigN' } 
		local sigG2   = nn.Exp()( nn.MulConstant(2)( logSigN ) )
		sigG2 = nn.MulConstant( trialDuration )( sigG2 )
		if spikeCountDistributionModel:find('Shared') then 
			local coupling = nn.SelectTable( 12 )( params )
			local couplingVar = nn.Sum(3)( nn.Square()( coupling ) )
			sigG2 = nn.CAddTable()({ sigG2, couplingVar })
		end
		sigG2 = nn.Replicate( nsmpl, 2 )( sigG2 ) 

		rate = nn.SoftPlus()( memb ):annotate{ name = 'preRateTrajectory' }
		rate = nn.CMulTable()({ rate, nn.Sqrt()( sigG2 ) })
		rate = nn.MulConstant( 0.5 )( rate ) 

		local r1 = nn.Exp()( nn.MulConstant( 1)( rate )  ) 
		local r2 = nn.Exp()( nn.MulConstant(-1)( rate ) ) 
		rate = nn.CSubTable()({r1,r2})
		rate = nn.MulConstant( 0.5 )( rate ) 
		rate = nn.Square()( rate ) 
		rate = nn.CDivTable()({ rate, sigG2 }):annotate{ name = 'rateTrajectory' }

	else
		local spikeNL = nn[spikeNLmodel]() 
		rate = spikeNL( memb ):annotate{ name = 'rateTrajectory' }
	end

	-- rate
	rate = nn.Replicate( data:size(1), 2 )( rate ) 
	
	-- local mulLength = nn.CMul( trialLength:size() )
	-- mulLength.weight:copy( trialLength ) 
	-- rate = mulLength( rate ) 
	rate = nn.MulConstant( trialDuration )( rate ) 

	-- rate:annotate{ name = 'rateTrajectory' }

	local nll 
	if     spikeCountDistributionModel == 'Poisson' then 

		-- self.nll = nn.PoissonNLL_normalized( data, dataMask )
		self.nll = nn.PoissonNLL_normalized_masked( mb, data, dataMask )
		nll = self.nll( rate )

	elseif spikeCountDistributionModel:find( 'Modulated' ) then 

		-- self.nll = nn.ModulatedPoissonNLL( data, dataMask ) 
		self.nll = nn.ModulatedPoissonNLL_masked( mb, data, dataMask ) 

		local logSigG_cheat = spikeCountDistributionModel:gsub( 'Modulated', '' )
		logSigG_cheat = tonumber( logSigG_cheat )
		local logSigG

		if logSigG_cheat and logSigG_cheat > 0 then 

			logSigG_cheat = math.log( logSigG_cheat ) 
			logSigG = nn.MulConstant( 0 )( rate ) 
			logSigG = nn.AddConstant( logSigG_cheat )( logSigG ) 

		else

			logSigG = nn.SelectTable( 10 )( params )
			logSigG = nn.Replicate( nsmpl, 2 )( logSigG ):annotate{ name = 'logSigG' } 
			logSigG = nn.Replicate( data:size(1), 2 )( logSigG )

		end

		nll = self.nll({ rate, logSigG }):annotate{ name = 'nll' }

	elseif spikeCountDistributionModel:find( 'Overdispersed' ) then 

		local logSigN = nn.SelectTable( 10 )( params ) 
		logSigN = nn.Select( 1, 1 )( logSigN ) 
		logSigN = nn.Replicate(    mb, 1 )( logSigN ):annotate{ name = 'logSigN' } 
		logSigN = nn.Replicate( nsmpl, 2 )( logSigN )
		if spikeCountDistributionModel:find('SharedOnly') then 
			logSigN = nn.MulConstant(  0 )( logSigN ) 
			logSigN = nn.AddConstant( -9 )( logSigN )
		else
			logSigN = nn.MulConstant( 1 )( logSigN )
		end
		logSigN = nn.View( mb, 1, nsmpl, dim )( logSigN ) 

		local muN = nn.Exp()( nn.MulConstant(2)( logSigN ) )
		if spikeCountDistributionModel:find('Shared') then 
			local coupling = nn.SelectTable( 12 )( params )
			coupling = nn.Select( 1, 1 )(     coupling )
			coupling = nn.Replicate( mb, 1 )( coupling ) 
			local couplingVar = nn.Sum(3)( nn.Square()( coupling ) )
			couplingVar = nn.Replicate( nsmpl, 2 )( couplingVar ) 
			muN = nn.CAddTable()({ muN, couplingVar })
		end
		muN = nn.MulConstant(-0.5)( muN ) 

		local tParams = nn.JoinTable(2)({ muN, logSigN })
		tParams = nn.MulConstant(1)( tParams )

		local qParams = nn.SelectTable( 11 )( params ):annotate{ name = 'noisePosterior' } 

		if spikeCountDistributionModel:find('Shared') then

			local coupling = nn.SelectTable( 12 )( params ):annotate{ name = 'coupling' } 
			local sParams  = nn.SelectTable( 13 )( params ):annotate{ name = 'sParams' }

			self.nll = nn.OverdisperedPoissonNLL_VB_singleSample_shared( mb, muSigDim, data, dataMask )
			nll = self.nll({ nn.Log()(rate), qParams, tParams, coupling, sParams }):annotate{ name = 'likelihood' }

		else

			self.nll = nn.OverdisperedPoissonNLL_VB_singleSample( mb, muSigDim, data, dataMask )
			-- nll = self.nll({ nn.Log()(rate), qParams, tParams })
			nll = self.nll({ rate, qParams, tParams }):annotate{ name = 'likelihood' }

		end

	end
	 
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

	print( 'load data likelihood' )

	self.nll:loadData( data, dataMask ) 

end

function likelihood:cuda()

	self.network:cuda()

	self.gradOutput = self.gradOutput:cuda()

end

