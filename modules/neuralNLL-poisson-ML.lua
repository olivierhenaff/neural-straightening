require 'cephes'
require 'randomkit'
require 'nngraph'

-- require 'untangling-perceptual/modules/SampleGaussian'
-- require 'untangling-perceptual/modules/GaussianKL'

require 'common-torch/modules/segment'
require 'common-torch/modules/distCurvAcc-to-Traj'
require 'common-torch/modules/poisson'


local likelihood, parent = torch.class('nn.Neural_NLL_poisson_ML', 'nn.Module')

function likelihood:__init( data, dim, mb, init )

	parent.__init(self)

	local losses = {} 

	local inode  = nn.Identity()()
	local params = Segment( init, 2 )( inode ):annotate{ name = 'segment'  }

	local      d  = nn.SelectTable( 1 )( params ):annotate{ name =  'preDistParams' }
	local      t  = nn.SelectTable( 2 )( params ):annotate{ name = 'preThetaParams' }
	local      a  = nn.SelectTable( 3 )( params ):annotate{ name =   'preAccParams' }
	local      v0 = nn.SelectTable( 4 )( params ):annotate{ name = 'v0' }
	local      z0 = nn.SelectTable( 5 )( params ):annotate{ name = 'z0' }
	-- local priorD  = nn.SelectTable( 4 )( params ):annotate{ name = 'priorDist'  }
	-- local priorT  = nn.SelectTable( 5 )( params ):annotate{ name = 'priorTheta' }

	--- preprocess params 

	t = nn.Sigmoid()(              t ):annotate{ name = 'thetaParams01' }
	t = nn.MulConstant( math.pi )( t ):annotate{ name = 'thetaParamsPi' }

	-- local  muT = nn.Narrow( 2, 1, 1 )( t ) 
	-- local sigT = nn.Narrow( 2, 2, 1 )( t ) 
	-- muT = nn.Sigmoid()(              muT ):annotate{ name = 'thetaParams01' }
	-- muT = nn.MulConstant( math.pi )( muT ):annotate{ name = 'thetaParamsPi' }
	-- t = nn.JoinTable( 2 )( { muT, sigT } )

	-- a      = nn.View( mb, 2, (nsmpl-2)*dim )( a )

	-- priorD = nn.Replicate( nsmpl-1, 3 )( priorD )
	-- priorT = nn.Replicate( nsmpl-2, 3 )( priorT )
--[[
	if constrain and constrain.priorT then 

		self.gatePriorT = nn.GradientReversal(0)
		priorT = self.gatePriorT( priorT ) 

	end
]]
	-- local priorA = nn.MulConstant(0)( a ):annotate{ name = 'priorAcc'   }

	-- local paramsPost  = {      d,      t,      a } 
	-- local paramsPrior = { priorD, priorT, priorA }
	-- rotation = lowerTriangle( nZ )( rotation ) 

	--- compute kl divergence 

	-- local paramsPost  = nn.JoinTable( 3 )( paramsPost  )
	-- local paramsPrior = nn.JoinTable( 3 )( paramsPrior )
	-- local kl          = GaussianKLdiagonalLearnedMuStdFull( mb, 2, nZ, 'scalar', rotateFirst )({ paramsPost, paramsPrior, rotation })
	-- table.insert( losses, kl ) 

	--- sample from posterior 

	-- if sampleMode then 

	-- 	if sampleMode.all or sampleMode.d then d = setLogStdToNegInf( 2 )( d ) end 
	-- 	if sampleMode.all or sampleMode.t then t = setLogStdToNegInf( 2 )( t ) end 
	-- 	if sampleMode.all or sampleMode.a then a = setLogStdToNegInf( 2 )( a ) end

 -- 	end

	-- local paramsPost  = { d, t, a } 
	-- if lapseDim > 1 then table.insert( paramsPost, l ) end 
 -- 	paramsPost = nn.JoinTable( 3 )( paramsPost )

 -- 	local samplePost
 -- 	if rotateFirst then 
 -- 		samplePost = SampleGaussianRotationLogStdMu( 2, mb, nZ )( {paramsPost, rotation } ):annotate{ name = 'distThetaAccLapseSample' }
 -- 	else
	-- 	samplePost = SampleGaussianMuLogStdRotation( 2, mb, nZ )( {paramsPost, rotation } ):annotate{ name = 'distThetaAccLapseSample' }
 -- 	end

	-- d = nn.Narrow( 2, 1 , nsmpl-1                              )( samplePost ) 
	-- t = nn.Narrow( 2, 1 + nsmpl-1 , nsmpl-2                    )( samplePost ) 
	-- a = nn.Narrow( 2, 1 + nsmpl-1 + nsmpl-2 , (nsmpl-2)*dim    )( samplePost )

	-- d = nn.Select( 2, 1 )( d ) 
	-- t = nn.Select( 2, 1 )( t ) 
	-- a = nn.Select( 2, 1 )( a ) 

	--- run samples through transfer functions 

	d = nn.SoftPlus()( d ):annotate{ name = 'distSample' }

	-- a = nn.View( mb, nsmpl-2, dim )( a )
	-- a = nn.Transpose({2,3})( a ):annotate{ name = 'accSample' } 

	--- construct trajectory and evaluate likelihood 

	local memb = distCurvAccToTraj( mb, dim, nsmpl, true, true )({ d, t, a, v0, z0 })
	memb = nn.Transpose({2,3})( memb ):annotate{ name = 'membTrajectory' }

	local spikeNL = nn[spikeNLstr]() 

	local rate = spikeNL( memb ):annotate{ name = 'rateTrajectory' }

	self.poisson = nn.PoissonNLL_unnormalized( data ) 
	local nll = self.poisson( rate ) 
	nll = nn.View( mb, -1 )( nll ) 
	nll = nn.Sum( 2 )( nll ) 

	local loss 
	if #losses > 0 then

		table.insert( losses, nll )
		loss = nn.CAddTable()( losses ):annotate{ name = 'lossPerBatchSample' }

	else

		loss = nll

	end

	self.network = nn.gModule({ inode }, { loss })

	self:loadData( data )

	self.gradOutput = torch.Tensor( mb ):fill( 1 ) 

end

function likelihood:updateOutput( input ) 

	self.output = self.network:updateOutput( input ) 

	return self.output 

end

function likelihood:updateGradInput( input ) 

	self.gradInput = self.network:updateGradInput( input, self.gradOutput )

	return self.gradInput 

end

function likelihood:loadData( data )

	self.poisson:loadData( data ) 

end
--[[
function likelihood:descendPriorT( prop )

	self.gatePriorT:setLambda( -prop )

end

function likelihood:descendPostT( prop )

	self.gatePostT:setLambda(  -prop )

end
]]

