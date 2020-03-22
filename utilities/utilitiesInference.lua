require 'optim'
require 'utilities/utilitiesInit'
require 'modules/neuralNLL-poisson-VB'

function inferCurvature( data, dataMask, trialLength ) 

	local init, curvML = initialize( data, dataMask, trialLength )

	local init = nn.FlattenTable():updateOutput( init )

	local nparams = 0; for i = 1, #init do nparams = nparams + init[i]:nElement()/mb end 
	local fused = torch.Tensor( mb, nparams ) 
	local segm = Segment( init, 2 ); segm:updateOutput( fused ) 
	local z = segm:updateGradInput( fused, init ):cuda()

	local likelihood = nn.Neural_NLL_poisson_VB(data, dataMask, trialLength, dim, mb, init)
	likelihood:updateOutput( z ) 

	local trajNetwork = findNode( likelihood.network, 'trajectoryFine' )

	local theta01    = findNode( trajNetwork, 'thetaParams01'     ).output
	local thetaPrior = findNode( trajNetwork, 'priorTheta'        ).output

	local trajNetwork = likelihood.network

	local membs_traj = findNode( trajNetwork, 'membTrajectory').output:squeeze()
	local rates_traj = findNode( trajNetwork, 'rateTrajectory').output:squeeze()
	local lossMB     = findNode( trajNetwork, 'totalLoss'     ).output

	local function opfunc( z )

		local loss = likelihood:updateOutput( z )
		local dz   = likelihood:updateGradInput( z )
		return loss[1], dz

	end

	local config = { learningRate = learningRate or 0.01 }
	local state  = {}

	local loss = torch.Tensor( maxiter,  1 ) 
	local curv = torch.Tensor( maxiter, mb ) 
	local printInt = 10
	local  lossInt = torch.Tensor( printInt ) 
	sys.tic()

	for i = 1, maxiter do 

		if i > collectLossFrom then 
			likelihood:updateOutput( z ) 
		else
			optim.adam( opfunc, z, config, state )
		end

		loss[i]:copy( lossMB ) 
		curv[i]:copy( thetaPrior:select(2,1) ):div( math.pi )

		local lossInd = (i-1) % printInt + 1 
		lossInt[lossInd] = lossMB:squeeze()

		if lossInd == printInt then 
			print( 'iteration', i, 'loss', lossInt:mean(), 'time', round( sys.toc(), 3 ) ) 
			sys.tic()
		end

	end

	local results = { curv = curv, loss = loss, curvML = curvML, z = z }

	return results

end

function getBestCurvature( results ) 

	return results.curv[maxiter]:clone():mul( 180 ) 

end