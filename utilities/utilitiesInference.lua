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


	local likelihood
	if     inferenceMeth == 'ML' then 
		likelihood = nn.Neural_NLL_poisson_ML( data, dataMask, trialLength, dim, mb, init )
	elseif inferenceMeth == 'VB' then 
		likelihood = nn.Neural_NLL_poisson_VB( data, dataMask, trialLength, dim, mb, init )
	end
	likelihood:updateOutput( z ) 

	local trajNetwork = findNode( likelihood.network, 'trajectoryFine' )

	local theta01    = findNode( trajNetwork, 'thetaParams01'     ).output
	local thetaPrior = findNode( trajNetwork, 'priorTheta'        ).output

	local trajNetwork = likelihood.network

	local membs_traj = findNode( trajNetwork, 'membTrajectory'    ).output:squeeze()
	local rates_traj = findNode( trajNetwork, 'rateTrajectory'    ).output:squeeze()
	-- local lossMB     = findNode( trajNetwork, 'lossPerBatchSample').output:squeeze()
	local lossMB     = findNode( trajNetwork, 'totalLoss').output--:squeeze()

	local function opfunc( z )

		local loss = likelihood:updateOutput( z )
		local dz   = likelihood:updateGradInput( z )
		return loss[1], dz

	end

	local config = { learningRate = learningRate or 0.01 }
	local state  = {}

	-- local loss = torch.Tensor( maxiter, mb ) 
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

-- function getBestCurvature( results ) 

-- 	local loss = results.loss 

-- 	local lossEnd  = loss:narrow( 1, collectLossFrom+1, maxiter-collectLossFrom )
-- 	local lossMean = lossEnd:mean( 1 )
-- 	local lossStd  = lossEnd:std( 1 )
-- 	lossMean:add( - lossMean:min() )

-- 	_, lossInd = torch.sort( lossMean:squeeze() ) 
-- 	lossMean = lossMean:index( 2, lossInd )
-- 	lossStd  =  lossStd:index( 2, lossInd )

-- 	local lossStdPrime = lossStd:clone():pow(2)
-- 	lossStdPrime:add( lossStdPrime:squeeze()[1] ):div(2):sqrt()

-- 	local dPrime  = lossMean:clone():cdiv( lossStdPrime )
-- 	local include = dPrime:lt( 1 ):double()
-- 	local curv    = results.curv[ maxiter ]:index( 1, lossInd ) 

-- 	-- local bestCurv = curv:cmul( include ):sum() / include:sum()
-- 	local bestCurv = curv[1]

-- 	bestCurv = 180 * bestCurv 

-- 	return bestCurv, lossInd

-- end

function saveLossOverTime( results, analysisDir, cTrue, expname )

	local lossDir = analysisDir .. 'over-time/'
	os.execute( 'mkdir -p ' .. lossDir ) 
	local master   = {} 

	local loss = results.loss
	local curv = results.curv

	local lossEnd  = loss:narrow( 1, collectLossFrom+1, maxiter-collectLossFrom )
	local lossMean = lossEnd:mean( 1 )
	_, lossInd = torch.sort( lossMean:squeeze() )

	local params = { xlabel = 'iteration', ylabel = 'curvature', size = '1,0.5', origin = '0,0', raw = {'set yrange [0:1]', "set format y ''", 'unset ylabel'}} 
	local plot   = { { torch.Tensor(maxiter):fill( cTrue ), 'with lines lc rgb "grey"' } } 
	for n = mb, 1, -1 do table.insert( plot, { curv:select(2,lossInd[n]), 'with lines ls ' .. n } ) end 
	table.insert( master, { plot = plot, params = params } ) 

	local params = { xlabel = 'iteration', ylabel = 'loss', size = '1,0.5', origin = '0,0.5', raw = {'set yrange [*:*]', "set format y ''", 'unset ylabel', 'set border 2', 'unset xtics', 'unset xlabel'} } 
	local plot   = {} 
	for n = mb, 1, -1 do table.insert( plot, { loss:select(2,lossInd[n]), 'with lines ls ' .. n } ) end 
	table.insert( master, { plot = plot, params = params } ) 
	gnuplot.multiPackage( lossDir .. expname:gsub('.t7','.png'), master )

end