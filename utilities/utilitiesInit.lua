require 'common-torch/utilities/curvature'
require 'randomkit'

function initializeNeuronParams( data, embedding, memb_mean ) 

	local init = {} 

	init = initializeLogSigN(        init, data ) 
	init = initializeNoisePosterior( init, data ) 
	init = initializeCoupling(       init, data ) 
	init = initializeSharedParams(   init, data ) 
	init = initializeEmbedding( init, embedding ) 

	-- table.insert( init, memb_mean )

	return init 

end

function initialize( data, dataMask, trialLength )

	local init, curvML, embedding, memb_mean = initializeTrialAverage( data, dataMask, trialLength, spikeNLmodel )

	table.insert( init, initializeNeuronParams( data, embedding, memb_mean ) )

	return init, curvML 

end

function initializeEmbedding( init, embedding ) 

	local embeddingInit = torch.Tensor( embedding:size(1), 2, embedding:size(2), embedding:size(3) )
	embeddingInit:select( 2, 1 ):copy( embedding ) 
	embeddingInit:select( 2, 2 ):fill( -3 )

	table.insert( init, embeddingInit ) 

	return init 

end

function initializePriorAccRotationInverse( init ) 

	local unmaskedRotation = torch.Tensor( mb, dim, dim ):zero()

	table.insert( init, unmaskedRotation ) 

	return init 

end

function softPlus( x ) 

	if x > 10 then 
		return x 
	else
		return math.log( 1 + math.exp(x) )
	end

end

function softPlusInv( y ) 

	if y > 10 then 
		return y 
	else
		return math.log( math.exp(y) - 1 ) 
	end

end

function sigmoidInv( y )

	return math.log( y / ( 1 - y ) )

end

function sinhInv( y ) 

	return math.log( y + math.sqrt( y^2 + 1 ) )

end

function initializePolarParams( traj )

	local nsmpl = traj:size(1)
	local dim   = traj:size(2) 
	local mb    = traj:size(3)

	local dInit  = torch.Tensor( mb, nsmpl - 1 )
	local cInit  = torch.Tensor( mb, nsmpl - 2 )
	local z0Init = torch.Tensor( mb, dim, 1 )
	local v0Init = torch.Tensor( mb, dim, 1 )
	local aInit  = torch.Tensor( mb, dim, nsmpl - 2 )

	for n = 1, mb do 

		local traj  = traj:select( 3, n )
		local v_emp = traj:narrow( 1, 2, nsmpl-1 ):clone():add( -1, traj:narrow( 1, 1, nsmpl-1 ) )
		for i = 1, nsmpl-1 do v_emp[i]:div( v_emp[i]:norm() ) end 

		-- print( v_emp ) 

		local dHat, cHat = computeDistCurvature( traj )

		 dInit[n]:copy( dHat ):apply( softPlusInv )
		 cInit[n]:copy( cHat ):div(math.pi):apply( sigmoidInv ) 
		z0Init[n]:copy( traj[1] )
		v0Init[n]:copy( traj[2] ):add( -1, traj[1] )
		for i = nsmpl-2, 1, -1 do 

			aInit[n]:select( 2, i ):copy( v_emp[i+1] ):add( -math.cos(cHat[i]), v_emp[i] ) 

			-- if i == 1 then 

			-- 	print( 'n = ' .. n )
			-- 	print( v_emp[i+1] )
			-- 	print( v_emp[i] )

			-- end

		end



	end

	-- print( aInit )
	-- print( aInit:select( 2, 1 ) ) 

	-- print( aInit:eq(aInit):select(2,1) )

	local init   = { dInit, cInit, aInit, v0Init, z0Init }

	return init 

end

function initializeTrialAverage( data, dataMask, trialLength, spikeNL ) 

	--- ML estimator ------------------------------

	-- print(data:min(), data:max())
	-- print(dataMask:min(), dataMask:max() ) 

	-- local rates_emp  =  data:clone():div( trialDuration ):cmul( dataMask ):sum(1):squeeze():cdiv( dataMask:sum(1) )
	local rates_emp  =  data:clone():div( trialDuration ):cmul( dataMask ):sum(1):squeeze():cdiv( dataMask:sum(1):add(1) )
	local membs_emp  = rates_emp:clone():add( 0.1 )
	-- local membs_emp  = rates_emp:clone():add( 0.01 )
	-- local membs_emp  = rates_emp:clone():add( 1e-3 )

	-- print('rates_emp')
	-- print( rates_emp:mean() ) 
	-- print('membs_emp')
	-- print( membs_emp:mean() ) 
	-- error('ssotp here')

	-- print('initializeTrialAverage 1 dim = ', dim )

	dim = math.min( dim, data:size(3) )

	-- print('initializeTrialAverage 2 dim = ', dim )


	-- print( rates_emp:size() ) 

	if     spikeNL == 'Exp'      then 
		membs_emp:log()
	elseif spikeNL == 'SoftPlus' then 
		membs_emp:apply( softPlusInv )
	elseif spikeNL == 'SquaredSoftPlus' then 
		membs_emp:sqrt():apply( softPlusInv )
	elseif spikeNL == 'SquaredSinhSoftPlus' then 
		local sigG = math.sqrt( math.exp( sigNinit^2 ) ) 
		membs_emp:sqrt():mul( sigG ) 
		membs_emp:apply( sinhInv ) 
		membs_emp:mul( 2 / sigG ) 
		membs_emp:apply( softPlusInv ) 
	else
		error('unknown spikeNL')
	end

	-- print( membs_emp:mean() ) 
	-- error('sotp here')


	-- print('mb = ', mb )

	local membs_lowD = torch.Tensor( membs_emp:size(1), dim, membs_emp:size(3) )
	local embedding  = torch.Tensor( mb, membs_emp:size(2), dim ) 
	local membs_mean = torch.Tensor( mb, membs_emp:size(2) ) 

	-- print('membs_mean:size()')
	-- print( membs_mean:size() ) 
	-- print('membs_emp:size()')
	-- print( membs_emp:size() )

	for i = 1, membs_emp:size(3) do 

		local x = membs_emp:select( 3, i ):clone()
		local m = x:mean(1)
		membs_mean[i]:copy( m ) 

		-- print( x:size() ) 
		-- print( x:mean() )

		local u, s, v = torch.svd( x:clone():add( -1, m:expandAs( x ) ) ) 

		-- print( 'dim', dim )
		-- error('stop here')

		v = v:narrow( 2, 1, dim )
		membs_lowD:select( 3, i ):copy( x * v ) 
		embedding[i]:copy( v ) 

	end
	membs_emp = membs_lowD
	membs_emp = membs_emp:transpose( 1, 3 ):contiguous()
	membs_emp = membs_emp:view( mb*dim, nsmpl ) 
	local init 
	if multiscale then 
		init = multiscaleAnalysis( multiscale.p[1], nsmpl ):forward( membs_emp ) 
	else
		init = { membs_emp } 
	end

	for i = 1, #init do

		init[i] = init[i]:view( mb,dim, -1 ):transpose( 1, 3 )
		init[i] = initializePolarParams( init[i] ) 

		local maxInd = 3 
		init[i] = initializeZeroUncertainty( init[i], maxInd )
		init[i] = initializePrior( init[i], maxInd )
		init[i] = initializePriorAccRotationInverse( init[i] )

	end

	local cInit
	if multiscale then 
		cInit = init[2][2]
	else
		cInit = init[1][2] 
	end
	local curvML = cInit:select( 2, 1 ):mean(2):div(math.pi)


	-- print( init )
	-- for i = 1, #init[1] do 
	-- 	print( init[1][i]:mean() ) 
	-- end
	-- error('sotp here')


	return init, curvML, embedding, membs_mean

end

function initializeZeroUncertainty( init, maxInd ) 

	local maxInd = maxInd or #init 

	for i = 1, maxInd do

		local oldSize = init[i]:size()

		local newSize = torch.LongStorage( init[i]:dim()+1 )
		newSize[1] = mb 
		newSize[2] = 2 
		for j = 3, init[i]:dim()+1 do newSize[j] = init[i]:size(j-1) end 

		local newInit = torch.Tensor( newSize ) 
		newInit:select( 2, 1 ):copy( init[i] )
		newInit:select( 2, 2 ):fill( -3 ) 

		init[i] = newInit 

	end 

	return init 

end

function initializePrior( init, maxInd )

	for i = 1, maxInd do 

		local d = init[i]:dim()
		local priorInit = torch.Tensor( init[i]:select( d, 1 ):size() )

		priorInit:select( 2, 1 ):copy( init[i]:select( 2, 1 ):mean(d-1) )

		if init[i]:select( 2, 1 ):size(d-1) == 1 then 

			priorInit:select( 2, 2 ):copy( init[i]:select( 2, 2 ) )

		else

			priorInit:select( 2, 2 ):copy( init[i]:select( 2, 1 ):var(d-1) )
									:add(  init[i]:select( 2, 2 ):clone():mul(2):exp():mean(d-1) )
									:log():div( 2 ) 

		end

		table.insert( init, priorInit )

	end 

	if     curvInitPrior == 'tile' then 

		local margin = 0.1 
		local cInits = torch.linspace( margin, 1-margin, mb-1 ):mul( math.pi )

		for n = 2, mb do init[7][n][1] = cInits[n-1] end 

	elseif curvInitPrior == 'ML'   then

		local thetaTransfer = nn.Sequential():add( nn.Sigmoid() ):add( nn.MulConstant( math.pi ) ) 

		init[7]:select( 2, 1 ):copy( thetaTransfer:updateOutput( init[7]:select( 2, 1 ) ) )

	end

	return init 

end

function initializeLogSigN( init, data ) 

	local dim = data:size( 3 ) 

	local logSigNinit = torch.Tensor( mb, dim ):fill( sigNinit ):log() 

	table.insert( init, logSigNinit )

	return init 

end

function initializeNoisePosterior( init, data )

	local qParams = torch.Tensor( mb, 2, data:size(1), data:size(2), data:size(3) )
	qParams:select( 2, 1 ):fill( 0 ) 
	qParams:select( 2, 2 ):fill( sigNinit ):log() 

	table.insert( init, qParams )

	return init 

end

function initializeCoupling( init, data ) 

	local dim = data:size( 3 ) 

	local coupling = torch.Tensor( mb, dim, nModulatorsModel ):zero() 

	table.insert( init, coupling ) 

	return init 


end

function initializeSharedParams( init, data ) 

	local sParams = torch.Tensor( mb, 2, data:size(1), data:size(2), nModulatorsModel ) 
	sParams:zero()

	table.insert( init, sParams )

	return init 

end

function initializeLogSigP( init ) 

	local logSigPinit = torch.Tensor( mb, dim ):fill( 0 )

	table.insert( init, logSigPinit )

	return init 

end
