require 'cephes'

function tableToTensor( t ) 

	local x = torch.Tensor( #t, t[1]:size(1) )
	for i = 1, #t do x[i]:copy( t[i] ) end

	return x 

end

require 'utilities/utilitiesDataV1V2'

function pValFromF( df1, df2, F ) 

	local p = F:clone()
	for i = 1, F:size(1) do p[i] = 1 - cephes.fdtr( df1[i], df2[i], F[i] ) end 

	return p 

end

function char2str( t ) 

	local s = '' 
	for i = 1, t:size(2) do s = s .. string.char( t[1][i] ) end 
	return s 

end 

function filterChannels( data, dataMask, criterion )

	local     dataFilt = torch.Tensor( data:size(1), data:size(2), criterion:sum() ) 
	local dataMaskFilt = torch.Tensor( data:size(1), data:size(2), criterion:sum() ) 

	local counter = 1 
	for i = 1, data:size(3) do 
		if criterion[i] == 1 then
			    dataFilt:select( 3, counter ):copy(     data:select( 3, i ) ) 
			dataMaskFilt:select( 3, counter ):copy( dataMask:select( 3, i ) ) 
			counter = counter + 1 
		end
	end

	return dataFilt, dataMaskFilt

end

function loadData( dataset )

	local data, dataMask, trialLength

	if dataset:find('ITdata') then

		data, dataMask, trialLength = loadDataIT( dataset ) 

	else
		
		-- data, dataMask, trialLength = loadDataV1V2( dataset ) 
		data, dataMask, trialLength = loadDataV1V2_dataset( dataset ) 

	end

	if bootstrapInd > 1 then 

		data, dataMask = bootstrapData( data, dataMask )

	end

	-- print( data:size() ) 
	-- error('data:size()')

	return data, dataMask, trialLength 

end

function bootstrapData( data, dataMask ) 

	local d, m = data:clone(), dataMask:clone()

	for i = 1, data:size(1) do 

		for j = 1, data:size(2) do 

			for k = 1, data:size(4) do 

				local randInd = torch.randperm( data:size(1) )[1] 
				d[i][j]:select(2,k):copy(     data[randInd][j]:select(2,k) )
				m[i][j]:select(2,k):copy( dataMask[randInd][j]:select(2,k) )

			end

		end

	end

	-- print( data:size() )

	-- print( dataMask:sum(1) )

	-- local nTrialsPerChannel1 = dataMask:sum(1):sum(2):sum(4)[1][1]
	-- local nTrialsPerChannel2 =        m:sum(1):sum(2):sum(4)[1][1]
	local nTrialsPerChannel1 = dataMask:sum(1):min(2):min(4)[1][1]
	local nTrialsPerChannel2 =        m:sum(1):min(2):min(4)[1][1]

	-- local nCountsPerChannel1 =     data:sum(1):min(2):min(4)[1][1]
	-- local nCountsPerChannel2 =        d:sum(1):min(2):min(4)[1][1]
	local nSpikesPerChannel1 =     data:sum(1):sum(2):min(4)[1][1]
	local nSpikesPerChannel2 =        d:sum(1):sum(2):min(4)[1][1]
	-- local nSpikesPerChannel1 =     data:sum(1):sum(2):sum(4)[1][1]
	-- local nSpikesPerChannel2 =        d:sum(1):sum(2):sum(4)[1][1]

	-- local nTrialsPerChannel1 = dataMask:sum(1):mean(2):mean(4)[1][1]
	-- local nTrialsPerChannel2 =        m:sum(1):mean(2):mean(4)[1][1]
	-- print( torch.cat( nTrialsPerChannel1, nTrialsPerChannel2, 2 ) )
	-- print( torch.cat( nSpikesPerChannel1, nSpikesPerChannel2, 2 ) )
	-- print( nTrialsPerChannel1 )
	-- print( nTrialsPerChannel2 )
	-- print( d:size() ) 
	-- print( m:size() ) 
	-- error('stop here')

	return d, m 

end

function loadStimuliLowD()

	local x = loadStimuli()

	if ditherSamplesJoint then

		local nsmpl = 0
		for i = 1, #x do nsmpl = nsmpl + x[i]:size(2) end

		if nsmpl == 11 then 
			-- x[#x] = torch.cat( x[#x], x[#x]:narrow( 2, x[#x]:size(2), 1 ):clone():zero(), 2 ):clone()
			x[#x] = torch.cat( x[#x], torch.rand( x[#x]:narrow( 2, x[#x]:size(2), 1 ):size() ), 2 )
		end
		x = torch.cat( x, 1 )

	end

	local mb    = x:size(1)
	local nsmpl = x:size(2)
	-- local dim   = nsmpl-1 

	local y = torch.Tensor( mb, nsmpl, dim )

	for i = 1, mb do 

		local x = x[i]:view( x:size(2), -1 ):t():clone()
		x:add( -1, x:mean( 2 ):expandAs( x ) )

		local u, s, v = torch.svd( x )

		local d = dim -- s:size(1)-1 --10 
		u = u:narrow( 2, 1, d ) 
		s = s:narrow( 1, 1, d ) 
		v = v:narrow( 2, 1, d ) 

		local p = torch.diag( s ) * v:t() 

		y[i]:copy( p:t() ) 

	end

	return y 

end

function loadRecoveryCurvatures( datasetInd )

	local curv = { 0.8161, 0.9922, 0.8683, 0.5851, 0.0160, 0.9019, 0.6225, 0.5790, 0.9154, 0.0831, 0.8994, 0.9725, 0.0998, 0.7080, 0.5244, 0.6668, 0.2794, 0.2745, 0.0546, 0.7613, 0.5174, 0.0245, 0.7332, 0.2551, 0.5184, 0.3808, 0.7784, 0.6140, 0.7519, 0.6247, 0.2590, 0.7758, 0.6546, 0.1945, 0.8169, 0.6017, 0.8140, 0.2704, 0.6924, 0.1229, 0.8436, 0.2102, 0.4663, 0.6639, 0.6926, 0.6010, 0.4052, 0.0874, 0.4871, 0.2500, 0.3096, 0.1327, 0.3689, 0.0059, 0.8479, 0.4722, 0.3772, 0.2191, 0.6766, 0.5285, 0.7339, 0.8472, 0.3686, 0.2168, 0.4626, 0.3625, 0.0120, 0.8619, 0.9179, 0.9354, 0.2086, 0.8146, 0.8115, 0.0001, 0.0203, 0.1999, 0.6452, 0.1990, 0.6866, 0.8199, 0.9985, 0.6557, 0.5218, 0.9042, 0.7956, 0.4330, 0.0355, 0.1136, 0.1498, 0.9749, 0.1114, 0.2157, 0.9444, 0.8003, 0.6971, 0.4786, 0.8825, 0.8590, 0.1245, 0.1453 }
	curv = torch.Tensor( curv )

	local recoveryInd  = (datasetInd-1)%5 + 1
	curv = curv:narrow( 1, (recoveryInd-1)*20 + 1, 20 )


	return curv:clone():mul( 180 ) 

end

function makeDataPixel( domain ) 

	-- local deterministic = false 
	local deterministic = true
	local likelihood, z, bestInd = loadResults( 'neural', deterministic ) 
	-- likelihood:updateOutput( z ) 
	local segmOutput     = findNode( likelihood.network, 'segment' ).output
	local currentFR

	if     domain == 'pixel-curv' then 

		local distPixel, pixelCurvature = loadPixelDistCurvature()
		local curvPixel      = pixelCurvature:div( 180 ) 

		-- print( curvPixel:mean(2):mul(180) )
		-- error('stop here')

		segmOutput[2]:select( 2, 1 ):copy( curvPixel ):apply( sigmoidInv ) 
		segmOutput[2]:select( 2, 2 ):fill( -99 ) 

	elseif domain == 'pixel-tile' then 

		local curvRecovery = loadRecoveryCurvatures( datasetInd ):div( 180 ) 

		for i = 1, curvRecovery:size(1) do

			segmOutput[2][i][1]:fill( curvRecovery[i] ):apply( sigmoidInv )  
			segmOutput[2][i][2]:fill( -99 ) 

		end

	elseif domain == 'pixel-full' then 

		local y = findNode( likelihood.network, 'membTrajectory' ).output:clone()


		currentFR = y:mean(2):squeeze()

		local x = loadStimuliLowD():cuda()

		for i = 1, x:size(1) do 

			local m = y[i]:mean()
			if m ~= m or m == math.huge or m == -math.huge then y[i][y:size(2)]:copy( y[i][y:size(2)-1] ) end 

			local u, s, v = torch.svd( y[i] )
			x[i]:copy( x[i] * v:t() )
		end

		local init = torch.Tensor( nsmpl, dim, mb )

		for i = 1, mb do init:select( 3, i ):copy( x[i] ) end 

		init = init:transpose( 1, 3 ):contiguous()
		init = init:view( mb*dim, nsmpl ) 
		if multiscale then 
			init = multiscaleAnalysis( multiscale.p[1], nsmpl ):forward( init ) 
		else
			init = { init } 
		end
		for i = 1, #init do
			init[i] = init[i]:view( mb,dim, -1 ):transpose( 1, 3 )
			init[i] = initializePolarParams( init[i] ) 
			local maxInd = 3 
			init[i] = initializeZeroUncertainty( init[i], maxInd )
			init[i] = initializePrior( init[i], maxInd )
			init[i] = initializePriorAccRotationInverse( init[i] )
		end
		init = nn.FlattenTable():updateOutput( init )


		local pl1 = segmOutput[1]:select( 2, 1 ):clone():apply( softPlus ):sum( 2 ) -- original     path length 

		if     recoveryBoost and recoveryBoost.dist then 
			pl1:mul( recoveryBoost.dist ) 
		elseif recoverySet   and recoverySet.dist   then 
			pl1:fill( recoverySet.dist )
		end

		for i = 1, 3 do
			segmOutput[i]:copy( init[i] )
			segmOutput[i]:select( 2, 2 ):fill( -99 )
		end 
		local pl2 = segmOutput[1]:select( 2, 1 ):clone():apply( softPlus ):sum( 2 ) -- pixel-domain path length 
		local plR = pl1:clone():cdiv( pl2 ) 
		segmOutput[1]:select( 2, 1 ):apply( softPlus ):cmul( plR:expandAs( segmOutput[1]:select( 2, 1 ) ) ):apply( softPlusInv )

	else

		error('unknown pixel domain') 

	end

	local segment = Segment( segmOutput, 2 )
	segment:cuda()
	segment:updateOutput( z ) 
	z = segment:updateGradInput( z, segmOutput ) 
	likelihood:updateOutput( z )

	if recoverySet and recoverySet == 'Matched' then 

		local y = findNode( likelihood.network, 'membTrajectory' ).output
		local newFR = y:mean(2):squeeze()

		segmOutput[5]:add( currentFR ):add( -1, newFR )
		z = segment:updateGradInput( z, segmOutput ) 
		likelihood:updateOutput( z )

	end

	local likelihood  = findNode( likelihood.network, 'likelihood'  )
	local poissonRate = findNode( likelihood.network, 'poissonRate' ).output--[bestInd]
	local poissonRatePre = findNode( likelihood.network, 'prePoissonRate' ).output--[bestInd]

	-- print(poissonRatePre:min(), poissonRatePre:max())
	-- print(poissonRate:min(), poissonRate:max())

	poissonRate = nn.Unsqueeze(5):cuda():updateOutput(poissonRate):transpose(1,5):squeeze()
	local rate        = poissonRate[1]:clone()
	local rate        = torch.Tensor( poissonRate[1]:size() )

	data, dataMask, trialLength = loadData( dataset )

	-- local neuralFR = data:mean(1):mean(2):squeeze():clone():log():div( math.log(10) )

	-- print(poissonRate:min(), poissonRate:max())

	for i = 1, data:size(1) do 

		rate:copy( poissonRate[i] )
		randomkit.poisson( data[i], rate ) 

	end 

	-- print(data:min(), data:max())

	return data, dataMask, trialLength

end


function makeDataNeural()

	require 'cunn'

	ditherSamplesJoint = 2
	ditherInds, dim = ditherSamplesInds(ditherSamplesJoint)
	mb = mb * ditherSamplesJoint

	params.seed = 6
	makeExpParams( params.seed )

	local domain = 'neural'
	params.domain = domain
	dataset = datasets[1]
	local dataDir, resultsDir, analysisDir = makeDirsNew(params.domain, true)
	local master = loadSummary(datasets)

	local deterministic = false
	-- local deterministic = true
	local likelihood, z, bestInd = loadResults( 'neural', deterministic )

	local likeliModul = findNode(  likelihood.network, 'likelihood'  )
	local poissonRate = findNode( likeliModul.network, 'poissonRate' ).output

	poissonRate = nn.Unsqueeze(5):cuda():updateOutput(poissonRate):transpose(1,5):squeeze()
	local rate = torch.Tensor( poissonRate[1]:size() )

	local data, dataMask = loadData( dataset )
	local data_old, dataMask_old = data, dataMask

	local n = data:size(1)
	local N = 100
	    data = torch.Tensor(N, data:size(2), data:size(3), data:size(4))
	dataMask = torch.Tensor(N, data:size(2), data:size(3), data:size(4)):fill(1)

	for i = 1, N do
		likelihood:updateOutput(z)
		local ind = randomkit.randint(1, n)
		rate:copy( poissonRate[ind] )
		randomkit.poisson( data[i], rate )
	end

	return

	-- local likelihoodPerSample = findNode(  likeliModul.network, 'perSample'  ).output

	-- local n_repeats = 1
	-- local likelihoodPerFrame_repeats = torch.Tensor(n_repeats, N, likelihoodPerSample:size(1), likelihoodPerSample:size(3)) --, likelihoodPerSample:size(4)

	-- function compute_likelihood(d)

	-- 	for r = 1, n_repeats do 
	-- 		for i = 1, N do 
	-- 			print(r, i)

	-- 			local     data_i =        d:narrow(1, i, 1):expandAs(data_old)
	-- 			local dataMask_i = dataMask:narrow(1, i, 1):expandAs(dataMask_old)

	-- 			likelihood:loadData(data_i, dataMask_i)
	-- 			likelihood:updateOutput(z)

	-- 			local likelihood_i = likelihoodPerSample:mean(4):mul(-1):exp():mean(2):squeeze()
	-- 			likelihoodPerFrame_repeats[r][i]:copy(likelihood_i)

	-- 		end
	-- 	end
	-- 	local logLikelihoodPerFrame = likelihoodPerFrame_repeats:mean(1):squeeze()

	-- 	return logLikelihoodPerFrame

	-- end

	-- likelihoodPerFrame_orig = compute_likelihood(data)

	-- local flip = {1,6}

	-- local data_flip = data:clone()
	-- data_flip:select( 2, flip[1] ):copy( data:select( 2, flip[2] ) ) 
	-- data_flip:select( 2, flip[2] ):copy( data:select( 2, flip[1] ) ) 

	-- likelihoodPerFrame_flip = compute_likelihood(data_flip)

	-- correct = likelihoodPerFrame_orig:gt(likelihoodPerFrame_flip)
	-- prop_correct = correct:sum(1):squeeze()

	-- print(prop_correct)

end
