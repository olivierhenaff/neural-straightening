function pixelCurvatureDataset( pixelCurvature )

	local c = pixelCurvature:transpose( 1, 2 ):clone()
	c = c:view( c:size(1), c:size(2)*c:size(3), c:size(4), c:size(5) )

	local movieAlive = c:mean(3):mean(4):squeeze()
	movieAlive = movieAlive:eq( movieAlive ) 

	local pixelCurvature = torch.Tensor()
	for i = 1, movieAlive:size(1) do
		local c = filterDimension( c[i], movieAlive[i], 1 ) 
		pixelCurvature = torch.cat( pixelCurvature, c, 1 ) 
	end

	return pixelCurvature

end

function sequenceList()

	local list = { 'chironomus', 'bees', 'dogville', 'egomotion', 'prairieTer', 'carnegie-dam', 'walking', 'smile', 'water', 'leaves-wind' }

	return list 

end

function get_movieInds(dataset)

	local data = loadDataV1V2_dataset_raw( dataset )
	-- print(data)
	-- print(data.movie_id_list)
	-- print(data.movie_titles)

	print(data.movie_titles)

	data.movie_titles = {
		"NATURAL_01_CHIRONOMUS", 
		"NATURAL_04_EGOMOTION", 
		"NATURAL_05_PRAIRIE1", 
		"NATURAL_06_CARNEGIE_DAM", 
		"NATURAL_02_BEES", 
		"NATURAL_07_WALKING", 
		"NATURAL_10_LEAVES_WIND", 
		"NATURAL_11_ICE", 
		"SYNTHETIC_01_CHIRONOMUS", 
		"SYNTHETIC_04_EGOMOTION", 
		"SYNTHETIC_05_PRAIRIE1", 
		"SYNTHETIC_06_CARNEGIE_DAM", 
		"SYNTHETIC_02_BEES", 
		"SYNTHETIC_07_WALKING", 
		"SYNTHETIC_10_LEAVES_WIND", 
		"SYNTHETIC_11_ICE",
		} 


	local movieTitles = {}
	for u,v in pairs(data.movie_titles) do 
		v = v:gsub('NATURAL_', '' )
		v = v:gsub('SYNTHETIC_', '' )
		v = v:sub(1,2)
		table.insert(movieTitles, tonumber(v))
	end

	local movieInds = data.movie_id_list:squeeze()
	local movieIndsBis = movieInds:clone()

	-- print(data.movie_titles)
	-- -- print(movieTitles)
	-- print(movieInds)

	for i = 1, movieInds:size(1) do 

		-- print(i, movieInds[i], movieTitles[ movieInds[i] ] )

		movieIndsBis[i] = movieTitles[ movieInds[i] ] 

	end

	return movieIndsBis

end

function loadStimuli()

	local nPixels = 512 
	local nSequences, nFrames, loaddir, sequenceType, frameNames
	-- local dataset = dataset or datasets[1]

	-- if dataset:find('NaturalSequences_ay') then --acute
	if dataset:find('ay') then --acute

		nSequences =  44 
		nFrames    =  11 
		loaddir = 'images/acute/stimulus_fullfield/stimSelected-zoom'

		frameNames = {}
		frameNames['natural']   = {  'natural01',   'natural02',   'natural03',   'natural04',   'natural05',   'natural06',   'natural07',   'natural08',   'natural09',   'natural10', 'natural11', 'natural12' }
		frameNames['synthetic'] = {  'natural01', 'synthetic02', 'synthetic03', 'synthetic04', 'synthetic05', 'synthetic06', 'synthetic07', 'synthetic08', 'synthetic09', 'synthetic10', 'natural11' }
		frameNames['contrast']  = { 'contrast01',  'contrast02',  'contrast03',  'contrast04',  'contrast05',  'contrast06',  'contrast07',  'contrast08',  'contrast09',  'contrast10', 'natural01' }

		function loadpath(i,j)
			local sequenceType = i <= 20 and 'natural' or i <= 40 and 'synthetic' or i <= 44 and 'contrast' 
			local  zoomInd = ( math.ceil( i / 10 ) - 1 ) % 2 + 1
			local movieInd = (i-1) % 10 + 1
			if sequenceType == 'contrast' then
				local movieInds = { 2, 4, 7, 10 }
				movieInd = movieInds[ movieInd ]
			end
			local     path = loaddir .. zoomInd .. 'x/movie' .. movieInd .. '/' .. frameNames[sequenceType][j]
			return path
		end

	elseif dataset:find('pilot_naturalSequences_V') then --awake

		nSequences =   4 
		nFrames    =  12 
		loaddir = 'images/awake/stimSelected-zoom'
		function loadpath(i,j)
			local sequenceType = 'natural'
			local  zoomInd = 1 
			local movieInd = i
			local     path = loaddir .. zoomInd .. 'x/movie' .. movieInd .. '/' .. string.format('natural%02d', j )
			return path
		end

	elseif dataset:find('awake_sequences_V') then --awake2

		local seqList = { 'chironomus', 'bees', 'dogville', 'egomotion', 'prairie1', 'carnegie_dam', 'walking', 'smile', 'water', 'leaves-wind', 'ice3'}

		nSequences = get_mb(params.seed)
		nFrames    =  12 
		loaddir = 'images/awake2/stimSelected-zoom'
		movieInds = get_movieInds(dataset)
		function loadpath(i,j)
			local sequenceType = i <= (nSequences/2) and 'natural' or i <= nSequences and 'synthetic'
			local  zoomInd = 1
			-- local movieInd = (i-1) % 3 + 1
			-- local movieInds
			-- if     dataset == 'awake_sequences_V1_03132019' then 
			-- 	movieInds = { 1, 4, 6 }
			-- elseif dataset == 'awake_sequences_V1_03152019' then 
			-- 	movieInds = { 4, 5, 6 }
			-- elseif dataset == 'awake_sequences_V1_03202019' then 
			-- 	movieInds = { 2, 7, 10}
			-- elseif dataset == 'awake_sequences_V1_03262019' then 
			-- 	movieInds = { 1, 4, 11}
			-- elseif dataset == 'awake_sequences_V2_03182019' then 
			-- 	movieInds = { 2,10, 11}
			-- elseif dataset == 'awake_sequences_V2_03212019' then 
			-- 	movieInds = { 1, 5, 7 }
			-- elseif dataset == 'awake_sequences_V2_03282019' then 
			-- 	movieInds = { 4, 6, 2 }
			-- elseif dataset == 'awake_sequences_V1_05082019' then 
			-- 	movieInds = { 6, 7, 10, 11 }
			-- elseif dataset == 'awake_sequences_V1_05172019' then 
			-- 	movieInds = { 1, 4,  5,  2 }
			-- elseif dataset == 'awake_sequences_V2_05062019' then 
			-- 	movieInds = { 6, 7, 10, 11 }
			-- elseif dataset == 'awake_sequences_V2_05102019' then 
			-- 	movieInds = { 1, 4,  5,  2 }
			-- elseif dataset == 'awake_sequences_V2_05152019' then 
			-- 	movieInds = { 4, 5,  6,  7 }
			-- end
			-- movieInd = torch.Tensor(movieInds):narrow(1,movieInd,1):squeeze()
			movieInd = movieInds[i]

			-- print(movieInd, movieIndBis)

			local path = loaddir .. zoomInd .. string.format('x/movie%02d-', movieInd ) .. seqList[movieInd] .. '/' .. sequenceType .. string.format('%02d', j )
			return path
		end

	-- elseif dataset:find('Geodesics_V') then 
	elseif dataset:find('Geodesics') then 

		local seqList = { 'chironomus', 'bees', 'dogville', 'egomotion', 'prairie1', 'carnegie_dam', 'walking', 'smile', 'water', 'leaves-wind', 'ice3'}

		nPixels = 256 
		nSequences =  15
		nFrames    =   6
		loaddir = 'images/geodesics/movie'
		function loadpath(i,j)
			local sequenceType = (i-1) % 5 
			local movieInd = math.ceil(i/5)
			local movieInds = { 4, 6, 7 }
			local frameNumbers = torch.range(0, 10, 2)
			local frameNumber = frameNumbers[j]

			movieInd = torch.Tensor(movieInds):narrow(1,movieInd,1):squeeze() 

			local path = loaddir .. string.format('%02d-', movieInd ) .. seqList[movieInd] .. '/geodesic' .. sequenceType .. '_' .. frameNumber
			return path
		end


	end

	local x = torch.Tensor( nSequences, nFrames, nPixels, nPixels ) 

	for i = 1, nSequences do
		for j = 1, nFrames do 
			x[i][j]:copy( image.load( loadpath(i,j) .. '.png' ) ) 
		end
	end

	if ditherSamples then 

		x = ditherSamplesData( x, ditherSamples, 2 ) 

	elseif binSamples then 

		x = binSamplesData( x:view( 1, x:size(1), x:size(2), x:size(3), x:size(4) ), binSamples, 3 ):mean(1):squeeze()

	elseif ditherSamplesJoint then 

		x = ditherSamplesJointData( x, ditherSamplesJoint, 2 )

	end

	return x 

end

-- function loadPixelCurvature() 

-- 	local t = matio.load( loaddir .. 'pixelCurvatures.mat' ) 

-- 	local curvPixel = {}
-- 	for i = 1, 6 do 
-- 		local name = char2str( t.datafiles[i] )
-- 		name = name:gsub( '_0.*', '')
-- 		-- curvPixel[ 'NaturalSequences_' .. name ] = t.pixelCurvatureList[i] 
-- 		curvPixel[ 'NaturalSequences_' .. name ] = pixelCurvatureDataset( t.pixelCurvatureList[i] )
-- 	end 

-- 	return curvPixel 

-- end

function char2str_table( t )

	for u, v in pairs(t) do 

		-- print(u)
		-- print(type(v))

		if type(v) == 'userdata' and v:type() == 'torch.CharTensor' then 

			t[u] = char2str(v)

		elseif	type(v) == 'table' then

			char2str_table(v)

		end

	end

end

function loadPixelDistCurvatureSingle( x, params ) 

	local d, c, e = torch.Tensor(), torch.Tensor(), torch.Tensor() -- torch.Tensor( x:size(1) )

	for i = 1, x:size(1) do 

		local dist, curv 
		if params then 
			dist, curv = computeDistCurvatureMultiscale( x[i], params )
		else
			dist, curv = computeDistCurvature(           x[i] )
		end

		if i == 1 then 

			d:resize( x:size(1), dist:size(1) ):zero() 
			c:resize( x:size(1), curv:size(1) ):zero() 
			e:resize( x:size(1),           1  ):zero() 

		end

		d[i]:copy( dist )
		c[i]:copy( curv )
		e[i] = x[i][1]:dist( x[i][ x:size(2) ] ) / ( x:size(2)-1 )

	end

	c:mul( 180 / math.pi )

	return d, c, e 

end

function loadPixelDistCurvature( params ) 

	local x = loadStimuli() 

	local d, c, e 

	if ditherSamplesJoint then 

		d, c, e = {}, {}, {} 

		for i = 1, math.min( 3, #x ) do 

			local D, C, E = loadPixelDistCurvatureSingle( x[i], params )

			table.insert( d, D ) 
			table.insert( c, C ) 
			table.insert( e, E ) 

		end

		d = torch.cat( d, 2 )
		c = torch.cat( c, 2 )
		e = torch.cat( e, 2 )

	else 

		d, c, e = loadPixelDistCurvatureSingle( x, params )

	end

	return d, c, e 

end

function binaryToInd( x )

	local t = {} 

	for i = 1, x:size(1) do
		if x[i] == 1 then table.insert( t, i ) end 
	end

	return torch.LongTensor( t ) 

end

function filterDimension( x, dimensionMask, dim )

	-- print( dimensionMask )

	local ind = binaryToInd( dimensionMask )
	return x:index( dim, ind )
end

function ditherSamplesData( x, dither, dim ) 

	local ind, _ = ditherSamplesInds( dither, x:size(dim) )

	return x:index( dim, ind ) 

end

function ditherSamplesJointData( x, dither, dim ) 

	local t = {}

	for i = 1, dither do table.insert( t, ditherSamplesData( x, dither*10 + i, dim ) ) end

	return t

end

function binSamplesData( x, bin, dim ) 

	local y = torch.Tensor() 

	for i = 1, bin do 

		local ind = torch.range( i, math.floor(x:size(dim)/bin)*bin, bin ):long()

		y = torch.cat( y, x:index( dim, ind ), 1 ) 

	end

	return y 

end



function loadDataV1V2_dataset_raw( dataset )

	local data 

	local t7dir = loaddir .. dataset .. '.t7'
	local mtdir = loaddir .. dataset .. '.mat'

	if paths.filep( t7dir ) then 
		data = torch.load( t7dir ) 
	else
		data = matio.load( loaddir .. dataset .. '.mat' )
		torch.save( loaddir .. dataset .. '.t7', data ) 
	end

	if dataset:find('Geodesics') then 
		data = data.geoStruct
		for u, v in pairs( data.movie_titles ) do if type(v) == 'userdata' and v:type():find('Char') and v:size(1) == 1 then data.movie_titles[u] = char2str( v ) end end
	else	 
		data = data.naturalStruct 
	end

	data.unit_spike_times = nil 

	char2str_table(data)

	return data

end

function loadDataV1V2_dataset( dataset ) 

	-- local verbose = true 
	local verbose = false
	-- local threshold_numTrials_minAcrossFrames = 8 
	local threshold_numTrials_minAcrossFrames = 1

	local data = loadDataV1V2_dataset_raw(dataset)

	-- char2str_table(data)
	-- print( data )
	-- print( data.movie_id_list )
	-- error('stop here')

	-- corticalArea = char2str( data.cortical_area )

	for u, v in pairs( data ) do if type(v) == 'userdata' and v:type():find('Char') and v:size(1) == 1 then data[u] = char2str( v ) end end
	local allUnits = data.units 

	data = data.sortedResponses


	if not dataset:find('Geodesics') then 
		data     = data:transpose( 1, 2 ):clone() 
		data     = data:view( data:size(1), data:size(2)*data:size(3), data:size(4), data:size(5), data:size(6) )
	end

	-- if dataset:find('Geodesics_JP_') then 
	-- 	for i = 2, 5 do
	-- 		data[i]:select(2,1):copy(data[1]:select(2,1))
	-- 	end
	-- end

	dataMask = data:eq( data ):double()

	-- print( dataMask:sum(1):squeeze() )
	-- print( dataMask:sum(1):squeeze():size() )

	-- print( dataMask:sum(5):squeeze():select(4,1)) 
	-- error('stop here')

	if verbose then print('data size, raw'); print(data:size() ) end

	-- plotNumTrialsBis( dataMask ) 


	local nTrialsPerChannel = dataMask:sum(1):sum(2):sum(3):sum(5):squeeze()

	local channelAlive      = nTrialsPerChannel:gt( 100 )
	data     = filterDimension(     data, channelAlive, 4 ) 
	dataMask = filterDimension( dataMask, channelAlive, 4 ) 

	-- print( data:size() ) 
	-- print( dataMask:size() ) 

	if verbose then print( 'data size, filtering channels with less than 100 trials'); print(data:size()) end

	-- local nTrialsPerMovie   = dataMask:sum(3):sum(4):sum(5):squeeze()
	local nTrialsPerMovie   = dataMask:sum(3):sum(4):sum(5):select( 3, 1 ):select( 3, 1 ):select( 3, 1 )

	local movieAlive        = nTrialsPerMovie:gt( 0 )  

	local d, dm             = torch.Tensor(), torch.Tensor()
	for i = 1, movieAlive:size(1) do
		local data     = filterDimension(     data[i], movieAlive[i], 1 ) 
		local dataMask = filterDimension( dataMask[i], movieAlive[i], 1 ) 
		d  = torch.cat( d , data    , 1 ) 
		dm = torch.cat( dm, dataMask, 1 ) 
	end
	data, dataMask = d, dm 

	    data =     data:transpose( 1, 4 ) 
	dataMask = dataMask:transpose( 1, 4 ) 


	data:maskedFill( data:ne( data ), 0 ) 

	if verbose then print( 'data size, filtering movies with 0 trials'); print(data:size()) end
	-- print( dataMask:size() ) 

	-- print( dataMask:sum(1):squeeze():size() ) 
	-- local danger = dataMask:sum(1):select( 3, 1 )--:min(2)
	-- print( danger ) 


	local minTrialsPerChannel = dataMask:sum(1):min(2):min(4):squeeze()

	-- plotNumTrials( dataMask )

	-- print( minTrialsPerChannel )

	-- local channelSafe         = minTrialsPerChannel:gt(10)
	local channelSafe         = minTrialsPerChannel:ge( threshold_numTrials_minAcrossFrames )


	-- print( data:sum(1):squeeze() )
	-- -- print( dataMask:sum(1):squeeze() )
	-- print( minTrialsPerChannel ) 
	-- print( channelSafe )
	-- print( data:size() ) 
	-- error('stop here')


	data     = filterDimension(     data, channelSafe, 3 ) 
	dataMask = filterDimension( dataMask, channelSafe, 3 ) 

	if verbose then print( 'data size, filtering channels with a frame wih less than ' .. threshold_numTrials_minAcrossFrames .. ' trials'); print(data:size()) end

	local nSpikesPerChannel   =     data:sum(1):sum(2):min(4):squeeze()

	-- print( nSpikesPerChannel)

	-- min_number_of_spikes_per_channel = 10
	min_number_of_spikes_per_channel = 50

	-- local channelSafe         = nSpikesPerChannel:gt(10)
	local channelSafe         = nSpikesPerChannel:gt(min_number_of_spikes_per_channel)
	data     = filterDimension(     data, channelSafe, 3 ) 
	dataMask = filterDimension( dataMask, channelSafe, 3 ) 

	if verbose then print( 'data size, filtering channels with less than ' .. min_number_of_spikes_per_channel .. ' spikes'); print(data:size()) end

	-- print( filterDimension(nSpikesPerChannel, channelSafe, 1 ) )


	local nsmpl = data:size(2)

	if  ditherSamples then 

		data     = ditherSamplesData( data    , ditherSamples, 2 )
		dataMask = ditherSamplesData( dataMask, ditherSamples, 2 )

	elseif binSamples then 

		data     = binSamplesData( data    , binSamples, 2 )
		dataMask = binSamplesData( dataMask, binSamples, 2 )

	elseif ditherSamplesJoint then 

		local d  = data:narrow( 2, data:size(2), 1 ):clone()
		local m  = data:narrow( 2, data:size(2), 1 ):clone():zero()

		data     = ditherSamplesJointData( data    , ditherSamplesJoint, 2 ) 
		dataMask = ditherSamplesJointData( dataMask, ditherSamplesJoint, 2 ) 

		if nsmpl == 11 then 

			data[    #data] = torch.cat( data[    #data], d, 2 ) 
			dataMask[#data] = torch.cat( dataMask[#data], m, 2 ) 

		end

		data     = torch.cat( data    , 4 ) 
		dataMask = torch.cat( dataMask, 4 ) 

	end

	if verbose then print( 'data size, final' ); print(data:size()) end
	-- print( dataMask:size() ) 
	-- error('stop here')

	-- print('data:size()')
	-- print( data:size() ) 
	-- local plot = {} 
	-- for i = 1, data:size(4) do 
	-- 	local d = data:select( 4, i ):clone():view( -1, data:size(3) )
	-- 	local u, s, v = torch.svd( d ) 
	-- 	local c = torch.cumsum( s:pow(2) )
	-- 	c:div( c:max() ) 
	-- 	table.insert( plot, { c, '-' } )
	-- end
	-- gnuplot.savePlot( 'cumulative.png', plot ) 
	-- error('dataset')

	-- error('stop here')

	-- plotNumTrials( dataMask )

	-- print(data:sum() ) 
	-- print(dataMask:sum() ) 

	-- for i = 1, 15 do 

	-- 	di = data:select( 4, i )
	-- 	print( di:select(2,1):dist(di:select(2,2)))

	-- end

	-- print( data:size() ) 
	-- error('stop here')


	return data, dataMask 

end

--[[
function loadDataV1V2( dataset ) 

	local t7dir = loaddir .. dataset .. '.t7'
	local mtdir = loaddir .. dataset .. '.mat'
	if paths.filep( t7dir ) then 
		data = torch.load( t7dir ) 
	else
		data = matio.load( loaddir .. dataset .. '.mat' )
		torch.save( loaddir .. dataset .. '.t7', data ) 
	end

	if dataset:find('NaturalSequences_') then 

		data = data.naturalStruct 

		corticalArea = char2str( data.cortical_area )

		for u, v in pairs( data ) do if type(v) == 'userdata' and v:type():find('Char') and v:size(1) == 1 then data[u] = char2str( v ) end end
		local allUnits = data.units 

		data     = data.sortedResponses[ scaleInd ][ typeInd ][ videoInd ]:transpose( 1, 3 ):transpose( 2, 3 )--:clone()
		dataMask = data:eq( data ):double()

		if filterData then 

			local tuned_units = matio.load( loaddir .. 'tuned_units.mat' )
			for i = 1, 6 do tuned_units.natSequences[i] = char2str( tuned_units.natSequences[i] ) end 
			local datasetInd
			for i = 1, 6 do if (dataset..'.mat') == tuned_units.natSequences[i] then datasetInd = i end end 

			if filterData == 'orientation' then 

				local orientationTunedUnits = tuned_units.ori_selective_units[datasetInd]
				local isOrientationTuned    = torch.ByteTensor( 1, allUnits:size(2) ):fill(0)
				for i = 1, allUnits:size(2) do 
					for j = 1, orientationTunedUnits:size(2) do 
						if orientationTunedUnits[1][j] == allUnits[1][i] then isOrientationTuned[1][i] = 1 end 
					end
				end

				data, dataMask = filterChannels( data, dataMask, isOrientationTuned:squeeze() )

			elseif filterData == 'movie' then 

				local movieTunedUnits = tuned_units.movie_selective_units[datasetInd]
				local lookup = torch.Tensor( 2, 3, 10 ); s = lookup:storage(); for i = 1, lookup:nElement() do s[i] = i end 
				movieTunedUnits = movieTunedUnits[ lookup[scaleInd][typeInd][videoInd] ]
				local nMovieTunedUnits =  movieTunedUnits:nElement()

				if nMovieTunedUnits < 2 then 

					error('Sorry. Less than 2 movie-tuned units, can\'t resolve curvature')

				else

					local isMovieTuned    = torch.ByteTensor( 1, allUnits:size(2) ):fill(0)
					for i = 1, allUnits:size(2) do 
						for j = 1, movieTunedUnits:size(2) do 
							if movieTunedUnits[1][j] == allUnits[1][i] then isMovieTuned[1][i] = 1 end 
						end
					end

					data, dataMask = filterChannels( data, dataMask, isMovieTuned:squeeze() )

				end

			end

		end

		local    nTrialsAlive     = dataMask:sum(1):squeeze()
		-- local    nTrialsAliveMean = nTrialsAlive:mean(1)
		local nTrialsAliveMin, _  = nTrialsAlive:min(1)

		-- local channelAliveMean = nTrialsAliveMean:gt(0):squeeze()
		local channelAliveMin  =  nTrialsAliveMin:gt(0):squeeze()
		local channelAlive     = channelAliveMin

		data, dataMask = filterChannels( data, dataMask, channelAlive ) 

		data:maskedFill( data:ne( data ), 0 ) 

	else

		    data = data.summary[ videoType .. '_spikeCounts' ][ scaleInd ][ videoInd ]:transpose( 1, 3 )
		dataMask = data:clone():fill( 1 ) 

	end

	trialLength = data:clone():fill( trialDuration )

	return data, dataMask, trialLength 

end	
]]
