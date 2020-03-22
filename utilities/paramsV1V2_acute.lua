require 'utilities/paramsV1V2'

function get_datasets(batch)

	local datasets = {
		'ay2_u003_image_sequences',  -- V1 1   --   1 -  20 
		'ay4_u001_image_sequences',  -- V1 2   --  21 -  40 
		'ay4_u006_image_sequences',  -- V1 3   --  41 -  60
		'ay4_u005_image_sequences',  -- V1 4   --  61 -  80
		'ay3_u001_image_sequences',  -- V1 5   --  81 - 100 
		'ay3_u002_image_sequences',  -- V1 6   -- 101 - 120 
		'ay2_u007_image_sequences',  -- V1 7 
		'ay5_u002_image_sequences',  -- V1 8
		'ay5_u005_image_sequences',  -- V1 9
	}

	local datasets1, datasets2, datasets3, datasets12 = {}, {}, {}, {} 
	for i =  1,  5 do table.insert(datasets1 , datasets[i]) end
	for i =  6,  9 do table.insert(datasets2 , datasets[i]) end
	for i = 10, 14 do table.insert(datasets3 , datasets[i]) end
	for i =  1,  9 do table.insert(datasets12, datasets[i]) end

	if not batch or batch == 'all' then 
		return datasets
	elseif batch == 1 then 
		return datasets1
	elseif batch == 2 then 	
		return datasets2
	elseif batch == 3 then 	
		return datasets3
	elseif batch == 12 then 	
		return datasets12
	else
		error('unknown batch index')
	end

end

datasets = get_datasets(datasetBatch)

areas = {'V1'}
if #datasets > 14 then table.insert( areas, 'V2' ) end 

-- videoTypes = { 'natural', 'synthetic', 'contrast' } 
-- videoTypes = {'natural'}
-- videoTypes = {'natural', 'synthetic'} 
videoTypes = {'synthetic'} 

-- learningRate = 0.005

sequenceInd = { natural   = { beg =  1, num = 20 },
				synthetic = { beg = 21, num = 20 },
				contrast  = { beg = 41, num =  4 }}

function get_mb()
	return 44 
end

mb      = get_mb() --44
nScales =  2 
nVideos = 10

nsmpl = 11 
dim = nsmpl - 1 
-- dim = nsmpl 

function ditherSamplesInds( dither, n ) 

	local nsmpl = n or 11 
	local inds
	
	local dither = dither or 1 

	if dither > 10 then 
		inds = torch.range( dither % 10, nsmpl, math.floor( dither / 10 ) ):long() 
	else
		inds = torch.range(           1, nsmpl,             dither        ):long()
	end

	-- local dim  = inds:size(1)
	local dim  = inds:size(1)-1


	return inds, dim

end

function getCorticalArea( dataset )

	if dataset == 'NaturalSequences_ay2_u003' 
	or dataset == 'NaturalSequences_ay4_u001' 
	or dataset == 'NaturalSequences_ay4_u006' 
	or dataset == 'NaturalSequences_ay4_u005'
	or dataset == 'NaturalSequences_ay3_u001' 

	or dataset == 'ay2_u003_image_sequences'   
	or dataset == 'ay2_u007_image_sequences' 
	or dataset == 'ay3_u001_image_sequences' 
	or dataset == 'ay3_u002_image_sequences'
	or dataset == 'ay4_u001_image_sequences'
	or dataset == 'ay4_u005_image_sequences'
	or dataset == 'ay4_u006_image_sequences'
	or dataset == 'ay5_u002_image_sequences'
	or dataset == 'ay5_u005_image_sequences' then 

		return 'V1'

	elseif dataset == 'ay2_u004_image_sequences'
	or dataset == 'ay2_u005_image_sequences'
	or dataset == 'ay2_u006_image_sequences'
	or dataset == 'ay4_u000_image_sequences'
	or dataset == 'ay4_u004_image_sequences'
	or dataset == 'ay4_u007_image_sequences'
	or dataset == 'ay5_u004_image_sequences' then 

		return 'V2'

	else

		error('Unknown dataset')

	end

end