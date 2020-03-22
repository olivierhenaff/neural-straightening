require 'utilities/utilities'

analysisType = 'main'

-- require 'analyzeData/paramsV1V2_awake_main'
require 'utilities/paramsV1V2_acute'
-- require 'analyzeData/paramsV1V2_geodesics' -- next time we analyse new geodesic data, re-format data tensor such that sequences of a given type are contiguous (i.e. follow one another) within the sequence dimension.


matio = require 'matio'


sys.tic()

cmd = torch.CmdLine()

-- cmd:option('-seed',  -1, 'job id')
-- cmd:option('-seed',  0, 'job id')
cmd:option('-seed',  1, 'job id')

cmd:option('-domain', 'neural', 'neural or pixel')
-- cmd:option( '-domain', 'pixel-full' , 'neural or pixel' )

cmd:option('-dither',  1, '1, 2, 3, or 5')
-- cmd:option( '-dither', 21, '1, 2, 3, or 5' ) -- 21 and 22 for even and odd, subsampling by a factor of 2
-- cmd:option( '-dither', 22, '1, 2, 3, or 5' ) -- 21 and 22 for even and odd, subsampling by a factor of 2
-- cmd:option( '-dither', 31, '1, 2, 3, or 5' ) -- 31 and 32 for even and odd, subsampling by a factor of 3 (can't do third sampling)
-- cmd:option( '-dither', 32, '1, 2, 3, or 5' ) -- 31 and 32 for even and odd, subsampling by a factor of 3 (can't do third sampling)
-- cmd:option('-ditherJoint', 1, '1, 2, 3, or 5')
cmd:option( '-ditherJoint', 2, '1, 2, 3, or 5' )
-- cmd:option('-ditherJoint', 3, '1, 2, 3, or 5')

cmd:option('-repeatNumber', 1, 'repeat number')


cmd:option('-datasetBatch', 0, 'dataset batch. 0: all. >0: that batch index')

params = cmd:parse(arg)

if params.repeatNumber > 0 then repeatNumber = params.repeatNumber end
if params.datasetBatch > 0 then datasetBatch = params.datasetBatch end

print('datasetBatch', datasetBatch)

if     params.dither > 1 then

	ditherSamples = params.dither
	ditherInds, dim = ditherSamplesInds(ditherSamples)

elseif params.ditherJoint > 1 then

	ditherSamplesJoint = params.ditherJoint
	ditherInds, dim = ditherSamplesInds(ditherSamplesJoint)
	mb = mb * ditherSamplesJoint

end

print( 'mb', mb, 'dim', dim, 'ditherSamples', ditherSamples, 'binSamples', binSamples, 'ditherSamplesJoint', ditherSamplesJoint )

if params.seed ~= 0 then require 'cunn' end

if params.seed >  0 then

	makeExpParams( params.seed )

	if     params.domain == 'neural' then

		data, dataMask, trialLength = loadData(dataset)

	elseif params.domain:find( 'pixel' ) then

		data, dataMask, trialLength = makeDataPixel( params.domain )

	end

	--- data parameters ------------------------------------------

	nTrials = data:size(1)
	nsmpl   = data:size(2)

	-- dataDir, resultsDir, analysisDir = makeDirsMain()
	dataDir, resultsDir, analysisDir = makeDirsNew( params.domain, true )

	local expname    = experimentName()
	local    dataDir =    dataDir .. expname
	local resultsDir = resultsDir .. expname

	print('going to save to', resultsDir )

	local results = inferCurvature( data, dataMask, trialLength )

	torch.save( resultsDir, results )

elseif params.seed < 0 then 

	collectAllResults( - params.seed )

end

print( 'done after', sys.toc(), 'seconds' )

