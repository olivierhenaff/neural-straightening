require 'utilities/paramsAll'

project = 'V1V2'

spikeCountDistributionModel = 'OverdispersedShared'; sigNinit = 0.1; nModulatorsModel = 2
-- spikeCountDistributionModel = 'OverdispersedShared'; sigNinit = 0.1; nModulatorsModel = 3

function getCorticalArea( dataset )

	local area = dataset:match('_V%d_'):gsub('_','')

	return area 

end

function makeGlobalParams() 

	nTypes  = #videoTypes 
	nJobs   = #datasets * nScales * nVideos * nTypes  

end

if analysisType == 'main' then 

	function makeExpParamsOld( seed ) 

		local jobInd = seed 
		typeInd   = math.ceil( jobInd/(#datasets*nScales*nVideos) )
		jobInd    = (jobInd-1) % ( #datasets*nScales*nVideos ) + 1 
		local datasetInd = math.ceil( jobInd/(nScales*nVideos) )
		jobInd    = (jobInd-1) % ( nScales*nVideos ) + 1 
		scaleInd  = math.ceil( jobInd / nVideos ) 
		videoInd  = (jobInd-1) % (         nVideos ) + 1 
		videoType = videoTypes[ typeInd ]

		dataset = datasets[ datasetInd ]

	end

	function experimentNameOld() return videoType .. '_video' .. videoInd .. '_scale' .. scaleInd .. '.t7' end

	function makeExpParams( seed )

		local jobInd =  seed
		  datasetInd = (jobInd-1) % ( #datasets ) + 1
		bootstrapInd = math.ceil( jobInd / #datasets )

		dataset = datasets[ datasetInd ]

	end

	-- function experimentName() return dataset .. '.t7' end
	function experimentName() return 'bootstrap' .. bootstrapInd .. '.t7' end

elseif analysisType == 'recovery' then 

	function makeExpParams( seed ) 

		local datasetInd = math.ceil( seed / nCurv )
		local curvInd = ( seed - 1 ) % nCurv + 1

		scaleInd = torch.randperm( nScales )[1]
		videoInd = torch.randperm( nVideos )[1]
		 typeInd = torch.randperm( nTypes  )[1] 

		dataset = datasets[ datasetInd ]
		cTrue   = curvTrue[    curvInd ] 

	end

	function experimentName( c ) return string.format( 'curv%.2f.t7', c ) end

end
