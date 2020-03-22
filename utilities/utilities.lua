require 'common-torch/utilities'
require 'utilities/utilitiesData'
require 'utilities/utilitiesInference'

function capFirst( str ) 

	return str:sub(1,1):upper()..str:sub(2,-1)

end

function baseDir() 

	return '/scratch/ojh221/results/neural-straightening/project' .. project .. '/' 

end 

function dataName( main, dataset ) 

	local datadir 

	if main then 

		datadir = dataset 

		if filterData then 

			datadir = datadir .. '_filtered' .. capFirst( filterData )

		end

	else

		datadir = spikeCountDistributionData .. '_nl' .. spikeNLdata

		if dataset == 'explicit' then 

			datadir = datadir 	
						.. '_trialDuration' .. trialDuration 
						.. '_nSamples' .. nsmpl
						.. '_nNeurons' .. dim
						.. '_nTrials'  .. nTrials
						.. '_distMin'  .. dMin .. 'Max' .. dMax 
						.. '_z0mean'   .. z0_mean .. '_std' .. z0_std 

		else

			datadir = dataset .. '_' .. datadir .. '_acc' .. dataAcc 

		end

	end

	return datadir 

end 

function makeDirsNew( domain, main )

	local domainStr = domain 
	if domain == 'pixel-full' and recoveryBoost then 
		domainStr = domainStr .. '-boost'
		if recoveryBoost.dist then domainStr = domainStr .. 'Dist' .. recoveryBoost.dist end 
		if recoveryBoost.rate then domainStr = domainStr .. 'Rate' .. recoveryBoost.rate end  
	end
	if domain == 'pixel-full' and recoverySet   then 
		domainStr = domainStr ..   '-set'
		if recoverySet.dist   then domainStr = domainStr .. 'Dist' .. recoverySet.dist end 
		if recoverySet.rate   then domainStr = domainStr .. 'Rate' .. recoverySet.rate end  
	end
	if domain == 'pixel-full' and recoveryClamp then 
		domainStr = domainStr ..   '-clamp'
		if recoverySet.min    then domainStr = domainStr .. 'Min' .. recoverySet.min end 
		if recoverySet.max    then domainStr = domainStr .. 'Max' .. recoverySet.max end 
	end

	local savedir  = baseDir(main)

	savedir = savedir .. 'dataset'
	if     ditherSamples      then
		savedir = savedir .. 'Dithered' .. ditherSamples
	elseif binSamples         then 
		savedir = savedir .. 'Binned'   ..    binSamples
	elseif ditherSamplesJoint then 
		savedir = savedir .. 'DitheredJoint' .. ditherSamplesJoint 
	else
		savedir = savedir .. 'Full'
	end
	savedir = savedir .. '/' 

	local analysisName = spikeCountDistributionModel
	if spikeCountDistributionModel:find('OverdispersedShared') then analysisName = analysisName .. nModulatorsModel end
    analysisName = analysisName 
					.. '_spikeNl' .. spikeNLmodel
					-- .. '_'        .. inferenceMeth 
					.. '_trialDuration' .. trialDuration 
					.. '_maxiter' .. maxiter 
					.. '_collect' .. collectLossFrom
					.. '_mb'  .. mb .. '_unc' .. capFirst( uncertaintyInit ) 
					.. '_curvPost' .. capFirst( curvInitPost ) .. 'Prior' .. capFirst( curvInitPrior )   
					.. '_accPrior' .. priorAcc 
					.. '_dim' .. dim 

	if priorAccDataset then analysisName = analysisName .. '_lambda' .. priorAccDataset end 

	if multiscaleSynthesis then analysisName = analysisName .. '_multiscale' .. multiscale.p[1] end

	if repeatNumber    then analysisName = analysisName .. '_repeat' .. repeatNumber    end 

	savedir = savedir .. analysisName .. '/' 

	local analysisStr = 'analysis' 
	if multiscaleAnalysis  then analysisStr = analysisStr .. '-' .. multiscaleAnalysis.mode .. '-p' .. multiscaleAnalysis.p[1] end --.. ( multiscale.p[2] and '-' .. multiscale.p[2] or '' )
	if reportCurvatureFrom then analysisStr = analysisStr .. '-' .. reportCurvatureFrom end 

	-- local dataDir  = savedir .. dataName( main, dataset ) .. '/' 
	local resultsDir  = savedir ..         'results/' .. domainStr .. '/' .. dataName( main, dataset ) .. '/'
	-- local analysisDir = savedir .. 'analysis/' .. domainStr .. '/' .. dataName( main, dataset ) .. '/'
	local analysisDir = savedir .. analysisStr .. '/' .. domainStr .. '/' .. dataName( main, dataset ) .. '/'
	local dataDir     = savedir ..            'data/' .. domainStr .. '/' .. dataName( main, dataset ) .. '/'

	if not main then 
		os.execute( 'mkdir -p ' .. dataDir     )
	end
	-- os.execute( 'mkdir -p ' .. analysisDir )
	os.execute( 'mkdir -p ' .. resultsDir  )

	return dataDir, resultsDir, analysisDir 

end

