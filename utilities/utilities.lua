require 'common-torch/utilities'
require 'utilities/utilitiesData'
require 'utilities/utilitiesInference'

function capFirst( str ) 

	return str:sub(1,1):upper()..str:sub(2,-1)

end

function baseDir() 

	return '/scratch/ojh221/results/neural-straightening/project' .. project .. '/' 

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

	local savedir = baseDir() .. 'dataset_dither' .. ditherSamplesJoint .. '/' 

	local analysisName = spikeCountDistributionModel
	if spikeCountDistributionModel:find('OverdispersedShared') then analysisName = analysisName .. nModulatorsModel end
    analysisName = analysisName 
					.. '_spikeNl' .. spikeNLmodel
					.. '_maxiter' .. maxiter 
					.. '_collect' .. collectLossFrom
					.. '_mb'  .. mb
					.. '_accPrior' .. priorAcc 
					.. '_dim' .. dim 

	if priorAccDataset then analysisName = analysisName .. '_lambda' .. priorAccDataset end 

	if repeatNumber    then analysisName = analysisName .. '_repeat' .. repeatNumber    end 

	savedir = savedir .. analysisName .. '/' 

	local analysisStr = 'analysis' 
	if reportCurvatureFrom then analysisStr = analysisStr .. '-' .. reportCurvatureFrom end 

	local resultsDir  = savedir ..         'results/' .. domainStr .. '/' .. dataset .. '/'
	local analysisDir = savedir .. analysisStr .. '/' .. domainStr .. '/' .. dataset .. '/'
	local dataDir     = savedir ..            'data/' .. domainStr .. '/' .. dataset .. '/'

	os.execute( 'mkdir -p ' .. resultsDir  )

	return dataDir, resultsDir, analysisDir 

end

