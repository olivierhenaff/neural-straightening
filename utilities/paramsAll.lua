loaddir  = 'data/'

--- inference parameters -------------------------------------

spikeNLmodel = 'SquaredSinhSoftPlus'
inferenceMeth   = 'VB'
maxiter         = 50000; collectLossFrom = 40000

uncertaintyInit = 'zero'
curvInitPost    = 'ML'
curvInitPrior   = 'ML'

priorAcc = 'FullWishartSig1' 

repeatNumber = 251 

nBootstraps = 1
-- nBootstraps = 5

-- multiscaleAnalysis = { p = {3}, mode = 'LP' } 
-- multiscaleAnalysis = { p = {2}, mode = 'LP' } 

-- multiscale = { p = { 3 }, mode = 'LP' } 
-- multiscale = { p = 2, mode = 'LP' } 
-- multiscale = { p = 1, mode = 'LP' } 

reportCurvatureFrom = 'preRate'
-- reportCurvatureFrom = 'fine'
-- reportCurvatureFrom = 'coarse'
-- reportCurvatureFrom = 'trialAverage'

-- recoveryBoost = { dist = 5, rate = 1 }
-- recoveryBoost = { dist = 5, rate = 10 }
-- recoveryBoost = { dist = 5, rate = 100 }
-- recoveryBoost = { dist = 1, rate = 5 }
-- recoveryBoost = { dist = 2, rate = 5 }
-- recoveryBoost = { dist = 3, rate = 5 }

-- recoverySet   = { dist = 4, rate = 5 }
-- recoverySet   = { dist = 20 } --, rate = 5 
-- recoverySet   = { dist = 40, rate = 5 }

-- recoverySet   = { dist = 20 } --, rate = 5 
recoverySet   = { rate = 'Matched' } --, rate = 5 
-- recoverySet   = { dist = 20, rate = 'Matched' } --, rate = 5