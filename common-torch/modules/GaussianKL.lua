require 'nngraph'

function GaussianKLstandard( ncond, muSigDim )

	local inode = nn.Identity()()

	local mean   = nn.Select( muSigDim, 1 )( inode )
	local logStd = nn.Select( muSigDim, 2 )( inode )

	local mean2 = nn.Square()( mean ) 
	local std2  = nn.Square()( nn.Exp()( logStd ) )

	local kl = nn.CAddTable()({nn.MulConstant(-2)(logStd), mean2, std2})

	kl = nn.AddConstant(-1)( kl ) 
	kl = nn.View(ncond, -1)( kl ) 
	kl = nn.Sum(2)( kl )
	kl = nn.MulConstant( 0.5 )( kl ) 

	local network = nn.gModule({ inode }, { kl })

	return network

end

function GaussianKLdiagonalLearnedMuStd( ncond, muSigDim ) 

	local muSigDim = muSigDim or 1 
	
	local inode = nn.Identity()()
	local terms = {} 

	local z       = nn.SelectTable(1)( inode )
	local logSigZ = nn.Select(muSigDim,2)( z )

	local t       = nn.SelectTable(2)( inode )
	local logSigT = nn.Select(muSigDim,2)( t )

	local twoLogSigT = nn.MulConstant( 2 )( logSigT )
	local sigThetaInv2 = nn.Exp()( nn.MulConstant(-1)(twoLogSigT) ) 

	local sigZ2 = nn.Exp()( nn.MulConstant(2)( logSigZ ) ) 
	local stdRatio = nn.CMulTable()({sigZ2, sigThetaInv2})

	local minusTwoLogSigZ = nn.MulConstant(-2)( logSigZ )

	local terms = {twoLogSigT, minusTwoLogSigZ, stdRatio}

	local muZ  = nn.Select(muSigDim,1)( z )
	local muT  = nn.Select(muSigDim,1)( t ) --ncond, nsmpl
	local muZ2 = nn.Square()( nn.CSubTable()({muZ, muT}) ) 
	local differences = nn.CMulTable()({muZ2, sigThetaInv2})
	table.insert( terms, differences ) 

	local kl = nn.CAddTable()( terms ) 
	kl = nn.AddConstant(-1)( kl ) 
	kl = nn.View(ncond, -1)( kl ) 
	kl = nn.Sum(2)( kl )
	kl = nn.MulConstant( 0.5 )( kl ) 

	local network = nn.gModule({ inode }, { kl })

	return network

end

function GaussianKLdiagonalLearnedMuStdPriorFull( mb, muSigDim, nsmpl, dim ) 

	local muSigDim = muSigDim or 1 
	
	local inode = nn.Identity()()

	local z       = nn.SelectTable(1)( inode ) -- [mb, 2, nsmpl, dim] 
 	local muZ     = nn.Select(muSigDim,1)( z ) -- [mb,    nsmpl, dim] 
	local logSigZ = nn.Select(muSigDim,2)( z )

	local t       = nn.SelectTable(2)( inode ) -- [mb, 2, nsmpl, dim] 
	local muT     = nn.Select(muSigDim,1)( t ) -- [mb,    nsmpl, dim] 
	local logSigT = nn.Select(muSigDim,2)( t )

	local      twoLogSigT = nn.MulConstant( 2 )( logSigT )
	local minusTwoLogSigZ = nn.MulConstant(-2 )( logSigZ )
	local terms = {twoLogSigT, minusTwoLogSigZ}

	local sigTi = nn.Exp()( nn.MulConstant(-1)( logSigT ) ) 
	local sigZ  = nn.Exp()(                     logSigZ ) 

	local lInversePrior = nn.SelectTable(3)( inode ) 
	lInversePrior = lowerTriangle( dim )( lInversePrior ) 
	lInversePrior = nn.Replicate( nsmpl, 2 )( lInversePrior ) 
	lInversePrior = nn.Reshape( mb*nsmpl, dim, dim )( lInversePrior ) 

	local deltaMu    = nn.View( mb * nsmpl, dim, 1 )( nn.CSubTable()({muZ, muT}) ) -- [mb * nsmpl, dim, 1] 
	local deltaMuRot = nn.MM()({ lInversePrior, deltaMu })                         -- [mb * nsmpl, dim, 1]
	local deltaSc    = nn.CMulTable()({deltaMuRot, sigTi})
	local mse        = nn.Square()( deltaSc ) 
	table.insert( terms, mse ) 

	sigZ  = nn.Replicate( dim, 3 )( sigZ  ) 
	sigTi = nn.Replicate( dim, 4 )( sigTi )
	local trace = nn.Sum(3)( nn.Square()( nn.CMulTable()({sigZ, sigTi, lInversePrior}) ) ) 
	table.insert( terms, trace ) 

	local kl = nn.CAddTable()( terms ) 
	kl = nn.AddConstant(-1)( kl ) 
	kl = nn.View(mb, -1)( kl ) 
	kl = nn.Sum(2)( kl )
	kl = nn.MulConstant( 0.5 )( kl ) 

	local network = nn.gModule({ inode }, { kl })

	return network

end

function GaussianKLdiagonalLearnedMuStdPriorFullWishart( mb, muSigDim, nsmpl, dim, p ) 

	local muSigDim = muSigDim or 1 
	
	local inode = nn.Identity()()

	local z       = nn.SelectTable(1)( inode ) -- [mb, 2, nsmpl, dim] 
 	local muZ     = nn.Select(muSigDim,1)( z ) -- [mb,    nsmpl, dim] 
	local logSigZ = nn.Select(muSigDim,2)( z )

	local t       = nn.SelectTable(2)( inode ) -- [mb, 2, nsmpl, dim] 
	local muT     = nn.Select(muSigDim,1)( t ) -- [mb,    nsmpl, dim] 
	local logSigT = nn.Select(muSigDim,2)( t )

	local      twoLogSigT = nn.MulConstant( 2 )( logSigT )
			   twoLogSigT = nn.MulConstant( 1 / p )( twoLogSigT ) 
	local minusTwoLogSigZ = nn.MulConstant(-2 )( logSigZ )
	local terms = {twoLogSigT, minusTwoLogSigZ}

	local sigTi = nn.Exp()( nn.MulConstant(-1)( logSigT ) ) 
	local sigZ  = nn.Exp()(                     logSigZ ) 

	local lInversePrior = nn.SelectTable(3)( inode ) 
	lInversePrior = lowerTriangle( dim )( lInversePrior ) 
	lInversePrior = nn.Replicate( nsmpl, 2 )( lInversePrior ) 
	lInversePrior = nn.Reshape( mb*nsmpl, dim, dim )( lInversePrior ) 

	local deltaMu    = nn.View( mb * nsmpl, dim, 1 )( nn.CSubTable()({muZ, muT}) ) -- [mb * nsmpl, dim, 1] 
	local deltaMuRot = nn.MM()({ lInversePrior, deltaMu })                         -- [mb * nsmpl, dim, 1]
	      deltaMuRot = nn.View( mb , nsmpl, dim, 1 )( deltaMuRot )                 -- [mb , nsmpl, dim, 1] 
	local deltaSc    = nn.CMulTable()({deltaMuRot, sigTi})
	local mse        = nn.Square()( deltaSc ) 
	table.insert( terms, mse ) 

	sigZ  = nn.Replicate( dim, 3 )( sigZ  ) 
	sigZ  = nn.Sqrt()( nn.AddConstant( (1-p)/p )( nn.Square()( sigZ ) ) )
	sigTi = nn.Replicate( dim, 4 )( sigTi )
	local trace = nn.Sum(3)( nn.Square()( nn.CMulTable()({sigZ, sigTi, lInversePrior}) ) ) 
	table.insert( terms, trace )

	local kl = nn.CAddTable()( terms ) 
	kl = nn.AddConstant(-1)( kl ) 
	kl = nn.View(mb, -1)( kl ) 
	kl = nn.Sum(2)( kl )
	kl = nn.MulConstant( 0.5 )( kl ) 

	local network = nn.gModule({ inode }, { kl })

	return network

end

function IdentityWishartNLL( lambda ) 

	local logSig = nn.Identity()()

	local twoLogSig = nn.MulConstant( 2 )( logSig )
	local sig2i     = nn.Exp()( nn.MulConstant( -2 )( logSig ) )

	local loss      = nn.CAddTable()({ twoLogSig, sig2i })
	loss = nn.View( mb, -1 )( loss ) 
	loss = nn.Sum(2)( loss ) 
	loss = nn.MulConstant( lambda )( loss )

	local network = nn.gModule( { logSig }, { loss } ) 

	return network 

end

function GaussianKLdiagonalLearnedMuStdPriorFullWishartLearned( mb, muSigDim, nsmpl, dim, nu ) 

	local muSigDim = muSigDim or 1 
	
	local inode = nn.Identity()()

	local z       = nn.SelectTable(1)( inode ) -- [mb, 2, nsmpl, dim] 
 	local muZ     = nn.Select(muSigDim,1)( z ) -- [mb,    nsmpl, dim] 
	local logSigZ = nn.Select(muSigDim,2)( z )

	local t       = nn.SelectTable(2)( inode ) -- [mb, 2, nsmpl, dim] 
	local muT     = nn.Select(muSigDim,1)( t ) -- [mb,    nsmpl, dim] 
	local logSigT = nn.Select(muSigDim,2)( t )

	local      twoLogSigT = nn.MulConstant( 2 )( logSigT )
			   twoLogSigT = nn.MulConstant( ( nsmpl + nu + dim + 1 ) / nsmpl )( twoLogSigT ) 
	local minusTwoLogSigZ = nn.MulConstant(-2 )( logSigZ )
	local terms = {twoLogSigT, minusTwoLogSigZ} -- OK 

	local sigTi = nn.Exp()( nn.MulConstant(-1)( logSigT ) ) 
	local sigZ  = nn.Exp()(                     logSigZ ) 

	local lInversePrior = nn.SelectTable(3)( inode ) 
	lInversePrior = lowerTriangle( dim )( lInversePrior ) 
	lInversePrior = nn.Replicate( nsmpl, 2 )( lInversePrior ) 
	lInversePrior = nn.Reshape( mb*nsmpl, dim, dim )( lInversePrior ) 

	local deltaMu    = nn.View( mb * nsmpl, dim, 1 )( nn.CSubTable()({muZ, muT}) ) -- [mb * nsmpl, dim, 1]
	local deltaMuRot = nn.MM()({ lInversePrior, deltaMu })                         -- [mb * nsmpl, dim, 1]
	      deltaMuRot = nn.View( mb , nsmpl, dim, 1 )( deltaMuRot )                 -- [mb , nsmpl, dim, 1]
	local deltaSc    = nn.CMulTable()({deltaMuRot, sigTi})
	local mse        = nn.Square()( deltaSc ) 
	table.insert( terms, mse )  -- OK 

	local logSigP = nn.SelectTable(4)( inode )
	logSigP = nn.Replicate( nsmpl, 2 )( logSigP ) 
	local minusTwoLogSigP = nn.MulConstant( - 2 * nu / nsmpl )( logSigP )
	table.insert( terms, minusTwoLogSigP ) 

	local sigZ2 = nn.Square()( sigZ ) 
	local sigP2 = nn.Square()( nn.Exp()( logSigP ) )
	-- sigZ2 = nn.CAddTable()({ sigZ2, nn.MulConstant( (nu+dim+1)/nsmpl )( sigP2 ) })
	sigZ2 = nn.CAddTable()({ sigZ2, nn.MulConstant( nu/nsmpl )( sigP2 ) })
	local sigZ  = nn.Sqrt()( sigZ2 ) 
	sigZ  = nn.Replicate( dim, 3 )( sigZ  ) 
	-- sigZ  = nn.Sqrt()( nn.AddConstant( (nu+dim+1)/nsmpl )( nn.Square()( sigZ ) ) )
	sigTi = nn.Replicate( dim, 4 )( sigTi )
	local trace = nn.Sum(3)( nn.Square()( nn.CMulTable()({sigZ, sigTi, lInversePrior}) ) ) 
	table.insert( terms, trace ) -- OK 

	local kl = nn.CAddTable()( terms ) 
	kl = nn.AddConstant(-1)( kl ) 
	kl = nn.View(mb, -1)( kl ) 
	kl = nn.Sum(2)( kl )
	kl = nn.MulConstant( 0.5 )( kl ) 

	if priorAccDataset then 

		kl = nn.CAddTable()({ kl, IdentityWishartNLL( priorAccDataset )( logSigP ) })

	end

	local network = nn.gModule({ inode }, { kl })

	return network

end


----------------------------------------------------------------------------------------
--- tests ------------------------------------------------------------------------------
----------------------------------------------------------------------------------------

function testGaussianKLdiagonalLearnedMuStdPriorFull( makeFail )

	require 'common-torch/modules/triangle'

	nngraph.setDebug( true )

	local mb       = 5
	local muSigDim = 2 
	local dim      = 10
	local nsmpl    = 11

	local kl1 = GaussianKLdiagonalLearnedMuStd(          mb, muSigDim )
	local kl2 = GaussianKLdiagonalLearnedMuStdPriorFull( mb, muSigDim, nsmpl, dim, makeFail )

	local      a = torch.randn( mb, 2, nsmpl, dim ) 
	local priorA = torch.randn( mb, 2, nsmpl, dim ) 

	local e = torch.zeros( mb, dim, dim ) 

	local y1 = kl1:updateOutput{ a, priorA }
	local y2 = kl2:updateOutput{ a, priorA, e } 

	print( y1 )
	print( y2 )


end
-- testGaussianKLdiagonalLearnedMuStdPriorFull( false )
-- testGaussianKLdiagonalLearnedMuStdPriorFull( true )

--[[
function testGaussianKLdiagonalLearnedMuStdFull() -- OK: analytical and sampled solutions match. 

	nngraph.setDebug( true ) 

	local mb    = 1 
	local nsmpl = 11 
	local kl1   = GaussianKLdiagonalLearnedMuStdFull( mb, 2, nsmpl, 'scalar' )

	local z = torch.randn( mb, 2, nsmpl )
	local t = torch.randn( mb, 2, nsmpl ) 
	local r = torch.randn( mb, nsmpl, nsmpl ) 

	local muZ     = z:select( 2, 1 )
	local muT     = t:select( 2, 1 ) 
	local logSigZ = z:select( 2, 2 )
	local logSigT = t:select( 2, 2 ) 

	for i = 1, mb do r[i]:cmul( torch.tril( torch.ones(nsmpl,nsmpl), -1 ) ):add( torch.eye( nsmpl ) ) end 

	local o1 = kl1:updateOutput({ z, t, r }):squeeze()

	local sigZ = logSigZ:clone():exp() 
	local sigT = logSigT:clone():exp() 

	local nSamples = 100000
	local e = torch.Tensor( nsmpl )
	local z = torch.Tensor( nsmpl ) 
	local kl2 = torch.Tensor( nSamples ) 
	local de = e:clone()

	for i = 1, nSamples do 

		e:randn( nsmpl ) 
		z:mv( r:squeeze(), de:copy(e):cmul( sigZ ) ):add( muZ ) 

		kl2[i] = logSigT:sum() - logSigZ:sum() - ( e:norm()^2 )/2 + ( z:add( -1, muT ):cdiv( sigT ):norm()^2 )/2 

	end

	o2 = kl2:mean() 

	print( o1, o2, math.abs(o1-o2) ) 

end 
-- testGaussianKLdiagonalLearnedMuStdFull()

function testGaussianKLstandardVsLearned() 

	local mb = 11
	local muSigDim = 2 
	local nsmpl = 1 

	local posterior = torch.randn( mb, 2, nsmpl )
	local prior     = posterior:clone():zero()

	local kl0 = GaussianKLstandard( mb, muSigDim )
	local kl1 = GaussianKLdiagonalLearnedMuStd( mb, muSigDim )

	kl0:updateOutput(  posterior )
	kl1:updateOutput({ posterior, prior })

	print( kl0.output:dist( kl1.output ) ) 

end 

----------------------------------------------------------------------------------------
-- sample-based KL modules for unit testing analytical ones ----------------------------
----------------------------------------------------------------------------------------

require 'untangling-perceptual/modules/SampleGaussian'
require 'untangling-perceptual/modules/GaussianLikelihood'

function GaussianKLsampled( mu, sig, nsamples )

	local inode = nn.Identity()()

	local samples = nn.Replicate( nsamples, 2 )( inode ) 
	samples = SampleGaussianMuLogStd()( samples ) 

	local q = GaussianLikelihood( mu, sig )( samples )
	local p = GaussianLikelihood(  0,   1 )( samples )

	local kl = nn.Log()( nn.CDivTable()({q,p}) )
	kl = nn.Mean( 1 )( kl ) 

	local network = nn.gModule({ inode }, { kl })

	return network

end

function testGaussianKLstandard()

	local KL1 = GaussianKLstandard()

	for i = 1, 100 do 

		local muLogSig = torch.randn( 2, 1 ) 
		local  mu = 		  muLogSig[1]:squeeze()
		local sig = math.exp( muLogSig[2]:squeeze() ) 

		local KL2 = GaussianKLsampled( mu, sig, 100000 )

		local kl1 = KL1:updateOutput( muLogSig ) 
		local kl2 = KL2:updateOutput( muLogSig ) 

		print( kl1:dist( kl2 ) ) 

	end

end 
-- testGaussianKLstandard()



function testGaussianPosteriorFull() 

	local mb = 1 
	local n  = 11 

	local kl1 = GaussianKL_priorStandard_posteriorFull( mb, 2 )
	local kl2 = GaussianKLdiagonalLearnedMuStdFull(     mb, 2, n, 'scalar' )

	local z = torch.randn( mb, 2, n ) 
	local t = torch.randn( mb, 2, n ) 
	local r = torch.randn( mb, n, n )

	local o1 = kl1:updateOutput{ z,    r }
	local o2 = kl2:updateOutput{ z, t, r }
	print( o1:squeeze(), o2:squeeze() ) 

	t:zero()

	local o1 = kl1:updateOutput{ z,    r }
	local o2 = kl2:updateOutput{ z, t, r }
	print( o1:squeeze(), o2:squeeze() ) 

end
-- testGaussianPosteriorFull()


]]



