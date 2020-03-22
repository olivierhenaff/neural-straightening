function makeSpikeNL( spikeNL, spikeCountDistribution, nsmpl )

	if spikeNL == 'WhitenModulatedPoisson' then 

		return WhitenModulatedPoissonNL( spikeCountDistribution, nsmpl )

	elseif spikeNL == 'SquaredSoftPlus' then 

		return nn.Sequential():add( nn.SoftPlus() ):add( nn.Square() )

	else

		return nn[spikeNL]()

	end

end

function WhitenModulatedPoissonNL( spikeCountDistribution, nsmpl )

	local inode = nn.Identity()() 

	local memb     = nn.SelectTable(1)( inode ) 
	local logSigN  = nn.SelectTable(2)( inode ) 

	local sigG2   = nn.Exp()( nn.MulConstant(2)( logSigN ) )
	sigG2 = nn.MulConstant( trialDuration )( sigG2 )
	if spikeCountDistribution:find('Shared') then 
		local coupling    = nn.SelectTable(3)( inode )  
		local couplingVar = nn.Sum(3)( nn.Square()( coupling ) )
		sigG2 = nn.CAddTable()({ sigG2, couplingVar })
	end
	-- sigG2 = nn.Replicate( nsmpl, 2 )( sigG2 ) 

	rate = nn.SoftPlus()( memb ):annotate{ name = 'preRateTrajectory' }
	rate = nn.CMulTable()({ rate, nn.Sqrt()( sigG2 ) })
	rate = nn.MulConstant( 0.5 )( rate ) 

	local r1 = nn.Exp()( nn.MulConstant( 1)( rate )  ) 
	local r2 = nn.Exp()( nn.MulConstant(-1)( rate ) ) 
	rate = nn.CSubTable()({r1,r2})
	rate = nn.MulConstant( 0.5 )( rate ) 
	rate = nn.Square()( rate ) 
	rate = nn.CDivTable()({ rate, sigG2 }):annotate{ name = 'rateTrajectory' }

	return nn.gModule( {inode}, {rate} )

end


function HyperbolicSine()

	local inode = nn.Identity()()

	local e1 = nn.Exp()( nn.MulConstant(  1 )( inode )  ) 
	local e2 = nn.Exp()( nn.MulConstant( -1 )( inode ) ) 
	local output = nn.MulConstant( 0.5 )( nn.CSubTable()({ e1, e2 }) ) 

	local network = nn.gModule( {inode}, {output} )

	return network 

end
     
function effectiveSigG( data )

	local dim = data:size(3)

	local inode = nn.Identity()()

	local qParams  = nn.SelectTable( 1 )( inode ) -- [ mb, 2, nTrials, nsmpl, dim ] 
	local sParams  = nn.SelectTable( 2 )( inode ) -- [ mb, 2, nTrials, nsmpl, nModulators ]
	local coupling = nn.SelectTable( 3 )( inode ) -- [ mb, dim, nModulators ]

	local  muNi    =                        nn.Select( 2, 1 )( qParams )     -- [ mb, nTrials, nsmpl, dim         ] 
	local  muSi    =                        nn.Select( 2, 1 )( sParams )     -- [ mb, nTrials, nsmpl, nModulators ] 
	local sigN2i   = nn.Square()( nn.Exp()( nn.Select( 2, 2 )( qParams ) ) ) -- [ mb, nTrials, nsmpl, dim         ] 
	local sigS2i   = nn.Square()( nn.Exp()( nn.Select( 2, 2 )( sParams ) ) ) -- [ mb, nTrials, nsmpl, nModulators ] 

	muNi = nn.View( -1,              dim, 1 )( muNi ) 
	muSi = nn.View( -1, nModulatorsModel, 1 )( muSi ) 
	coupling = nn.Replicate( nTrials, 2 )( coupling ) -- [ mb, nTrials,        dim, nModulators ]
	coupling = nn.Replicate( nsmpl  , 3 )( coupling ) -- [ mb, nTrials, nsmpl, dim, nModulators ]

	local coupling2 = nn.Square()( coupling ) -- [ mb, nTrials, nsmpl, dim, nModulators ]

	coupling = nn.Copy( nil, nil, true )(  coupling ) 
	coupling = nn.View( mb*nTrials*nsmpl, dim, nModulatorsModel )( coupling ) 
	muNi = nn.CAddTable()({ muNi, nn.MM()({ coupling, muSi }) })
	muNi = nn.View( mb, nTrials, nsmpl, dim )( muNi ):annotate{ name = 'muNi' }

	sigS2i = nn.Replicate( dim, 4 )( sigS2i ) -- [ mb, nTrials, nsmpl, dim, nModulators ]
	local aux = nn.CMulTable()({ sigS2i, coupling2 }) -- [ mb, nTrials, nsmpl, dim, nModulators ]
	aux = nn.Sum( 5 )( aux )  -- [ mb, nTrials, nsmpl, dim ]

	sigN2i = nn.CAddTable()({ sigN2i, aux }):annotate{ name = 'sigN2i' }


	local muGi = nn.Exp()( nn.CAddTable()({ muNi, nn.MulConstant( 0.5 )( sigN2i ) }) )

	local muNiBis = nn.Exp()( nn.CAddTable()({ nn.MulConstant(2)( muNi ), sigN2i }) )

	local sigG2i = nn.CMulTable()({ nn.AddConstant( -1 )( nn.Exp()( sigN2i ) ), 
									muNiBis })

	local sigG2 = nn.CAddTable()({ sigG2i, nn.Square()( nn.AddConstant( -1 )( muGi ) ) })
	sigG2 = nn.Mean( 2 )( sigG2 )
	sigG2 = nn.Mean( 2 )( sigG2 )

	local sigG = nn.Sqrt()( sigG2 ) 

	local network = nn.gModule( { inode }, { sigG } )

	return network

end