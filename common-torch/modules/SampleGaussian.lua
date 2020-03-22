require 'nn'
require 'nngraph'

function setLogStdToNegInf( muSigDim ) 

	local muSigDim = muSigDim or 1 

	local inode      = nn.Identity()()

	local mean       = nn.Narrow(muSigDim,1,1)( inode )
	local logStd     = nn.Narrow(muSigDim,2,1)( inode )

	logStd           = nn.MulConstant(   0)( logStd ) 
	logStd           = nn.AddConstant(-999)( logStd ) 

	local meanLogStd = nn.JoinTable(muSigDim)({mean, logStd})

	local network = nn.gModule({ inode }, { meanLogStd })

	return network

end
--[[
function SampleGaussianMuLogStdRotation( muSigDim, mb, nsmpl )

	local muSigDim = muSigDim or 1 

	local inode   = nn.Identity()()

	local meanLogStd = nn.SelectTable( 1 )( inode ) 
	local rotation   = nn.SelectTable( 2 )( inode ) 

	local mean    = nn.Narrow(muSigDim,1,1)( meanLogStd )
	local logStd  = nn.Narrow(muSigDim,2,1)( meanLogStd )

	local zero    = nn.MulConstant( 0 )( mean )
	local std     = nn.Exp()( logStd )

	local meanStd = nn.JoinTable(muSigDim)({zero, std})

	local sample  = nn.SampleGaussianMuStd( muSigDim )( meanStd ) 
	sample = nn.View( mb , nsmpl, 1 )( sample )
	sample = nn.MM()({ rotation, sample })
	sample = nn.View( mb , nsmpl )( sample )
	sample = nn.CAddTable()({sample, mean})

	local network = nn.gModule({ inode }, { sample })

	return network

end

function SampleGaussianMuLogStdRotationEps( muSigDim, mb, nsmpl )

	local muSigDim = muSigDim or 1 

	local inode   = nn.Identity()()

	local meanLogStd = nn.SelectTable( 1 )( inode ) 
	local rotation   = nn.SelectTable( 2 )( inode ) 

	local mean    = nn.Narrow(muSigDim,1,1)( meanLogStd )
	local logStd  = nn.Narrow(muSigDim,2,1)( meanLogStd )

	local zero    = nn.MulConstant( 0 )( mean )
	local one     = nn.AddConstant( 1 )( zero ) 
	local zeroOne = nn.JoinTable(muSigDim)({zero, one})

	local epsilon = nn.SampleGaussianMuStd( muSigDim )( zeroOne ) 

	local std     = nn.Exp()( logStd )
	local sample  = nn.CMulTable()({ epsilon, std } ) 
	sample = nn.View( mb , nsmpl, 1 )( sample )
	sample = nn.MM()({ rotation, sample })
	sample = nn.View( mb , nsmpl )( sample )
	sample = nn.CAddTable()({sample, mean})

	local network = nn.gModule({ inode }, { sample, epsilon })

	return network

end

function SampleGaussianRotationLogStdMu( muSigDim, mb, nsmpl ) 

	local muSigDim = muSigDim or 1 

	local inode   = nn.Identity()()

	local meanLogStd = nn.SelectTable( 1 )( inode ) 
	local rotation   = nn.SelectTable( 2 )( inode ) 

	local mean    = nn.Narrow(muSigDim,1,1)( meanLogStd )
	local logStd  = nn.Narrow(muSigDim,2,1)( meanLogStd )
	local std     = nn.Exp()( logStd )

	local zero    = nn.MulConstant( 0 )( mean ) 
	local one     = nn.AddConstant( 1 )( zero ) 
	local zeroOne = nn.JoinTable(muSigDim)({zero, one})

	local sample  = nn.SampleGaussianMuStd( muSigDim )( zeroOne ) 
	sample = nn.View( mb, nsmpl, 1 )( sample )
	sample = nn.MM()({ rotation, sample })
	sample = nn.CMulTable()( {sample, std} ) 
	sample = nn.CAddTable()( {sample, mean} )

	local network = nn.gModule({ inode }, { sample })

	return network 

end
]]

function SampleGaussianMuLogStd( muSigDim )

	local muSigDim = muSigDim or 1 

	local inode   = nn.Identity()()

	local mean    = nn.Narrow(muSigDim,1,1)( inode )
	local logStd  = nn.Narrow(muSigDim,2,1)( inode )
	local std     = nn.Exp()( logStd )
	local meanStd = nn.JoinTable(muSigDim)({mean, std})
	local sample = nn.SampleGaussianMuStd( muSigDim )( meanStd ) 

	local network = nn.gModule({ inode }, { sample })

	return network

end


local sample, parent = torch.class('nn.SampleGaussianMuStd'   , 'nn.Module')
--[[takes as input a tensor whose muSigDim (default: 1) dimesion is 
1: mean of Gaussian 
2: std  of Gaussian 
]]

function sample:__init( muSigDim )

	parent.__init(self)

	self.muSigDim = muSigDim or 1 

end

function sample:updateOutput( input ) 

	local mu  = input:select( self.muSigDim, 1 )
	local sig = input:select( self.muSigDim, 2 )

	self.random = self.random or torch.Tensor():typeAs( input )
	self.random:randn( mu:size() )
	self.output:resizeAs( mu ):copy( mu ):addcmul( sig, self.random ) 

	return self.output 

end 

function sample:updateGradInput( input, gradOutput ) 

	self.gradInput:resizeAs( input ) 
	self.gradInput:select( self.muSigDim, 1 ):copy( gradOutput ) 
	self.gradInput:select( self.muSigDim, 2 ):copy( gradOutput ):cmul( self.random ) 

	return self.gradInput 

end 

function testSample()

	local inode = nn.Identity()()
	local mu  = nn.Narrow(1,1,1)( inode )
	local std = nn.Narrow(1,2,1)( inode ); std = nn.SoftPlus()( std ) 
	local muStd = nn.JoinTable(1)({mu,std})
	local smpl = nn.SampleGaussianMuStd()( muStd )
	smpl = nn.Square()( smpl )  

	local net = nn.gModule({ inode }, { smpl })
	local g = torch.Tensor{1}

	local x = torch.randn( 2, 1 )

	for i = 1, 10000 do 

		y = net:updateOutput( x ) 
		dx = net:updateGradInput( x, g )
		x:add( -0.1, dx )

		print( {x[1][1], x[2][1], y[1]} ) 

	end

end 
--testSample()

--[[
function sample:__init( muSigDim, nSamples )

	parent.__init(self)

	self.muSigDim = muSigDim or 1
	self.nSamples = nSamples or 1  

end

function sample:updateOutput( input ) 

	local mu  = input:narrow( self.muSigDim, 1, 1 )
	local sig = input:narrow( self.muSigDim, 2, 1 )

	self.random = self.random or torch.Tensor():typeAs( input )

	if not self.size then 

		self.size = input:size()
		self.size[ self.muSigDim ] = self.nSamples 

	end

	self.random:randn( self.size ) 
	self.output:resize(self.size ):copy( mu:expand(self.size) ):addcmul( sig:expand(self.size), self.random ) 

	return self.output 

end 

function sample:updateGradInput( input, gradOutput ) 

	self.gradInput:resizeAs( input ) 
	self.gradInput:select( self.muSigDim, 1 ):sum( gradOutput, self.muSigDim ) 
	self.gradInput:select( self.muSigDim, 2 ):sum( self.random:cmul( gradOutput ), self.muSigDim ) 

	return self.gradInput 

end 

function testSample()

	local muSigDim = 1 
	local nSamples = 10000

	local network = SampleGaussianMuLogStd( muSigDim, nSamples )

	local x = torch.randn( 2, 1 )
	local y = network:updateOutput( x ) 

	local mu  = y:mean()
	local sig = y:std()

	local target = torch.randn(1)[1]

	local gradOutput = torch.Tensor( nSamples )

	print( x )
	print( mu, math.log( sig ) ) 

	for i = 1, 10000 do 

		y  = network:updateOutput( x ) 

		gradOutput:fill( target ):add( -1, y )

		dx = network:updateGradInput( x, gradOutput )
		x:add( 0.1 / nSamples, dx )

		print( i, target, x[1][1]-target, math.exp(x[2][1]) ) 

	end

end 
-- testSample()
]]

