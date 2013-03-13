/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "pipe.h"

namespace eqMivt
{

#if 0
eq::WindowSystem Pipe::selectWindowSystem() const
{
	const Config* config = static_cast<const Config*>( getConfig( ));
	return config->getInitData().getWindowSystem();
}
#endif

bool Pipe::configInit( const eq::uint128_t& initID )
{
	if( !eq::Pipe::configInit( initID ))
		return false;

	Config*         config      		= static_cast<Config*>( getConfig( ));
	const InitParams& 	initParams    	= config->getInitParams();
	const eq::uint128_t&  	frameDataID 	= initParams.getFrameDataID();

	return config->mapObject( &_frameData, frameDataID );
}

bool Pipe::configExit()
{
	eq::Config* config = getConfig();
	config->unmapObject( &_frameData );

	return eq::Pipe::configExit();
}

void Pipe::frameStart( const eq::uint128_t& frameID, const uint32_t frameNumber)
{
	eq::Pipe::frameStart( frameID, frameNumber );
	_frameData.sync( frameID );
}
}
