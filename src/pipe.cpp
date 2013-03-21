/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "pipe.h"
#include "node.h"
#include "cuda_runtime.h"

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

	Node *  node =  static_cast<Node*>(getNode());
#if 0
int dev = -1;
cudaGetDevice(&dev);
std::cout<<"-----------------------------------------------___>"<<dev<<std::endl;
#endif

	// Reading octree file
	if (!_octreeContainer.readOctreeFile(initParams.getOctreeFile(), initParams.getMaxLevel()))
	{
		LBERROR<<"Error creating octree container"<<std::endl;
		return false;
	}

	// Creating gpu cache
	if (!_cache.init(node->getCubeCacheCPU(), 4, initParams.getMaxElements_GPU()))
	{
		LBERROR<<"Error creating cache in pipe"<<std::endl;
		return false;
	}

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
