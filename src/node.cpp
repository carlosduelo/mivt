/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "node.h"
#include "config.h"
#include "error.h"
#include "octreeContainer.h"

namespace eqMivt
{
bool Node::configInit( const eq::uint128_t& initID )
{
#if 0
	// All render data is static or multi-buffered, we can run asynchronously
	if( getIAttribute( IATTR_THREAD_MODEL ) == eq::UNDEFINED )
		setIAttribute( IATTR_THREAD_MODEL, eq::ASYNC );
#endif
	if( !eq::Node::configInit( initID ))
		return false;

	Config* config = static_cast< Config* >( getConfig( ));
	if( !config->loadData( initID, getPipes() ))
	{
		setError( ERROR_EQ_MIVT_FAILED_LOADING_DATA );
		return false;
	}
	
	const InitParams& 	initParams    	= config->getInitParams();
	const eq::uint128_t&  	frameDataID 	= initParams.getFrameDataID();


	// Creating CPU Cache
	vmml::vector<3, int> cubeDim;

	int nLevels = OctreeContainer::getnLevelsFromOctreeFile(initParams.getOctreeFile());

	int cDim = exp2(nLevels - initParams.getCubeLevel());
	cubeDim.set(cDim, cDim, cDim);

	if (!_cacheCPU.init(initParams.getTypeFile(), initParams.getDataFile(), initParams.getMaxElements_CPU(), cubeDim, initParams.getCubeInc(), initParams.getCubeLevel(), nLevels)) 
	{
		LBERROR<<"Error creating cpu cache"<<std::endl;
		return false;
	}

	return true;
}

}
