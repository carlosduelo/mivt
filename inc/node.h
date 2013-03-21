/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#ifndef EQ_MIVT_NODE_H
#define EQ_MIVT_NODE_H

#include "initParams.h"
#include "cubeCacheCPU.h"

namespace eqMivt
{
class Node : public eq::Node
{
	public:
		Node( eq::Config* parent ) : eq::Node( parent ) {}

		cubeCacheCPU	* getCubeCacheCPU(){ return &_cacheCPU; } 

	protected:
		virtual ~Node(){}

		virtual bool configInit( const eq::uint128_t& initID );

	private:
		cubeCacheCPU		_cacheCPU;
};

}

#endif /* EQ_MIVT_NODE_H */
