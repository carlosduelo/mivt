/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#ifndef EQ_MIVT_NODE_H
#define EQ_MIVT_NODE_H

#include "initParams.h"

namespace eqMivt
{
class Node : public eq::Node
{
	public:
		Node( eq::Config* parent ) : eq::Node( parent ) {}

	protected:
		virtual ~Node(){}

		virtual bool configInit( const eq::uint128_t& initID );

	private:
};

}

#endif /* EQ_MIVT_NODE_H */
