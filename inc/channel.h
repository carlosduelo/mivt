/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#ifndef EQ_MIVT_CHANNEL_H
#define EQ_MIVT_CHANNEL_H

#include <eq/eq.h>

namespace eqMivt
{
class Channel : public eq::Channel
{
	public:
		Channel( eq::Window* parent );

		virtual void frameDraw( const eq::uint128_t& frameID );

	protected:
		virtual ~Channel() {}
		virtual bool configInit( const eq::uint128_t& initID );
		virtual bool configExit();
	private:
	float r;
	float g;
	float b;
};
}

#endif /* EQ_MIVT_CHANNEL */
