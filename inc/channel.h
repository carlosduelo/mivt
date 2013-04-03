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


	protected:
		virtual ~Channel() {}
		virtual bool configInit( const eq::uint128_t& initID );
		virtual bool configExit();
		virtual void frameTilesStart (const eq::uint128_t &frameID);
		virtual void frameTilesFinish (const eq::uint128_t &frameID);
		virtual void frameDraw( const eq::uint128_t& frameID );
		virtual void frameStart( const eq::uint128_t& frameID, const uint32_t frameNumber );
		virtual void frameFinish( const eq::uint128_t& frameID,const uint32_t frameNumber );
		virtual void frameReadback( const eq::uint128_t& frameID );
		virtual void frameAssemble( const eq::uint128_t& frameID );
	private:
	float r;
	float g;
	float b;
	float * data;
	eq::PixelViewport currentViewport;
	float * screen;

};
}

#endif /* EQ_MIVT_CHANNEL */
