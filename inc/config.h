/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#ifndef EQ_MIVT_CONFIG_H
#define EQ_MIVT_CONFIG_H

// members
#include "initParams.h"
#include "frameData.h"
#include "octreeContainer.h"

namespace eqMivt
{

class Config : public eq::Config
{
	public:
		Config( eq::ServerPtr parent );

		/** @sa eq::Config::init. */
		virtual bool init();

		/** @sa eq::Config::exit. */
		virtual bool exit();

		/** @sa eq::Config::startFrame. */
		virtual uint32_t startFrame();

		void setInitParams( const eqMivt::InitParams& data ) { _initParams = data; }

		const InitParams& getInitParams() const { return _initParams; }

		/** @sa eq::Config::handleEvent */
		//virtual bool handleEvent( eq::EventICommand command );

		/** Map per-config data to the local node process */
	        bool loadData( const eq::uint128_t& initParamsID );

	protected:
		virtual ~Config();

	private:

		eqMivt::InitParams 	_initParams;
		eqMivt::FrameData	_frameData;

		OctreeContainer 	_octreeContainer;

		//bool _handleKeyEvent( const eq::KeyEvent& event );
};

}

#endif /* EQ_MIVT_CONFIG_H */
