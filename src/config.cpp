/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

// members
#include "config.h"

#include <math.h>

namespace eqMivt
{

Config::Config( eq::ServerPtr parent )	: eq::Config( parent )
{
}

Config::~Config()
{
}

bool Config::init()
{
	// Init frame data and register
	registerObject( &_frameData );

	// Add info to initParams and register
	_initParams.setFrameDataID( _frameData.getID( ));
	registerObject( &_initParams);

	// init config
	if( !eq::Config::init( _initParams.getID( )))
	{
		deregisterObject(&_initParams);
		deregisterObject(&_frameData);
		return false;
	}

	//_setMessage( "Welcome to eqMivt" );
	return true;
}

bool Config::exit()
{
	const bool ret = eq::Config::exit();
	deregisterObject(&_initParams);
	deregisterObject(&_frameData);

	// retain model & distributors for possible other config runs, dtor deletes
	return ret;
}

/** Map per-config data to the local node process */
bool Config::loadData( const eq::uint128_t& initParamsID, const std::vector<eq::Pipe*>& pipes)
{
	if( !_initParams.isAttached( ))
	{
		const uint32_t request = mapObjectNB( &_initParams, initParamsID, co::VERSION_OLDEST, getApplicationNode( ));

		if( !mapObjectSync( request ))
			return false;
		unmapObject( &_initParams ); // data was retrieved, unmap immediately
	}
	else // appNode, _initData is registered already
	{
		LBASSERT( _initParams.getID() == initParamsID );
	}

	// Check parameters are reached from each node
	if (!_initParams.checkParameters())
	{
		LBERROR<<"Error checking parameters"<<std::endl;
		return false;
	}

	return true; 
}

uint32_t Config::startFrame()
{
	//_updateData();
	const eq::uint128_t& version = _frameData.commit();

	//_redraw = false;
	return eq::Config::startFrame(version);
}



#if 0
bool Config::handleEvent( eq::EventICommand command )
{
#if 0
	switch( command.getEventType( ))
	{
		case eq::Event::KEY_PRESS:
			{
				const eq::Event& event = command.get< eq::Event >();
				if( _handleKeyEvent( event.keyPress ))
				{
					//_redraw = true;
					return true;
				}
				break;
			}
		case eq::Event::CHANNEL_POINTER_BUTTON_PRESS:
		case eq::Event::CHANNEL_POINTER_BUTTON_RELEASE:
		case eq::Event::CHANNEL_POINTER_MOTION:
		case eq::Event::CHANNEL_POINTER_WHEEL:
		case eq::Event::MAGELLAN_AXIS:
		case eq::Event::MAGELLAN_BUTTON:
		case eq::Event::WINDOW_EXPOSE:
		case eq::Event::WINDOW_RESIZE:
		case eq::Event::WINDOW_CLOSE:
		case eq::Event::VIEW_RESIZE:
		//	_redraw = true;
			break;

		case IDLE_AA_LEFT:
		default:
			break;
	}
	_redraw |= eq::Config::handleEvent( command );
#endif

	return eq::Config::handleEvent( command );;
}

bool Config::_handleKeyEvent( const eq::KeyEvent& event )
{
	switch( event.key )
	{
		case 'z':
			return true;

		case eq::KC_F1:
		case 'h':
		case 'H':
			return true;

	}
}
#endif
}
