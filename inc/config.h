/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#ifndef EQ_CONFIG_H
#define EQ_CONFIG_H

// members
#include "initParams.h"

#include <eq/eq.h>
#include <eq/admin/base.h>

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

		void setInitParams( const InitParams& data ) { _initParams = data; }

		const InitParams& getInitParams() const { return _initParams; }

		/** @sa eq::Config::handleEvent */
		virtual bool handleEvent( eq::EventICommand command );

	protected:
		virtual ~Config();

	private:

		InitParams 	_initParams;

		bool _handleKeyEvent( const eq::KeyEvent& event );

};

}

#endif /* EQ_CONFIG_H */
