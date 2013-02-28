/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_H
#define EQ_MIVT_H

#include <eq/eq.h>

/** The Equalizer mivt implementation. */
namespace eqMivt
{
	/** The mivt application instance */
	class EqMivt : public eq::Client
	{
		public:
			EqMivt();
			virtual ~EqMivt() {}

			/** Run an EqMivt instance. */
			int run();

			/** @return a string containing an online help description. */
			static const std::string& getHelp();

		protected:
			/** @sa eq::Client::clientLoop. */
			virtual void clientLoop();
	};
	enum LogTopics
	{
		LOG_STATS = eq::LOG_CUSTOM << 0, // 65536
		LOG_CULL  = eq::LOG_CUSTOM << 1  // 131072
	};
}
#endif // EQ_MIVT_H
