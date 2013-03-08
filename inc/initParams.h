/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/
#ifndef EQ_MIVT_INITPARAMS_H
#define EQ_MIVT_INITPARAMS_H

#include <eq/eq.h>

namespace eqMivt
{
class InitParams : public co::Object
{
	private:

		std::string _octreeiPathFile;

		std::string _dataPathFile;

		uint32_t    _maxFrames;

	public:
		InitParams();

		virtual ~InitParams();

		void parseArguments(const int argc, const char ** argv);

		bool checkParameters();

		uint32_t getMaxFrames()   const { return _maxFrames; }

	protected:
	virtual void getInstanceData( co::DataOStream& os );

	virtual void applyInstanceData( co::DataIStream& is );
};
}
#endif // EQ_MIVT_INITPARMAS
