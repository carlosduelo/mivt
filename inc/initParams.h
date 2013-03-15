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

		// Octree Parameters
		std::string 	_octreePathFile;
		int		_maxLevel;

		std::string 	_dataPathFile;

		uint32_t    	_maxFrames;

		eq::uint128_t   _frameDataID;

	public:
		InitParams();

		virtual ~InitParams();

		void parseArguments(const int argc, const char ** argv);

		bool checkParameters();

		eq::uint128_t getFrameDataID() const  		{ return _frameDataID; }
		void setFrameDataID( const eq::uint128_t& id ) 	{ _frameDataID = id; }

		std::string 	getDataFile() { return _dataPathFile; }

		std::string 	getOctreeFile() { return _octreePathFile; }

		int 		getMaxLevel() { return _maxLevel; } 

		uint32_t getMaxFrames()   const { return _maxFrames; }

	protected:
		virtual void getInstanceData( co::DataOStream& os );

		virtual void applyInstanceData( co::DataIStream& is );
};
}
#endif // EQ_MIVT_INITPARMAS
