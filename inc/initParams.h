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
		std::string 			_octreePathFile;
		int				_maxLevel;

		// Data Parameters
		std::string			_type_file;
		std::vector<std::string> 	_dataPathFile;
		int				_cubeInc;
		int				_cubeLevel;

		// Cache CPU parameters
		int				_maxElements_CPU;

		// Cache GPU parameters
		int				_maxElements_GPU;

		uint32_t    			_maxFrames;

		eq::uint128_t   		_frameDataID;

	public:
		InitParams();

		virtual ~InitParams();

		bool parseArguments(const int argc, const char ** argv);

		bool checkParameters();

		eq::uint128_t getFrameDataID() const  		{ return _frameDataID; }
		void setFrameDataID( const eq::uint128_t& id ) 	{ _frameDataID = id; }

		uint32_t getMaxFrames()   const { return _maxFrames; }

		// Data Parameters
		std::string			getTypeFile() const { return _type_file; }
		std::vector<std::string> 	getDataFile() const { return _dataPathFile; }
		int				getCubeInc() const { return _cubeInc; }
		int				getCubeLevel() const { return _cubeLevel; }

		// Octree Parameters
		std::string 	getOctreeFile() const { return _octreePathFile; }
		int 		getMaxLevel() const { return _maxLevel; } 

		// Cache CPU parameters
		int	getMaxElements_CPU() const { return _maxElements_CPU;}

		// Cache GPU parameters
		int	getMaxElements_GPU() const { return _maxElements_GPU; }


	protected:
		virtual void getInstanceData( co::DataOStream& os );

		virtual void applyInstanceData( co::DataIStream& is );
};
}
#endif // EQ_MIVT_INITPARMAS
