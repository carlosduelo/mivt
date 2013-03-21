/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "initParams.h"
#include "mivt.h"

#include <tclap/CmdLine.h>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>
#include <stdlib.h>

namespace eqMivt
{

InitParams::InitParams() : 	_maxFrames( 0xffffffffu ),
				_frameDataID()
				
{
}

InitParams::~InitParams()
{
	setFrameDataID(0);
}

bool InitParams::parseArguments(const int argc, const char ** argv)
{
	try
	{
		const std::string& desc = EqMivt::getHelp();
		TCLAP::CmdLine command( desc, ' ', eq::Version::getString( ));

		TCLAP::ValueArg<std::string> octree_file( "o", "octree-file", "File containing octree: path:max_level", true, "", "string", command );
		TCLAP::ValueArg<std::string> volume_file( "d", "data", "File containing volume type:type_file:file_path:options:level_cube", true, "", "string", command );

		TCLAP::ValueArg<int> cacheCPU( "c", "cpu-max-elements", "elements in cpu cache", true, 1, "int", command );
		
		TCLAP::ValueArg<int> cacheGPU( "g", "gpu-max-elements", "elements in gpu cache", true, 1, "int", command );

		TCLAP::UnlabeledMultiArg< std::string > ignoreArgs( "ignore", "Ignored unlabeled arguments", false, "any", command );

		command.parse(argc, argv);

		boost::char_separator<char> sep(":");

		// Octree parameters
		boost::tokenizer< boost::char_separator<char> > tokensO(octree_file.getValue(), sep);
		int i = 0;
		BOOST_FOREACH (const std::string& t, tokensO)
		{
			if (i==0) 
			{
				_octreePathFile = t;
			}
			else if (i==1)
			{
				_maxLevel = atoi(t.c_str());
			}	
			i++;
		}
		if (i > 2)
			return false;

		// Data parameters
		_cubeInc = 2;
		boost::tokenizer< boost::char_separator<char> > tokensD(volume_file.getValue(), sep);
		int size = 0;
		BOOST_FOREACH (const std::string& t, tokensD)
			size++;
		i = 0;
		BOOST_FOREACH (const std::string& t, tokensD)
		{
			if (i==0)
				_type_file = t;
			else if (i == (size-1))
				_cubeLevel = atoi(t.c_str());
			else
				_dataPathFile.push_back(t);

			i++;
		}

		if (_maxLevel < _cubeLevel)
		{
			LBERROR<<"Cube level have to be <= max level octree"<<std::endl;
			return false;
		}

		// Cache cpu parameters
		_maxElements_CPU = cacheCPU.getValue();

		// Cache gpu parameters
		_maxElements_GPU = cacheGPU.getValue();

		return true;

	}
	catch (const TCLAP::ArgException& exception)
	{
		LBERROR << "Command line parse error: " << exception.error()
			<< " for argument " << exception.argId() << std::endl;
		::exit( EXIT_FAILURE );
	}
}

bool InitParams::checkParameters()
{
	//Check Parameters are OK!
	if (!boost::filesystem::exists(_octreePathFile))
	{
		LBERROR << "Cannot open "<<_octreePathFile<< " file."<< std::endl;
		return false;
	}
	if (!boost::filesystem::exists(_dataPathFile[0]))
	{
		LBERROR << "Cannot open "<<_dataPathFile[0]<< " file."<< std::endl;
		return false;
	}
	return true;	
}

void InitParams::getInstanceData( co::DataOStream& os )
{
	os << _frameDataID; 
}

void InitParams::applyInstanceData( co::DataIStream& is )
{
	is >> _frameDataID;
}

}
