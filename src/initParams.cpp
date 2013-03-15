/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "initParams.h"
#include "mivt.h"

#include <tclap/CmdLine.h>
#include <boost/filesystem.hpp>

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

void InitParams::parseArguments(const int argc, const char ** argv)
{
	try
	{
		const std::string& desc = EqMivt::getHelp();
		TCLAP::CmdLine command( desc, ' ', eq::Version::getString( ));

		TCLAP::ValueArg<std::string> octree_file( "o", "octree-file", "File containing octree", true, "", "string", command );
		TCLAP::ValueArg<std::string> volume_file( "i", "volume-file", "File containing volume", true, "", "string", command );

		TCLAP::ValueArg<int> maxLevel_Octree( "m", "max-level", "Max Level Ocotree", true, 0, "int", command );

		TCLAP::UnlabeledMultiArg< std::string > ignoreArgs( "ignore", "Ignored unlabeled arguments", false, "any", command );

		command.parse(argc, argv);

		LBINFO << " Parameters mivt: "<<octree_file.getValue()<<" max level "<<maxLevel_Octree.getValue() <<" " << volume_file.getValue()<<std::endl;

		_octreePathFile 	= octree_file.getValue();
		_dataPathFile		= volume_file.getValue();
		_maxLevel		= maxLevel_Octree.getValue(); 
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
	if (!boost::filesystem::exists(_dataPathFile))
	{
		LBERROR << "Cannot open "<<_dataPathFile<< " file."<< std::endl;
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
