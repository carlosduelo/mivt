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
				_octreeiPathFile( "" ),
				_dataPathFile( "")
				
{
}

InitParams::~InitParams()
{
}

void InitParams::parseArguments(const int argc, const char ** argv)
{
	const std::string& desc = EqMivt::getHelp();
	TCLAP::CmdLine command( desc, ' ', eq::Version::getString( ));

	TCLAP::ValueArg<std::string> octree_file( "o", "octree-file", "File containing octree", true, "", "string", command );
	TCLAP::ValueArg<std::string> volumee_file( "i", "volume-file", "File containing volume", true, "", "string", command );

	command.parse(argc, argv);

	LBINFO << " Parameters mivt: "<<octree_file.getValue()<< " " << volumee_file.getValue()<<std::endl;

	_octreeiPathFile 	= octree_file.getValue();
	_dataPathFile		= volumee_file.getValue();
}

bool InitParams::checkParameters()
{
	//Check Parameters are OK!
	if (!boost::filesystem::exists(_octreeiPathFile))
	{
		LBERROR << "Cannot open "<<_octreeiPathFile<< " file."<< std::endl;
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
	os << _octreeiPathFile << _dataPathFile; 
}

void InitParams::applyInstanceData( co::DataIStream& is )
{
	is >> _octreeiPathFile >> _dataPathFile;
}

}
