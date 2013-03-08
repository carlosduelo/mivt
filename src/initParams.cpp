/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "initParams.h"

namespace eqMivt
{

InitParams::InitParams() : _maxFrames( 0xffffffffu )
{
}

InitParams::~InitParams()
{
}

void InitParams::parseArguments(const int argc, const char ** argv)
{
}

bool InitParams::checkParameters()
{
	//Check Parameters are OK!
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
