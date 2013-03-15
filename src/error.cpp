/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "error.h"

namespace eqMivt
{

namespace
{
struct ErrorData
{
	const uint32_t code;
	const std::string text;
};

ErrorData _errors[] = 
{
	{ ERROR_EQ_MIVT_FAILED, "mivt error" },
	{ ERROR_EQ_MIVT_FAILED_LOADING_DATA, "mivt error loading model data" },
	{ ERROR_EQ_MIVT_FAILED_LOADING_OCTREE_DATA, "mivt error loading octree file" },
	{ 0, "" } // last!
};
}

void initErrors()
{
	eq::fabric::ErrorRegistry& registry = eq::fabric::Global::getErrorRegistry();

	for( size_t i=0; _errors[i].code != 0; ++i )
		registry.setString( _errors[i].code, _errors[i].text );
}

void exitErrors()
{
	eq::fabric::ErrorRegistry& registry = eq::fabric::Global::getErrorRegistry();

	for( size_t i=0; _errors[i].code != 0; ++i )
		registry.eraseString( _errors[i].code );
}

}
