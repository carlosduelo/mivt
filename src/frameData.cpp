/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include "frameData.h"

namespace eqMivt
{
FrameData::FrameData()
{
}

void FrameData::serialize( co::DataOStream& os, const uint64_t dirtyBits )
{
#if 0
	co::Serializable::serialize( os, dirtyBits );
	if( dirtyBits & DIRTY_CAMERA )
		os << _position << _rotation << _modelRotation;
	if( dirtyBits & DIRTY_FLAGS )
		os << _modelID << _renderMode << _colorMode << _quality << _ortho
			<< _statistics << _help << _wireframe << _pilotMode << _idle
			<< _compression;
	if( dirtyBits & DIRTY_VIEW )
		os << _currentViewID;
	if( dirtyBits & DIRTY_MESSAGE )
		os << _message;
#endif
}

void FrameData::deserialize( co::DataIStream& is, const uint64_t dirtyBits )
{
#if 0
	co::Serializable::deserialize( is, dirtyBits );
	if( dirtyBits & DIRTY_CAMERA )
		is >> _position >> _rotation >> _modelRotation;
	if( dirtyBits & DIRTY_FLAGS )
		is >> _modelID >> _renderMode >> _colorMode >> _quality >> _ortho
			>> _statistics >> _help >> _wireframe >> _pilotMode >> _idle
			>> _compression;
	if( dirtyBits & DIRTY_VIEW )
		is >> _currentViewID;
	if( dirtyBits & DIRTY_MESSAGE )
		is >> _message;
#endif
}
}
