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

void FrameData::spinCamera( const float x, const float y )
{
	if( x == 0.f && y == 0.f )
		return;

	_rotation.pre_rotate_x( x );
	_rotation.pre_rotate_y( y );
	setDirty( DIRTY_CAMERA );
}

void FrameData::spinModel( const float x, const float y, const float z )
{
	if( x == 0.f && y == 0.f && z == 0.f )
		return;

	_modelRotation.pre_rotate_x( x );
	_modelRotation.pre_rotate_y( y );
	_modelRotation.pre_rotate_z( z );
	setDirty( DIRTY_CAMERA );
}

void FrameData::moveCamera( const float x, const float y, const float z )
{
	eq::Matrix4f matInverse;
	compute_inverse( _rotation, matInverse );
	eq::Vector4f shift = matInverse * eq::Vector4f( x, y, z, 1 );

	_position += shift;

	setDirty( DIRTY_CAMERA );
}

void FrameData::setCameraPosition( const eq::Vector3f& position )
{
	_position = position;
	setDirty( DIRTY_CAMERA );
}

void FrameData::setRotation( const eq::Vector3f& rotation )
{
	_rotation = eq::Matrix4f::IDENTITY;
	_rotation.rotate_x( rotation.x() );
	_rotation.rotate_y( rotation.y() );
	_rotation.rotate_z( rotation.z() );
	setDirty( DIRTY_CAMERA );
}

void FrameData::setModelRotation(  const eq::Vector3f& rotation )
{
	_modelRotation = eq::Matrix4f::IDENTITY;
	_modelRotation.rotate_x( rotation.x() );
	_modelRotation.rotate_y( rotation.y() );
	_modelRotation.rotate_z( rotation.z() );
	setDirty( DIRTY_CAMERA );
}

void FrameData::reset()
{
	eq::Matrix4f model = eq::Matrix4f::IDENTITY;
	model.rotate_x( static_cast<float>( -M_PI_2 ));
	model.rotate_y( static_cast<float>( -M_PI_2 ));

	if( _position == eq::Vector3f( 0.f, 0.f, -2.f ) &&
			_rotation == eq::Matrix4f::IDENTITY && _modelRotation == model )
	{
		_position.z() = 0.f;
	}
	else
	{
		_position   = eq::Vector3f::ZERO;
		_position.z() = -2.f;
		_rotation      = eq::Matrix4f::IDENTITY;
		_modelRotation = model;
	}
	setDirty( DIRTY_CAMERA );
}


void FrameData::serialize( co::DataOStream& os, const uint64_t dirtyBits )
{
	co::Serializable::serialize( os, dirtyBits );
	if( dirtyBits & DIRTY_CAMERA )
		os << _position << _rotation << _modelRotation;
#if 0
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
	co::Serializable::deserialize( is, dirtyBits );
	if( dirtyBits & DIRTY_CAMERA )
		is >> _position >> _rotation >> _modelRotation;
#if 0
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
