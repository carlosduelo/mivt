/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#ifndef EQ_MIVT_FRAMEDATA_H
#define EQ_MIVT_FRAMEDATA_H

#include <eq/eq.h>

namespace eqMivt
{

class FrameData : public co::Serializable
{
	public:
		FrameData();

		virtual ~FrameData() {};

		void reset();

		void setCameraPosition( const eq::Vector3f& position );
		void setRotation( const eq::Vector3f& rotation);
		void spinCamera( const float x, const float y );
		void moveCamera( const float x, const float y, const float z );

		const eq::Matrix4f& getCameraRotation() const { return _rotation; }
		const eq::Matrix4f& getModelRotation() const { return _modelRotation; }
		const eq::Vector3f& getCameraPosition() const { return _position; }

	protected:
		/** @sa Object::serialize() */
		virtual void serialize( co::DataOStream& os, const uint64_t dirtyBits );

		/** @sa Object::deserialize() */
		virtual void deserialize( co::DataIStream& is, const uint64_t dirtyBits );

		/** The changed parts of the data since the last pack(). */
		enum DirtyBits
		{
			DIRTY_CAMERA  = co::Serializable::DIRTY_CUSTOM << 0,
			DIRTY_FLAGS   = co::Serializable::DIRTY_CUSTOM << 1,
			DIRTY_VIEW    = co::Serializable::DIRTY_CUSTOM << 2,
			DIRTY_MESSAGE = co::Serializable::DIRTY_CUSTOM << 3,
		};
	private:
		eq::Matrix4f _rotation;
		eq::Matrix4f _modelRotation;
		eq::Vector3f _position;
};

}

#endif /* EQ_MIVT_FRAMEDATA_H */
