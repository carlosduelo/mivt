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
};

}

#endif /* EQ_MIVT_FRAMEDATA_H */
