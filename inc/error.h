/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_ERROR_H
#define EQ_MIVT_ERROR_H

#include <eq/eq.h>

namespace eqMivt
{
	/** Defines errors produced by mivt. */
	enum Error
	{
		ERROR_EQ_MIVT_FAILED = eq::ERROR_CUSTOM
	};

	/** Set up mivt-specific error codes. */
	void initErrors();

	/** Clear mivt-specific error codes. */
	void exitErrors();
}
#endif // EQ_MIVT_ERROR_H
