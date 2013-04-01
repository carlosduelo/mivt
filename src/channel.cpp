/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include "channel.h"
#include "frameData.h"
#include "pipe.h"

#include <stdlib.h>

#include "eq/client/gl.h"

namespace eqMivt
{

Channel::Channel(eq::Window* parent) : eq::Channel(parent)
{
	r = (float) rand()/(float)RAND_MAX;
	g = (float) rand()/(float)RAND_MAX;
	b = (float) rand()/(float)RAND_MAX;
}

bool Channel::configInit( const eq::uint128_t& initID )
{
	if( !eq::Channel::configInit( initID ))
		return false;

	setNearFar( 0.1f, 10.0f );

	return true;
}

bool Channel::configExit()
{
	return eq::Channel::configExit();
}

void Channel::frameDraw( const eq::uint128_t& frameID )
{
	EQ_GL_CALL( glEnable(GL_SCISSOR_TEST));
	EQ_GL_CALL( applyBuffer() );
	EQ_GL_CALL( applyViewport() );
	//eq::Channel::frameDraw( frameID ); // Setup OpenGL state

	// Get Camera
	const Pipe* pipe = static_cast<const Pipe*>( getPipe( ));
	const FrameData& frameData = pipe->getFrameData();

	// Camera transformations
	const eq::Vector3f& position = frameData.getCameraPosition();
	EQ_GL_CALL( glMultMatrixf( frameData.getCameraRotation().array) );

	EQ_GL_CALL( glTranslatef( position.x(), position.y(), position.z() ) );

	std::cout<<position<<std::endl;

	float modelview[16];

	EQ_GL_CALL( glGetFloatv(GL_MODELVIEW_MATRIX , modelview) );
	std::cout<<modelview[0]<<" "<<modelview[1]<<" "<<modelview[2]<<std::endl;
	std::cout<<modelview[4]<<" "<<modelview[5]<<" "<<modelview[6]<<std::endl;
	std::cout<<modelview[8]<<" "<<modelview[9]<<" "<<modelview[10]<<std::endl;


 	eq::PixelViewport  viewport = getPixelViewport();

	std::cout<<getName()<<" "<<" .............>"<<viewport.x<<" "<<viewport.y<<" "<<viewport.h<<" "<<viewport.w<<std::endl;


	float * data = new float [3*viewport.h*viewport.w];
	for(int i=0; i<3*viewport.w*viewport.h; i+=3)
	{
		data[i]   = r;
		data[i+1] = g;
		data[i+2] = b;
	}

	EQ_GL_CALL( glDrawPixels(viewport.w, viewport.h, GL_RGB, GL_FLOAT, data) );
	delete[] data;

	//drawStatistics();
}

void 	Channel::frameTilesStart (const eq::uint128_t &frameID)
{
	std::cout<<"Start tile"<<std::endl;
	eq::Channel::frameTilesStart (frameID);
}

void 	Channel::frameTilesFinish (const eq::uint128_t &frameID)
{
	std::cout<<"End tile"<<std::endl;
	eq::Channel::frameTilesFinish (frameID);
}

}
