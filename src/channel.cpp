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
	data = new float [3*256*256];
	for(int i=0; i<3*256*256; i+=3)
	{
		data[i]   = r;
		data[i+1] = g;
		data[i+2] = b;
	}
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
	if (data != 0)
		delete[] data;
	if (screen != 0)
		delete[] screen;
	return eq::Channel::configExit();
}



void Channel::frameDraw( const eq::uint128_t& frameID )
{
	//EQ_GL_CALL( glEnable(GL_SCISSOR_TEST));
	//EQ_GL_CALL( eq::Channel::applyBuffer() );
	//EQ_GL_CALL( eq::Channel::applyViewport() );
	//eq::Channel::frameDraw( frameID ); // Setup OpenGL state

 	eq::PixelViewport  viewport = getPixelViewport();
	std::cout<<getName()<<" "<<" .............>"<<viewport.x<<" "<<viewport.y<<" "<<viewport.h<<" "<<viewport.w<<std::endl;
	int a =0;
	for(int i=0; i<viewport.w; i++)
	{
		for(int j=0; j<viewport.h; j++)
		{
			int pos = 3*(i+viewport.x) + 3*currentViewport.w*(j+viewport.y);
			//std::cout<<pos<<std::endl;
			screen[pos]=1.0f;//(float)viewport.x/(float)currentViewport.w;
			screen[pos+1]=0.0f;//(float)viewport.y/(float)currentViewport.h;
			screen[pos+2]=0.0f;
			a++;
		}
	}
std::cout<<a<<std::endl;
#if 0
	glLineWidth(1); 
	glBegin(GL_LINES); 
	glVertex2i( 0,0); 
	glVertex2i( viewport.w, viewport.h); 
	glEnd(); 
#endif

#if 0
	eq::util::FrameBufferObject* buffer = getFrameBufferObject();
	if (buffer == 0)
		std::cout<<"CCCCCCCCCCAAAAAAAAAAAAAACCCCCCCCCAAAAAAAAAA"<<std::endl;

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

//	std::cout<<getName()<<" "<<" .............>"<<viewport.x<<" "<<viewport.y<<" "<<viewport.h<<" "<<viewport.w<<std::endl;

	EQ_GL_CALL( glDrawPixels(viewport.w, viewport.h, GL_RGB, GL_FLOAT, data) );
//	EQ_GL_CALL( "AHORA")

	//drawStatistics();
	#endif

}

void Channel::frameStart( const eq::uint128_t& frameID, const uint32_t frameNumber )
{
	std::cout<<"Frame Start "<<frameNumber<<std::endl;

	// Check current Viewport if different resize buffer
 	eq::PixelViewport  viewport = getPixelViewport();
	if (viewport != currentViewport)
	{
		currentViewport = viewport;

		if (screen != 0)
			delete[] screen;
		screen = new float[3*currentViewport.w*currentViewport.h];
	}
	std::cout<<getName()<<" "<<" .............>"<<viewport.x<<" "<<viewport.y<<" "<<viewport.h<<" "<<viewport.w<<std::endl;
	eq::Channel::frameStart(frameID, frameNumber);
}

void Channel::frameFinish( const eq::uint128_t& frameID,const uint32_t frameNumber )
{
	EQ_GL_CALL( glDrawPixels(currentViewport.w, currentViewport.h, GL_RGB, GL_FLOAT, screen) );
	std::cout<<"Frame Finish "<<frameNumber<<std::endl;
	eq::Channel::frameFinish(frameID, frameNumber);
}

void Channel::frameReadback( const eq::uint128_t& frameID )
{
	std::cout<<"Frame Readback"<<std::endl;
	eq::Channel::frameReadback( frameID );
}

void Channel::frameAssemble( const eq::uint128_t& frameID )
{
	std::cout<<"Frame Assemble"<<std::endl;
	eq::Channel::frameAssemble(frameID);
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
