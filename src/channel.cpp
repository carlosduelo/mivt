/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include "channel.h"
#include "frameData.h"
#include "pipe.h"

#include <stdlib.h>

#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glut.h>

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
	eq::Channel::frameDraw( frameID ); // Setup OpenGL state

	// Get Camera
	const Pipe* pipe = static_cast<const Pipe*>( getPipe( ));
	const FrameData& frameData = pipe->getFrameData();

	// Camera transformations
	const eq::Vector3f& position = frameData.getCameraPosition();
	glMultMatrixf( frameData.getCameraRotation().array );
	glTranslatef( position.x(), position.y(), position.z() );

	std::cout<<position<<std::endl;

	float modelview[16];

	glGetFloatv(GL_MODELVIEW_MATRIX , modelview);
	std::cout<<modelview[0]<<" "<<modelview[1]<<" "<<modelview[2]<<std::endl;
	std::cout<<modelview[4]<<" "<<modelview[5]<<" "<<modelview[6]<<std::endl;
	std::cout<<modelview[8]<<" "<<modelview[9]<<" "<<modelview[10]<<std::endl;


 	eq::PixelViewport  viewport = getPixelViewport();

#if 0
	glLineWidth(10); 
	glBegin(GL_LINES); 
	glVertex2i( 2.0f, 2.0f); 
	glVertex2i( 0.0f, 0.0f); 
	glEnd(); 
#endif

	std::cout<<getName()<<" "<<" .............>"<<viewport.x<<" "<<viewport.y<<" "<<viewport.h<<" "<<viewport.w<<std::endl;

	glEnable(GL_SCISSOR_TEST);
	//applyBuffer();
	//applyViewport();
	//glScissor(viewport.x,viewport.y,viewport.w, viewport.h);


#if 1
	float * data = new float [3*viewport.h*viewport.w];
	for(int i=0; i<3*viewport.w*viewport.h; i+=3)
	{
		data[i]   = r;
		data[i+1] = g;
		data[i+2] = b;
	}
	glDrawPixels(viewport.w, viewport.h, GL_RGB, GL_FLOAT, data);
	delete[] data;
#endif
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
