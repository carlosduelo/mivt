/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include "channel.h"
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

void Channel::frameDraw( const eq::uint128_t& frameID )
{
	eq::Channel::frameDraw( frameID ); // Setup OpenGL state

 	eq::PixelViewport  viewport = getPixelViewport();


	std::cout<<getName()<<" "<<" .............>"<<viewport.x<<" "<<viewport.y<<" "<<viewport.h<<" "<<viewport.w<<std::endl;


	float * data = new float [3*viewport.h*viewport.w];
	for(int i=0; i<3*viewport.w*viewport.h; i+=3)
	{
		data[i]   = r;
		data[i+1] = g;
		data[i+2] = b;
	}
	glEnable(GL_SCISSOR_TEST);
	applyBuffer();
	applyViewport();
	glDrawPixels(viewport.w, viewport.h, GL_RGB, GL_FLOAT, data);

	delete[] data;
}

}
