/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include "channel.h"

#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glut.h>

namespace eqMivt
{

Channel::Channel(eq::Window* parent) : eq::Channel(parent)
{
}

void Channel::frameDraw( const eq::uint128_t& frameID )
{
	eq::Channel::frameDraw( frameID ); // Setup OpenGL state
/*
	//return pipe->getFrameData();
*/	

	float * data = new float [3*100*100];
	for(int i=0; i<3*100*100; i+=3)
		data[i] = 1.0f;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDrawPixels(100, 100, GL_RGB, GL_FLOAT, data);

	delete[] data;
	std::cout<<"CACCAAAAAAAAAAAAAAAAAAAAAAAAAAA"<<std::endl;
}

}
