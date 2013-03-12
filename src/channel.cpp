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
}

void Channel::frameDraw( const eq::uint128_t& frameID )
{
	eq::Channel::frameDraw( frameID ); // Setup OpenGL state
/*
	//return pipe->getFrameData();
Hola Carlos,
Dentro de Channel::frameDraw puedes invocar getPixelViewport, que te devuelve un eq::PixelViewport con la esquina inferior izquierda del viewport, el ancho y el alto en píxels.
Aparte, para no tener que llamar a glDrawBuffer ni glViewport ni glScissor por tu cuenta, puedes invocar applyBuffer() y applyViewport() (GL_SCISSOR_TEST sí hay que habilitarlo a mano) y a continuación hacer glDrawPixels.
Eso debería funcionarte sin problemas. 

*/	



 	eq::PixelViewport  viewport = getPixelViewport();

	eq::Pipe * pipe = getPipe();


	float r = (float) rand()/(float)RAND_MAX;
	float g = (float) rand()/(float)RAND_MAX;
	float b = (float) rand()/(float)RAND_MAX;

	if (getName()[7]=='0')
	{
		r = 1.0f;
		g = 0.0f;
		b = 0.0f;
	}
	if (getName()[7]=='1')
	{
		r = 0.0f;
		g = 1.0f;
		b = 0.0f;
	}
	if (getName()[7]=='2')
	{
		r = 0.0f;
		g = 0.0f;
		b = 1.0f;
	}
	if (getName()[7]=='3')
	{
		r = 0.5f;
		g = 0.5f;
		b = 0.5f;
	}

	std::cout<<getName()[7]<<" "<<" .............>"<<viewport.x<<" "<<viewport.y<<" "<<viewport.h<<" "<<viewport.w<<std::endl;


	float * data = new float [3*viewport.h*viewport.w];
	for(int i=0; i<3*viewport.w*viewport.h; i+=3)
	{
		data[i]   = r;
		data[i+1] = g;
		data[i+2] = b;
	}
	applyBuffer();
	applyViewport();
	glEnable(GL_SCISSOR_TEST);
	glDrawPixels(viewport.w, viewport.h, GL_RGB, GL_FLOAT, data);

	delete[] data;
}

}
