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
Hola Carlos,
Dentro de Channel::frameDraw puedes invocar getPixelViewport, que te devuelve un eq::PixelViewport con la esquina inferior izquierda del viewport, el ancho y el alto en píxels.
Aparte, para no tener que llamar a glDrawBuffer ni glViewport ni glScissor por tu cuenta, puedes invocar applyBuffer() y applyViewport() (GL_SCISSOR_TEST sí hay que habilitarlo a mano) y a continuación hacer glDrawPixels.
Eso debería funcionarte sin problemas. 

*/	

 	eq::PixelViewport  viewport = getPixelViewport();


	std::cout<<".............>"<<viewport.x<<" "<<viewport.y<<std::endl;


	float * data = new float [3*viewport.h*viewport.w];
	for(int i=0; i<3*viewport.w*viewport.h; i+=3)
		data[i] = 1.0f;
	
	applyBuffer();
	applyViewport();
	glEnable(GL_SCISSOR_TEST);
	glDrawPixels(viewport.w, viewport.h, GL_RGB, GL_FLOAT, data);

	delete[] data;
}

}
