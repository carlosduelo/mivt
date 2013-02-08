#include "threadMaster.hpp"
#include "FreeImage.h"
#include <exception>
#include <iostream>
#include <fstream>
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glut.h>

float * screenC;
int W = 1024;
int H = 1024;

threadMaster * mivt;
void display()
{
	mivt->createFrame(screenC);	

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glDrawPixels(W, H, GL_RGB, GL_FLOAT, screenC);

	glutSwapBuffers();
}

void KeyDown(unsigned char key, int x, int y)
{
        switch (key)
        {
//              case 27:                //ESC
//                      PostQuitMessage(0);
//                      break;
                case 'a':
                        mivt->RotateY(5.0);
                        break;
                case 'd':
                        mivt->RotateY(-5.0);
                        break;
                case 'w':
                        mivt->MoveForward( -5.0 ) ;
                        break;
                case 's':
                        mivt->MoveForward( 5.0 ) ;
                        break;
                case 'x':
                        mivt->RotateX(5.0);
                        break;
                case 'y':
                        mivt->RotateX(-5.0);
                        break;
                case 'c':
                        mivt->StrafeRight(-5.0);
                        break;
                case 'v':
                        mivt->StrafeRight(5.0);
                        break;
                case 'f':
                        mivt->MoveUpward(-5.0);
                        break;
                case 'r':
                        mivt->MoveUpward(5.0);
                        break;
                case 'm':
                        mivt->RotateZ(-5.0);
                        break;
                case 'n':
                        mivt->RotateZ(5.0);
                        break;
                case '0':
                        mivt->increaseLevelOctree();
                        break;
                case '9':
                        mivt->decreaseLevelOctree();
                        break;
        }
	display();
}

int main(int argc, char ** argv)
{
	if (argc < 3)
	{
		std::cerr<<"Error, testFileManger file_type file [dataset_name]"<<std::endl;
		return 0;
	}

	int x,y,z;

	std::cout<<"Display Resolution:"<<std::endl;
	std::cout<<"Width: ";
	std::cin >> W;
	std::cout<<"Height: ";
	std::cin >> H;
	std::cout<<"Camera position (X,Y,Z):"<<std::endl;
	std::cout<<"X: ";
	std::cin >> x;
	std::cout<<"Y: ";
	std::cin >> y;
	std::cout<<"Z: ";
	std::cin >> z;

	initParams_masterWorker_t params;

	// Workers
	params.numDevices	= 3;
	params.numWorkers[0]	= 4;
	params.numWorkers[1]	= 4;
	params.numWorkers[2]	= 8;
	params.deviceID[0]	= 0;
	params.deviceID[1]	= 1;
	params.deviceID[2]	= 2;

	// Cache
	params.maxElementsCache[0]	= 2500;
	params.maxElementsCache[1]	= 2500;
	params.maxElementsCache[2]	= 20000;
	params.cubeInc			= 2;
	params.cubeDim			= make_int3(32,32,32);
	params.levelCube		= 4;

	// Octree
	params.maxLevelOctree	= 9;

	// ray caster
	params.rayCasterOptions.ligth_position = make_float3(512.0f, 512.0f, 512.0f);

	// Camera
	params.displayOptions.height		= H;
	params.displayOptions.width		= W;
	params.displayOptions.distance		= 50.0f;
	params.displayOptions.fov_H		= 30.0f;
	params.displayOptions.fov_W		= 30.0f;
	params.displayOptions.numRayPixel	= 1;
	params.displayOptions.tileDim		= make_int2(32,32);
	params.displayOptions.position		= make_float3(x,y,z);

	mivt = new threadMaster(&argv[1], &params);

	std::cerr<<"Allocating pixel buffer: ";
	if (cudaSuccess != cudaMallocHost((void**)&screenC, 3*params.displayOptions.height*params.displayOptions.width*sizeof(float)))
	{
		std::cerr<<"Fail"<<std::endl;
		throw;
	}
	else
		std::cerr<<"Ok"<<std::endl;

	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(W, H);
	glutCreateWindow(argv[1]);

	glutDisplayFunc(display);
	glutKeyboardFunc(KeyDown);

	glEnable(GL_DEPTH_TEST);
	glClearColor(0.0, 0.0, 0.0, 1.0);

	glutMainLoop();
}
