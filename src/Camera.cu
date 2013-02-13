/*
 * Camera
 *
 * Author: Carlos Duelo Serrano
 */

#include "Camera.hpp"
#include "cuda_help.hpp"
#include <exception> 
#include <math.h>
#include <iostream>
#include <fstream>

Camera::Camera(camera_settings_t * settings)
{
	height		= settings->height;
	width   	= settings->width;
	distance 	= settings->distance;
	fov_H 		= settings->fov_H;
	fov_W		= settings->fov_W;
	position	= settings->position;
	numRayPixel	= settings->numRayPixel;
	tileDim		= settings->tileDim;
	startPosition	= position;
	height_screen	= 2*distance*tanf(fov_H*(M_PI/180.0));
	width_screen   	= 2*distance*tanf(fov_W*(M_PI/180.0));

	std::cout << "Screen (" << height << "," << width << ")" << std::endl;
	std::cout << "Distance " << distance << "  fov(" << fov_H << "," << fov_W << ")" << std::endl;
	std::cout << "Tamaño pantalla en mundo (" << height_screen << "," << width_screen << ")" << std::endl;

	if (height < 1 || width < 1)
	{
		std::cerr<<"Camera: display resolution invalid"<<std::endl;
		throw ;
	}
	if (fov_H < 1 || fov_W < 1 || distance < 1)
	{
		std::cerr<<"Camera: invalid camera settings"<<std::endl;
		throw ;
	}

	if (numRayPixel > MAX_ALIASSING || numRayPixel < 1)
	{
		std::cerr<<"Camera: invalid super sampling option"<<std::endl;
		throw ;
	}


	look 	= make_float3(0.0f, 0.0f, -1.0f);
	up 	= make_float3(0.0f, 1.0f, 0.0f);
	right   = cross(look, up);
	RotatedX	= 0.0;
	RotatedY	= 0.0;
	RotatedZ	= 0.0;	
}

void	Camera::setNewDisplay(camera_settings_t * settings)
{
	height		= settings->height;
	width   	= settings->width;
	distance 	= settings->distance;
	fov_H 		= settings->fov_H;
	fov_W		= settings->fov_W;
	position	= settings->position;
	numRayPixel	= settings->numRayPixel;
	tileDim		= settings->tileDim;
	height_screen	= 2*distance*tanf(fov_H*(M_PI/180.0));
	width_screen   	= 2*distance*tanf(fov_W*(M_PI/180.0));

	std::cout << "Screen (" << height << "," << width << ")" << std::endl;
	std::cout << "Distance " << distance << "  fov(" << fov_H << "," << fov_W << ")" << std::endl;
	std::cout << "Tamaño pantalla en mundo (" << height_screen << "," << width_screen << ")" << std::endl;

	if (height < 1 || width < 1)
	{
		std::cerr<<"Camera: display resolution invalid"<<std::endl;
		throw ;
	}
	if (fov_H < 1 || fov_W < 1 || distance < 1)
	{
		std::cerr<<"Camera: invalid camera settings"<<std::endl;
		throw;
	}

	if (numRayPixel > MAX_ALIASSING || numRayPixel < 1)
	{
		std::cerr<<"Camera: invalid super sampling option"<<std::endl;
		throw;
	}
}

void	Camera::increaseSampling()
{
	numRayPixel = numRayPixel == MAX_ALIASSING ? MAX_ALIASSING : numRayPixel + 1;	
}

void	Camera::decreaseSampling()
{
	numRayPixel = numRayPixel == 1 ? 1 : numRayPixel - 1;	
}

void	Camera::resetCameraPosition()
{
	position = startPosition;
}

int	Camera::getHeight(){ return height; }

int	Camera::getWidth(){ return width; }

float	Camera::getHeight_screen(){ return height_screen; } 

float	Camera::getWidth_screen(){ return width_screen; }

float 	Camera::getDistance(){ return distance; } 

int2	Camera::getTileDim(){ return tileDim; }

int	Camera::getNumRayPixel(){ return numRayPixel*numRayPixel; }

int	Camera::getMaxRayPixel(){ return MAX_ALIASSING*MAX_ALIASSING; }

float3	Camera::get_look(){ return look; }

float3	Camera::get_up(){ return up; }

float3	Camera::get_right(){ return right; }

float3	Camera::get_position(){ return position; }

void	Camera::Move(float3 Direction)
{
	position += Direction;
}

void	Camera::RotateX(float Angle)
{
	float  sPI180 = sin(Angle*(M_PI/180.0));
	float  cPI180 = cos(Angle*(M_PI/180.0));

	RotatedX += Angle;

	//Rotate viewdir around the right vector:
	look = look*cPI180 + up*sPI180;
	normalize(look);

	//now compute the new UpVector (by cross product)
	up = (-1.0f) * cross(look,right);
}

void	Camera::RotateY(float Angle)
{
	float  sPI180 = sin(Angle*(M_PI/180.0));
	float  cPI180 = cos(Angle*(M_PI/180.0));

	RotatedY += Angle;

	//Rotate viewdir around the up vector:
	look = look*cPI180 - right*sPI180;
	normalize(look);

	//now compute the new RightVector (by cross product)
	right = cross(look, up);
}

void	Camera::RotateZ(float Angle)
{
	float  sPI180 = sin(Angle*(M_PI/180.0));
	float  cPI180 = cos(Angle*(M_PI/180.0));

	RotatedZ += Angle;

	//Rotate viewdir around the right vector:
	right = right*cPI180 + up*sPI180;
	normalize(right);

	//now compute the new UpVector (by cross product)
	up = (-1.0f) * cross(look,right);
}

void	Camera::MoveForward(float Distance)
{
	position += look*(-Distance);
}

void	Camera::MoveUpward(float Distance)
{
	position += right*Distance;
}

void	Camera::StrafeRight(float Distance)	
{
	position += up*Distance;
}
