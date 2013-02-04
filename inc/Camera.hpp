/*
 * Camera
 *
 * Author: Carlos Duelo Serrano
 */

#ifndef _CAMERA_H_
#define _CAMERA_H_

#include "cutil_math.h"

#define MAX_ALIASSING 4
#define MAX_RAY_PIXEL MAX_ALIASSING*MAX_ALIASSING

typedef struct
{
	// Display Resolution
	int			height;
	int			width;

	// Camera parameters
	float			distance;
	float			fov_H;
	float			fov_W;

	// initial camera position
	float3			position;

	// Antialiassing supersampling factor
	int			numRayPixel;

	// Tile dimension
	int2			tileDim;
	
} camera_settings_t;

class Camera
{
	protected:
		// Display Resolution
		int			height;
		int			width;

		// Camera parameters
		float			distance;
		float			fov_H;
		float			fov_W;

		// Virtual screen dimension
		float			height_screen;
		float			width_screen;

		// Camera matrix
		float3			look;
		float3			up;
		float3			right;

		float                  	RotatedX;
		float                  	RotatedY;
		float                  	RotatedZ;

		float3			position;
		float3			startPosition;

		// Antialiassing supersampling factor
		int			numRayPixel;

		// Tile dimension
		int2			tileDim;
	public:
		Camera(camera_settings_t * settings);

		void	setNewDisplay(camera_settings_t * settings);

		void	increaseSampling();

		void	decreaseSampling();

		void	resetCameraPosition();

		int	getHeight();

		int	getWidth();

		float	getHeight_screen();

		float	getWidth_screen();

		int2	getTileDim();

		int	getNumRayPixel();

		int	getMaxRayPixel();

		float3	get_look();

		float3	get_up();

		float3	get_right();

		float3	get_position();

		void	Move(float3 Direction);

		void	RotateX(float Angle);

		void	RotateY(float Angle);
		
		void	RotateZ(float Angle);

		void	MoveForward(float Distance);

		void	MoveUpward(float Distance);

		void	StrafeRight(float Distance);	
};
#endif/*_CAMERA_H_*/
