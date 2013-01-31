/*
 * rayCaster 
 *
 * Author: Carlos Duelo Serrano
 */
#ifndef _RAY_CASTER_H_
#define _RAY_CASTER_H_

#include "config.hpp"

typedef struct
{
	float3 	ligth_position;
	float	isosurface;
} rayCaster_options_t;

class rayCaster
{
	private:
		float 			iso;

		// Lighting
		float3			lightPosition;
		// Material parameters

		// rayCasing Parameters
		float step;
	public:
		rayCaster(rayCaster_options_t * options);

		~rayCaster();

		void render(float * rays, int numRays, float3 camera_position, int levelO, int levelC, int nLevel, visibleCube_t * cube, int3 cubeDim, int3 cubeInc, float * buffer, cudaStream_t stream);
};

#endif
