/*
 * rayCaster 
 *
 * Author: Carlos Duelo Serrano
 */
#ifndef _RAY_CASTER_H_
#define _RAY_CASTER_H_

#include "config.hpp"

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
		rayCaster(float isosurface, float3 lposition);

		~rayCaster();

		void render(float * rays, int numRays, float3 camera_position, int levelO, int levelC, int nLevel, visibleCube_t * cube, int3 cubeDim, int3 cubeInc, float * buffer, cudaStream_t stream);
};

#endif
