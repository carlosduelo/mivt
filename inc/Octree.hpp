/*
 * Octree 
 * 
 * Author: Carlos Duelo Serrano
 */

#ifndef _OCTREE_H_
#define _OCTREE_H_

#include <config.hpp>

class OctreeContainer
{
	private:
		float 	isosurface;
		int3 	realDim;
		int 	dimension;
		int 	nLevels;
		int 	maxLevel;

		index_node_t ** octree;
		index_node_t * 	memoryGPU;
		int	*	sizes;

	public:
		/* Lee el Octree de un fichero */
		OctreeContainer(const char * file_name, int p_maxLevel);

		~OctreeContainer();

		int getnLevels();

		float getIsosurface();

		index_node_t ** getOctree();

		int *		getSizes();
};

class Octree
{
	protected:
		int nLevels;
		int currentLevel;
		int maxLevel;

		index_node_t ** octree;
		index_node_t * 	memoryGPU;
		int	*	sizes;

	public:
		Octree(OctreeContainer * oc, int p_maxLevel)
		{
			maxLevel        = p_maxLevel;
			currentLevel    = p_maxLevel;
			nLevels         = oc->getnLevels();
			octree          = oc->getOctree();
			sizes           = oc->getSizes();
		}

		virtual ~Octree() {};

		void	increaseLevel()
		{
			currentLevel = currentLevel == maxLevel ? maxLevel : currentLevel + 1;
		}

		void	decreaseLevel()
		{
			currentLevel = currentLevel == 1 ? 1 : currentLevel - 1;
		}

		int 	getnLevels()
		{
			return nLevels;
		}

		int	getOctreeLevel()
		{
			return currentLevel;
		}

		virtual void resetState(cudaStream_t stream) = 0;

		/* Dado un rayo devuelve true si el rayo impacta contra el volumen, el primer box del nivel dado contra el que impacta y la distancia entre el origen del rayo y la box */
		virtual bool getBoxIntersected(float3 camera_position, float * rays, int numRays, visibleCube_t * visibleGPU, visibleCube_t * visibleCPU, cudaStream_t stream) = 0;

};

class Octree_completeGPU : public Octree
{
	private:
		int		maxRays;
		// Octree State
		int 	*	GstackActual;
		index_node_t * 	GstackIndex;
		int	*	GstackLevel;
	public:
		Octree_completeGPU(OctreeContainer * oc, int p_maxLevel, int p_maxRays);

		~Octree_completeGPU();

		void resetState(cudaStream_t stream);

		bool getBoxIntersected(float3 camera_position, float * rays, int numRays, visibleCube_t * visibleGPU, visibleCube_t * visibleCPU, cudaStream_t stream);
};

#if 0
class Octree_CPUGPU : public Octree
{
	
};
#endif
#endif
