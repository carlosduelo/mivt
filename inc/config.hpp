#ifndef _CONFIG_H_
#define _CONFIG_H_

/* indentifier type for octree's node */
typedef unsigned long long index_node_t;

#define CUBE		(unsigned char)8
#define PAINTED 	(unsigned char)4
#define CACHED 		(unsigned char)2
#define NOCACHED 	(unsigned char)1
#define NOCUBE		(unsigned char)0

typedef struct
{
	index_node_t 	id;
	float * 	data;
	unsigned char   state;
	index_node_t	cubeID;
} visibleCube_t;

typedef struct
{
	int		id;
	int		deviceID;
	cudaStream_t 	stream;
} threadID_t;

// Octree options
#define UP_LEVEL_OCTREE		0
#define DOWN_LEVEL_OCTREE	1
// RayCasting options
#define INCREASE_STEP		2
#define	DECREASE_STEP		3
// Image
#define CHANGE_ANTIALIASSING	4
#define NEW_TILE		5
#define END			999

typedef struct
{
	int 	work_id;
	int2	tile;
	float * pixel_buffer;
	
} work_packet_t;

#endif
