/*
 * Channel 
 *
 * Author: Carlos Duelo Serrano
 */

#ifndef _CHANNEL_H_
#define _CHANNEL_H_

#include "config.hpp"
#include <lunchbox/mtQueue.h>

class Channel
{
	private:
		lunchbox::MTQueue< work_packet_t > * 	queue;
	public:
		Channel(int p_size)
		{
			queue	= new lunchbox::MTQueue< work_packet_t >(p_size);
		}

		~Channel()
		{
			delete queue;
		}

		void pushBlock(work_packet_t work)
		{
			return queue->push(work);
		}

		bool push(work_packet_t work)
		{
			if (queue->getSize() == queue->getMaxSize())
				return false;
			
			queue->push(work);
			return true;
		}

		work_packet_t pop()
		{
			return queue->pop();
		}
		
};

#endif
