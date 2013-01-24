/*
 * Exceptions 
 * 
 * Author: Carlos Duelo Serrano
 */

#ifndef _EXECPTIONS_H_
#define _EXECPTIONS_H_

#include <exception>

class exceptionGeneric : public std::exception
{
	virtual const char* what() const throw()
	{
		return "Errors during executing";
	}

} excepGen;

class exceptionFile : public std::exception
{
	virtual const char* what() const throw()
	{
		return "File not found";
	}

} excepFileNotFound;

class exceptionHDF5DataSet : public std::exception
{
	virtual const char* what() const throw()
	{
		return "Data set not found";
	}

} excepHDF5DataSet;

class exceptionHDF5Read: public std::exception
{
	virtual const char* what() const throw()
	{
		return "Exception: reading hdf5 file";
	}

} excepHDF5READ;

#endif
