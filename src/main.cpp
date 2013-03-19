/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "mivt.h"

#include "config.h"
#include "error.h"
#include "channel.h"
#include "node.h"
#include "pipe.h"

class NodeFactory : public eq::NodeFactory
{
public:
	virtual eq::Config*  createConfig( eq::ServerPtr parent )
	{
		return new eqMivt::Config(parent);
	}
	virtual eq::Channel* createChannel(eq::Window* parent)
	{
		return new eqMivt::Channel(parent);
	}
	virtual eq::Node* createNode(eq::Config* parent)
	{
		return new eqMivt::Node(parent);
	}
	virtual eq::Pipe* createPipe(eq::Node* parent)
	{
		return new eqMivt::Pipe(parent);
	}
#if 0
    virtual eq::Window*  createWindow( eq::Pipe* parent )
        { return new eqMivt::Window( parent ); }
    virtual eq::View* createView( eq::Layout* parent )
        { return new eqMivt::View( parent ); }
#endif
};

int main( const int argc, char** argv )
{
	// 1. Equalizer initialization
	NodeFactory nodeFactory;

        eqMivt::initErrors();

	if( !eq::init(argc, argv, &nodeFactory))
	{
		LBERROR << "Equalizer init failed" << std::endl;
		return EXIT_FAILURE;
	}

	// 2. parse arguments
	eqMivt::InitParams initParams;
	if (!initParams.parseArguments( argc, (const char **) argv ))
	{
		LBERROR<<"Error in parameters"<<std::endl;
		return EXIT_FAILURE;
	}

	// 3. initialization of local client node
	lunchbox::RefPtr< eqMivt::EqMivt > client = new eqMivt::EqMivt(initParams);
	if( !client->initLocal( argc, argv ))
	{
		LBERROR << "Can't init client" << std::endl;
		eq::exit();
		return EXIT_FAILURE;
	}
	// 4. run client
	const int ret = client->run();

	// 5. cleanup and exit
	client->exitLocal();

	LBASSERTINFO( client->getRefCount() == 1, client );
	client = 0;

	eq::exit();
	eqMivt::exitErrors();

	return ret;
}
