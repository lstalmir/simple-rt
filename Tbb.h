#pragma once

// Do not link tbb.lib implicitly
#pragma push_macro( "__TBB_NO_IMPLICIT_LINKAGE" )
//#define __TBB_NO_IMPLICIT_LINKAGE 1

// All TBB includes go here
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_vector.h>

#pragma pop_macro( "__TBB_NO_IMPLICIT_LINKAGE" )
