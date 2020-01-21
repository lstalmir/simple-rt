#include "cudaObject.h"

namespace RT
{
    namespace CUDA
    {
        Object::Object( const Array<ObjectData>& array, int index )
            : DataWrapper( array, index )
        {
        }
    }
}
