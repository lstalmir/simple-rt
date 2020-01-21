#include "cudaRay.h"

namespace RT
{
    namespace CUDA
    {
        Ray::Ray( const Array<RayData>& array, int index )
            : DataWrapper( array, index )
        {
        }

        SecondaryRay::SecondaryRay( const Array<SecondaryRayData>& array, int index )
            : DataWrapper( array, index )
        {
        }
    }
}
