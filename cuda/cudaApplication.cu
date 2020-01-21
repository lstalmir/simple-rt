#include "cudaApplication.h"

namespace RT
{
    namespace CUDA
    {
        __global__
        void ObjectIntersectionKernel(
            const RayData* pRays,
            const ObjectData* pObjects,
            IntersectionData* pIntersection )
        {

        }

        void Application::ObjectIntersection( Array<RayData> rays )
        {
            // Cross each ray with each object
            DispatchParameters params( rays.Size() * m_Scene.Objects.size() );

            Array<IntersectionData> intersection( 1 );

            ObjectIntersectionKernel<<<params.NumBlocksPerGrid, params.NumThreadsPerBlock>>>
                (rays.Device(), m_Scene.Private.ObjectDeviceMemory.Device(), intersection.Device());
        }
    }
}
