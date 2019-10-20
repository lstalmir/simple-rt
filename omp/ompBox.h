#pragma once
#include "../Vec.h"
#include <limits>

namespace RT::OMP
{
    struct Box
    {
        float_t Xmin = std::numeric_limits<float_t>::infinity();
        float_t Xmax = -std::numeric_limits<float_t>::infinity();
        float_t Ymin = std::numeric_limits<float_t>::infinity();
        float_t Ymax = -std::numeric_limits<float_t>::infinity();
        float_t Zmin = std::numeric_limits<float_t>::infinity();
        float_t Zmax = -std::numeric_limits<float_t>::infinity();
    };
}
