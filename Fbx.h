#pragma once
#include <fbxsdk.h>
#include <memory>

namespace RT
{
    template<typename FbxType>
    struct FbxDeleter
    {
        inline void operator()( FbxType* object ) { object->Destroy(); }
    };
    
    template<typename FbxType>
    using FbxHandle = std::unique_ptr<FbxType, FbxDeleter<FbxType>>;

    template<typename FbxType, typename... ArgumentTypes>
    inline FbxHandle<FbxType> FbxCreate( ArgumentTypes... arguments )
    {
        return FbxHandle<FbxType>( FbxType::Create( arguments... ) );
    }
}
