#pragma once
#include "Application.h"
#include <string>
#include <unordered_map>

namespace RT
{
    class ApplicationTest : public Application
    {
    public:
        ApplicationTest( const CommandLineArguments& cmdargs );

        virtual int Run() override final;

    protected:
        std::unordered_map<std::string, int(ApplicationTest::*)()> m_pTests;

        int TEST_Vector_Normalize4();
        int TEST_Vector_Normalize3();
        int TEST_Vector_Normalize2();
        int TEST_Ray_Intersect_Plane();
    };
}
