#include <gtest/gtest.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "../include/angular_part.h"


class AngularPartTest : public ::testing::Test {
protected:
    int nju;
    double* vec;    

    static void SetUpTestSuite() {
        std::cout << "AngularPartTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "AngularPartTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        nju = 3;
        vec = (double*)malloc(sizeof(double) * 3);
        vec[0] = 1;
        vec[1] = 2;
        vec[2] = 3;
    }

    void TearDown() override {
        free(vec);
    }
};


TEST_F(AngularPartTest, get_level) {
    int mju = 2;
    int nju = 2;

    int level = matersdk::mtp::AngularPart<double>::get_level(mju, nju);
    printf("level = %5d\n", level);
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}