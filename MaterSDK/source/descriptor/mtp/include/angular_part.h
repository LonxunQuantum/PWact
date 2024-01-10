#ifndef MATERSDK_ANGULAR_PART_H
#define MATERSDK_ANGULAR_PART_H


#include <utility>
#include <vector>
#include <algorithm>



namespace matersdk {
namespace mtp {

template <typename CoordType>
class AngularPart {
public:
    AngularPart(int max_level, CoordType* vec);

    static int get_level(int mju, int nju);

    void calc_mjus_njus();

private:
    int max_level;
    CoordType* vec;
    std::vector<std::vector<std::pair<int, int>>> mjus_njus_lst;
};



template <typename CoordType>
AngularPart<CoordType>::AngularPart(int max_level, CoordType* vec) {
    this->max_level = max_level;
    this->vec = (CoordType*)malloc(sizeof(CoordType) * 3);
}


template <typename CoordType>
int AngularPart<CoordType>::get_level(int mju, int nju) {
    return (2 + 4*mju + nju);
}


/**
 * @brief 
 * 
 * @example 
 *  The MTP of 6-th level includes five basis functions:
 *      1. B_1 = M_{0, 0};      levB1 = 2 <= levmax = 6
 *      2. B_2 = M_{1, 0};      levB2 = 6 <= levmax = 6
 *      3. B_3 = M_{0, 0}^2;    levB3 = 4 <= levmax = 6
 *      4. B_4 = M_{0, 1} \cdot M_{0, 1};   levB4 = 6 <= levmax = 6
 *      5. B_5 = M_{0, 0}^3;    levB5 = 6 <= levmax = 6
 *  Then,
 *      N_{lin} = 5;
 *      N_{radial} = 2;
 * @tparam CoordType 
 */
template <typename CoordType>
void AngularPart<CoordType>::calc_mjus_njus() {
    
}


}   // namespace : mtp
}   // namespace : matersdk


#endif