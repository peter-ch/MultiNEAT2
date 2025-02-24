#ifndef _UTILS_H
#define _UTILS_H

#include <stdlib.h>
#include <math.h>
#include <sstream>
#include <string>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include "Assert.h"
#include "Random.h"

using namespace std;

inline void GetMaxMin(const vector<double>& a_Vals, double& a_Min, double& a_Max)
{
    if(a_Vals.empty()){
        a_Min = 0;
        a_Max = 0;
        return;
    }
    auto result = std::minmax_element(a_Vals.begin(), a_Vals.end());
    a_Min = *result.first;
    a_Max = *result.second;
}

inline std::string itos(const int a_Arg)
{
    std::ostringstream t_Buffer;
    t_Buffer << a_Arg;
    return t_Buffer.str();
}

inline std::string ftos(const double a_Arg)
{
    std::ostringstream t_Buffer;
    t_Buffer << a_Arg;
    return t_Buffer.str();
}

inline void Clamp(double &a_Arg, const double a_Min, const double a_Max)
{
    ASSERT(a_Min <= a_Max);
    if(a_Arg < a_Min)
    {
        a_Arg = a_Min;
        return;
    }
    if(a_Arg > a_Max)
    {
        a_Arg = a_Max;
        return;
    }
}

inline void Clamp(float &a_Arg, const float a_Min, const float a_Max)
{
    ASSERT(a_Min <= a_Max);
    if(a_Arg < a_Min)
    {
        a_Arg = a_Min;
        return;
    }
    if(a_Arg > a_Max)
    {
        a_Arg = a_Max;
        return;
    }
}

inline void Clamp(int &a_Arg, const int a_Min, const int a_Max)
{
    ASSERT(a_Min <= a_Max);
    if(a_Arg < a_Min)
    {
        a_Arg = a_Min;
        return;
    }
    if(a_Arg > a_Max)
    {
        a_Arg = a_Max;
        return;
    }
}

inline int Rounded(const double a_Val)
{
    int t_Integral = static_cast<int>(a_Val);
    double t_Mantissa = a_Val - t_Integral;
    return (t_Mantissa < 0.5) ? t_Integral : t_Integral + 1;
}

inline int RoundUnderOffset(const double a_Val, const double a_Offset)
{
    int t_Integral = static_cast<int>(a_Val);
    double t_Mantissa = a_Val - t_Integral;
    return (t_Mantissa < a_Offset) ? t_Integral : t_Integral + 1;
}

inline void Scale(double& a,
                  const double a_min,
                  const double a_max,
                  const double a_tr_min,
                  const double a_tr_max)
{
    if (fabs(a_max - a_min) < std::numeric_limits<double>::epsilon())
    {
        a = (a_tr_min + a_tr_max) / 2.0;
        return;
    }
    double t_a_r = a_max - a_min;
    double t_r = a_tr_max - a_tr_min;
    double rel_a = (a - a_min) / t_a_r;
    a = a_tr_min + t_r * rel_a;
}

inline void Scale(float& a,
                  const double a_min,
                  const double a_max,
                  const double a_tr_min,
                  const double a_tr_max)
{
    if (fabs(a_max - a_min) < std::numeric_limits<double>::epsilon())
    {
        a = static_cast<float>((a_tr_min + a_tr_max) / 2.0);
        return;
    }
    double t_a_r = a_max - a_min;
    double t_r = a_tr_max - a_tr_min;
    double rel_a = (a - a_min) / t_a_r;
    a = static_cast<float>(a_tr_min + t_r * rel_a);
}

inline double Abs(double x)
{
    return (x < 0) ? -x : x;
}

#endif