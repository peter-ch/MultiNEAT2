#include "Utils.h"

void Scale(vector<double>& a_Values, const double a_tr_min, const double a_tr_max)
{
    if (a_Values.empty())
        return;

    double t_max = 0.0;
    double t_min = 0.0;
    GetMaxMin(a_Values, t_min, t_max);
    vector<double> t_ValuesScaled;
    t_ValuesScaled.reserve(a_Values.size());
    for(vector<double>::const_iterator t_It = a_Values.begin(); t_It != a_Values.end(); ++t_It)
    {
        double t_ValueToBeScaled = (*t_It);
        Scale(t_ValueToBeScaled, t_min, t_max, a_tr_min, a_tr_max);
        t_ValuesScaled.push_back(t_ValueToBeScaled);
    }

    a_Values = t_ValuesScaled;
}



