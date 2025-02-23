#include "Utils.h"

void Scale(vector<double>& a_Values, const double a_tr_min, const double a_tr_max)
{
    double t_max = std::numeric_limits<double>::min(), t_min = std::numeric_limits<double>::max();
    GetMaxMin(a_Values, t_min, t_max);
    vector<double> t_ValuesScaled;
    for(vector<double>::const_iterator t_It = a_Values.begin(); t_It != a_Values.end(); ++t_It)
    {
        double t_ValueToBeScaled = (*t_It);
        Scale(t_ValueToBeScaled, t_min, t_max, 0, 1); // !!!!!!!!!!!!!!!!??????????
        t_ValuesScaled.push_back(t_ValueToBeScaled);
    }

    a_Values = t_ValuesScaled;
}



