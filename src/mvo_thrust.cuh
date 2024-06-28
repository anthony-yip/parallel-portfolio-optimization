//
// Created by anthony on 6/28/24.
//
#pragma once
#include "constants.h"
#include <thrust/device_vector.h>

class MVOThrust {
public:
    MVOThrust(TimeSeriesStockData data);
    static Portfolio solve();
private:
    m_historical_data;
};


