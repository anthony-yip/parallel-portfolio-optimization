//
// Created by anthony on 6/28/24.
//
#pragma once
#include "constants.h"
#include <thrust/device_vector.h>

class MVOThrust {
public:
    MVOThrust(TimeSeriesStockData data);
    Portfolio solve();
private:
    thrust::device_vector<float> m_historical_data;
    thrust::device_vector<float> m_covariance_matrix;
    thrust::device_vector<float> m_expected_returns;
    size_t m_num_timesteps;

    void estimate_expected_returns(); // or thrust::device_vector<float> estimate_expected_returns()???;
    void calculate_sample_covariance_matrix();
    void shrink_covariance_matrix(float shrinkage_factor);

};


