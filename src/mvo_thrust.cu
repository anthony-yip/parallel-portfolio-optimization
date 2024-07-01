//
// Created by anthony on 6/28/24.
//

#include "mvo_thrust.cuh"
#include "functors.cuh"
#include <thrust/iterator/discard_iterator.h>
#include <vector>

#include "matsol_thrust.cuh"

// array of vectors
MVOThrust::MVOThrust(TimeSeriesStockData data) {
    std::vector<float> single_vector;
    // stock1, then stock2, then stock3, etc.
    // this order will be good locality during the reduction
    for (int i = 0; i < num_stocks; i++)
    {
        single_vector.insert(single_vector.end(), data[i].begin(), data[i].end());
    }
    m_historical_data = thrust::device_vector<float>(single_vector);
    m_num_timesteps = data[0].size();
}

void MVOThrust::estimate_expected_returns() {
    thrust::counting_iterator<int> it(0);
    auto keys = thrust::make_transform_iterator(it, RowIndex(m_num_timesteps));

    thrust::reduce_by_key(keys, keys + (num_stocks * m_num_timesteps), m_historical_data.begin(),
        thrust::make_discard_iterator(), m_expected_returns.begin(), thrust::equal_to<int>(),
        AddAndDivide(num_stocks));
}

/**
 * The thought process when designing this function: We need to eventually reduce by key, which is somewhat
 * restrictive. The only things we can change then are the keys (which is standard - see the RowIndex functor)
 * and the values, through a transform iterator. Hence the CovarianceTransform functor.
 */
void MVOThrust::calculate_sample_covariance_matrix()
{
    // Create a counting iterator
    thrust::counting_iterator<int> it(0);
    // Create the key and value iterators
    auto keys = thrust::make_transform_iterator(it, RowIndex(m_num_timesteps));
    auto values = thrust::make_transform_iterator(it, CovarianceTransform(m_num_timesteps, m_historical_data, m_expected_returns));

    thrust::reduce_by_key(keys, keys + (m_num_timesteps * num_stocks * num_stocks), values,
        thrust::make_discard_iterator(), m_covariance_matrix.begin());
}

void MVOThrust::shrink_covariance_matrix(float shrinkage_factor)
{
    auto diagonal_it = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), SelectDiagonals(m_covariance_matrix));
    // this can also be a trasnform reduce
    float mean_variance = thrust::reduce(diagonal_it, diagonal_it + num_stocks) / num_stocks;
    thrust::counting_iterator<int> it(0);
    thrust::for_each(it, it + num_stocks * num_stocks, Shrink {shrinkage_factor, m_covariance_matrix, mean_variance});
}

Portfolio MVOThrust::solve()
{
    estimate_expected_returns();
    calculate_sample_covariance_matrix();
    shrink_covariance_matrix(shrinkage_factor);
    thrust::transform(m_expected_returns.begin(), m_expected_returns.end(), m_expected_returns.begin(), SubtractRiskFreeRate(risk_free_rate));
    thrust::device_vector<float> portfolio_weights(num_stocks);
    cuMatSol<num_stocks>::solve(m_covariance_matrix, m_expected_returns, portfolio_weights);
    // std::vector<float> result(num_stocks);
    Portfolio result;
    thrust::copy(portfolio_weights.begin(), portfolio_weights.end(), result.begin());
    return result;
}

