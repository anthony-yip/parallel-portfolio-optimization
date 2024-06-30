//
// Created by anthony on 6/28/24.
//
#pragma once
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include "constants.h"

// Functor to calculate the row index
struct RowIndex : public thrust::unary_function<int, int>
{
    int num_cols;
    explicit RowIndex(const int num_cols) : num_cols(num_cols) {}

    __device__ int operator()(int i)
    {
        return i / num_cols;
    }
};

// Functor to add and divide by the number of columns
struct AddAndDivide : public thrust::binary_function<float, float, float>
{
    int num_cols;
    explicit AddAndDivide(const int num_cols) : num_cols(num_cols) {}

    __device__ float operator()(float a, float b)
    {
        return a + b / num_cols;
    }
};

/// To be used to make a transform iterator as the values_input to the reduce by key
struct CovarianceTransform : public thrust::unary_function<int, float>
{
    int num_timesteps;
    thrust::device_vector<float>& historical_data;
    thrust::device_vector<float>& expected_returns;

    CovarianceTransform(int num_timesteps, thrust::device_vector<float>& historical_data, thrust::device_vector<float>& expected_returns)
        : num_timesteps(num_timesteps), historical_data(historical_data), expected_returns(expected_returns) {}

    __device__ float operator()(int i)
    {
        int timestep = i % num_timesteps;
        int stock_idx = i / num_timesteps;
        int stock_i = stock_idx / num_stocks;
        int stock_j = stock_idx % num_stocks;
        return ((historical_data[stock_i * num_timesteps + timestep] - expected_returns[stock_i]) * (historical_data[stock_j * num_timesteps + timestep] - expected_returns[stock_j])) / (num_timesteps - 1);
    }
};

/// Given some index, select the diagonal of the covariance matrix
struct SelectDiagonals : public thrust::unary_function<int, float>
{
    thrust::device_vector<float>& covariance_matrix;

    SelectDiagonals(thrust::device_vector<float>& covariance_matrix)
        : covariance_matrix(covariance_matrix) {}

    __device__ float operator()(int i)
    {
        return covariance_matrix[i * num_stocks + i];
    }
};

struct Shrink : public thrust::unary_function<int, void>
{
    float shrinkage_factor;
    thrust::device_vector<float>& covariance_matrix;
    float mean_variance;

    Shrink(float shrinkage_factor, thrust::device_vector<float>& covariance_matrix, float mean_variance)
        : shrinkage_factor(shrinkage_factor), covariance_matrix(covariance_matrix), mean_variance(mean_variance) {}

    __device__ void operator()(int i)
    {
        int row = i / num_stocks;
        int col = i % num_stocks;
        float shrinkage_target;
        if (row == col)
        {
            shrinkage_target = mean_variance;
        }
        else
        {
            shrinkage_target = 0;
        }
        covariance_matrix[i] = shrinkage_factor * shrinkage_target + (1 - shrinkage_factor) * covariance_matrix[i];
    }
};

struct SubtractRiskFreeRate
{
    float risk_free_rate;

    explicit SubtractRiskFreeRate(float risk_free_rate) : risk_free_rate(risk_free_rate) {}

    __device__
    float operator()(const float& x) const {
        return x - risk_free_rate;
    }
};
