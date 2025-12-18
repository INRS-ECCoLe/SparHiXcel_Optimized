#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

// --- Helper: Exact Python scale logic ---
double get_scale(const py::array_t<double>& x) {
    auto r = x.unchecked<4>();
    double max_val = -1e18;
    double min_val = 1e18;
    for (int i = 0; i < r.shape(0); ++i)
        for (int j = 0; j < r.shape(1); ++j)
            for (int k = 0; k < r.shape(2); ++k)
                for (int l = 0; l < r.shape(3); ++l) {
                    if (r(i,j,k,l) > max_val) max_val = r(i,j,k,l);
                    if (r(i,j,k,l) < min_val) min_val = r(i,j,k,l);
                }
    // s = 2 * max(|max|, |min|) / (2^8 - 1)
    return 2.0 * std::max(std::abs(max_val), std::abs(min_val)) / 255.0;
}

// --- Helper: Exact Python quantize logic ---
inline int quantize_val(double x, double s) {
    double v = std::round(x / s); // z=0
    if (v > 127.0) return 127;
    if (v < -128.0) return -128;
    return static_cast<int>(v);
}

double assign_PE_max_output_filter(int N_ROWS_ARRAY, int N_COLS_ARRAY, int max_output_filter, py::array_t<double> f_weights, int max_mux_trans) {
    auto r = f_weights.unchecked<4>();
    int W_H = (int)r.shape(0);
    int W_W = (int)r.shape(1);
    int C_IN = (int)r.shape(2);
    int C_OUT = (int)r.shape(3);

    double s = get_scale(f_weights);
    double near_zero_threshold = std::pow(2.0, -8);

    // Slicing logic
    int ch_per_slice = std::floor((double)N_ROWS_ARRAY / W_H);
    int n = (C_IN * W_H <= N_ROWS_ARRAY) ? 1 : (int)std::ceil((double)C_IN / (double)ch_per_slice);
    int m = (int)std::ceil((double)C_OUT / (double)max_output_filter);

    // State Variables (Equivalent to Python's arrays)
    std::vector<double> remaining_array(N_ROWS_ARRAY, (double)N_COLS_ARRAY);
    std::vector<double> full_array(N_ROWS_ARRAY, (double)N_COLS_ARRAY);
    std::vector<double> result_col(N_COLS_ARRAY, 0.0);
    
    int count = 0;

    for (int filter_slices = 0; filter_slices < m; ++filter_slices) {
        for (int ch_slices_num = 0; ch_slices_num < n; ++ch_slices_num) {

            int f_start = filter_slices * max_output_filter;
            int f_end = std::min((filter_slices + 1) * max_output_filter, C_OUT);

            for (int filter_num = f_start; filter_num < f_end; ++filter_num) {
                
                int ch_start = ch_slices_num * ch_per_slice;
                int ch_end = std::min((ch_slices_num + 1) * ch_per_slice, C_IN);

                // num_of_col calculation (filter_assign)
                std::vector<double> num_of_col(N_ROWS_ARRAY, 0.0);
                for (int ch_num = ch_start; ch_num < ch_end; ++ch_num) {
                    int max_nz = 0;
                    for (int row = 0; row < W_H; ++row) {
                        int nz = 0;
                        for (int col = 0; col < W_W; ++col) {
                            double val = r(row, col, ch_num, filter_num);
                            // Match: abs(quantize(x)) > threshold
                            if (std::abs(quantize_val(val, s)) > 0 && std::abs(val) > near_zero_threshold) nz++;
                        }
                        if (nz > max_nz) max_nz = nz;
                    }
                    int offset = W_H * (ch_num % ch_per_slice);
                    for (int i = 0; i < W_H; ++i) num_of_col[offset + i] = (double)max_nz;
                }

                // select_col logic
                std::vector<double> fill_array(N_ROWS_ARRAY);
                double max_fill = 0;
                for (int i = 0; i < N_ROWS_ARRAY; ++i) {
                    fill_array[i] = full_array[i] - remaining_array[i] + num_of_col[i];
                    if (fill_array[i] > max_fill) max_fill = fill_array[i];
                }

                int last_nz_idx = -1;
                for (int c = 0; c < N_COLS_ARRAY; ++c) if (result_col[c] != 0) last_nz_idx = c;

                int select_col;
                if (last_nz_idx == -1) select_col = (int)max_fill - 1;
                else if (last_nz_idx < (int)max_fill - 1) select_col = (int)max_fill - 1;
                else select_col = last_nz_idx + 1;

                // Check basic fit
                bool basic_fit = true;
                for (int i = 0; i < N_ROWS_ARRAY; ++i) if (remaining_array[i] - num_of_col[i] < 0) basic_fit = false;

                if (basic_fit && select_col < N_COLS_ARRAY) {
                    // Mux transfer penalty calculation
                    std::vector<double> num_of_col_temp(N_ROWS_ARRAY, 0.0);
                    for (int ch_num = ch_start; ch_num < ch_end; ++ch_num) {
                        int offset = W_H * (ch_num % ch_per_slice);
                        int td = select_col - (int)fill_array[offset] + 1;
                        if (td > max_mux_trans) {
                            for (int i = 0; i < W_H; ++i) num_of_col_temp[offset + i] = (double)(td - max_mux_trans);
                        }
                    }

                    // Check fit with penalty
                    bool fit_penalty = true;
                    for (int i = 0; i < N_ROWS_ARRAY; ++i) 
                        if (remaining_array[i] - num_of_col[i] - num_of_col_temp[i] < 0) fit_penalty = false;

                    if (fit_penalty) {
                        // Apply placement
                        for (int i = 0; i < N_ROWS_ARRAY; ++i) {
                            remaining_array[i] = remaining_array[i] - num_of_col[i] - num_of_col_temp[i];
                        }
                        // Update result_col tracking (mirrors result_col logic)
                        if (select_col >= 0 && select_col < N_COLS_ARRAY) result_col[select_col] = 1;
                    } else {
                        // Failure: increment count and reset
                        count++;
                        std::fill(remaining_array.begin(), remaining_array.end(), (double)N_COLS_ARRAY);
                        std::fill(result_col.begin(), result_col.end(), 0.0);
                        for (int i = 0; i < N_ROWS_ARRAY; ++i) remaining_array[i] -= num_of_col[i];
                        if (select_col >= 0 && select_col < N_COLS_ARRAY) result_col[select_col] = 1;
                    }
                } else {
                    count++;
                    std::fill(remaining_array.begin(), remaining_array.end(), (double)N_COLS_ARRAY);
                    std::fill(result_col.begin(), result_col.end(), 0.0);
                    for (int i = 0; i < N_ROWS_ARRAY; ++i) remaining_array[i] -= num_of_col[i];
                    // After reset, the first column of the new cycle is always based on num_of_col
                    // select_col is recalculated effectively
                }
            }
            // End of channel/filter slice
            count++;
            std::fill(remaining_array.begin(), remaining_array.end(), (double)N_COLS_ARRAY);
            std::fill(result_col.begin(), result_col.end(), 0.0);
        }
    }

    double denom = std::ceil((double)C_OUT / (N_COLS_ARRAY / (double)W_W)) * std::ceil((double)C_IN / (N_ROWS_ARRAY / (double)W_H));
    return 100.0 * (1.0 - (double)count / denom);
}

PYBIND11_MODULE(assign_PE_max_output_filter_cpp, m) {
    m.def("assign_PE_max_output_filter", &assign_PE_max_output_filter);
}