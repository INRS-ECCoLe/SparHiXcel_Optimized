#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
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
    return 2.0 * std::max(std::abs(max_val), std::abs(min_val)) / 255.0;
}

// --- Helper: Exact Python quantize logic ---
inline int quantize_val(double x, double s) {
    double v = std::round(x / s); 
    if (v > 127.0) return 127;
    if (v < -128.0) return -128;
    return static_cast<int>(v);
}

// --- allocate_mux_tran (100% Exact Parity) ---
void allocate_mux_tran(const std::vector<double>& num_of_col, 
                       const std::vector<double>& remaining_array, 
                       const std::vector<double>& full_array, 
                       std::vector<double>& result_col, 
                       int filter_num, 
                       std::vector<double>& filled_array_prev) {
    
    int rows = (int)remaining_array.size();
    std::vector<double> filled_array(rows);
    bool any_diff = false;

    for (int i = 0; i < rows; ++i) {
        filled_array[i] = full_array[i] - remaining_array[i];
        if (filled_array[i] != filled_array_prev[i]) any_diff = true;
    }

    if (any_diff) {
        bool all_neg_one = true;
        for (double val : result_col) if (val != -1.0) { all_neg_one = false; break; }

        if (all_neg_one) {
            double max_val = 0;
            for (double v : filled_array) if (v > max_val) max_val = v;
            int selected_col = (int)max_val - 1;
            if (selected_col >= 0 && selected_col < (int)result_col.size()) {
                result_col[selected_col] = (double)filter_num + 1.0;
            }
        } else {
            int m_val = 0;
            for (int i = 0; i < rows; ++i) {
                if (num_of_col[i] != 0) {
                    if ((int)filled_array[i] > m_val) m_val = (int)filled_array[i];
                }
            }
            for (int j = std::max(0, m_val - 1); j < (int)result_col.size(); ++j) {
                if (result_col[j] == -1.0) {
                    result_col[j] = (double)filter_num + 1.0;
                    break;
                }
            }
        }
    }
    filled_array_prev = filled_array; 
}

// --- Main Function ---
py::tuple assign_PE_max_output_filter(int N_ROWS_ARRAY, int N_COLS_ARRAY, int max_output_filter, py::array_t<double> f_weights, int max_mux_trans) {
    auto r = f_weights.unchecked<4>();
    int W_H = (int)r.shape(0);
    int C_IN = (int)r.shape(2);
    int C_OUT = (int)r.shape(3);

    double s = get_scale(f_weights);
    double near_zero_threshold = std::pow(2.0, -8);

    int ch_per_slice = std::floor((double)N_ROWS_ARRAY / W_H);
    int n = (C_IN * W_H <= N_ROWS_ARRAY) ? 1 : (int)std::ceil((double)C_IN / (double)ch_per_slice);
    int m = (int)std::ceil((double)C_OUT / (double)max_output_filter);

    // Initial State
    std::vector<double> remaining_array(N_ROWS_ARRAY, (double)N_COLS_ARRAY);
    std::vector<double> full_array(N_ROWS_ARRAY, (double)N_COLS_ARRAY);
    std::vector<double> filled_array_prev(N_ROWS_ARRAY, 0.0);
    std::vector<double> result_col(N_COLS_ARRAY + 1, -1.0); 
    
    // Tracking Variables
    std::vector<std::vector<double>> PE_filled_snapshots;
    std::vector<py::dict> PE_assignments;
    std::vector<int> current_pe_filters;
    std::vector<int> current_pe_channels;
    int count = 0;

    for (int filter_slices = 0; filter_slices < m; ++filter_slices) {
        for (int ch_slices_num = 0; ch_slices_num < n; ++ch_slices_num) {

            int f_start = filter_slices * max_output_filter;
            int f_end = std::min((filter_slices + 1) * max_output_filter, C_OUT);

            for (int filter_num = f_start; filter_num < f_end; ++filter_num) {
                int ch_start = ch_slices_num * ch_per_slice;
                int ch_end = std::min((ch_slices_num + 1) * ch_per_slice, C_IN);

                std::vector<double> num_of_col(N_ROWS_ARRAY, 0.0);
                bool any_nz_in_filter = false;

                // num_of_col calculation
                for (int ch_num = ch_start; ch_num < ch_end; ++ch_num) {
                    int max_nz = 0;
                    for (int row = 0; row < W_H; ++row) {
                        int row_nz = 0;
                        for (int col = 0; col < (int)r.shape(1); ++col) {
                            double val = r(row, col, ch_num, filter_num);
                            if (std::abs(quantize_val(val, s)) > 0 && std::abs(val) > near_zero_threshold) row_nz++;
                        }
                        if (row_nz > max_nz) max_nz = row_nz;
                    }
                    if (max_nz > 0) any_nz_in_filter = true;
                    int offset = W_H * (ch_num % ch_per_slice);
                    for (int i = 0; i < W_H; ++i) if (offset + i < N_ROWS_ARRAY) num_of_col[offset + i] = (double)max_nz;
                }

                if (any_nz_in_filter) {
                    std::vector<double> fill_array(N_ROWS_ARRAY);
                    for (int i = 0; i < N_ROWS_ARRAY; ++i) fill_array[i] = full_array[i] - remaining_array[i] + num_of_col[i];

                    // select_col calculation
                    int select_col = -1;
                    bool all_neg_one = true;
                    for (double val : result_col) if (val != -1.0) { all_neg_one = false; break; }

                    if (all_neg_one) {
                        double max_f = 0;
                        for (double v : fill_array) if (v > max_f) max_f = v;
                        select_col = (int)max_f - 1;
                    } else {
                        int m_search = 0;
                        for (int i = 0; i < N_ROWS_ARRAY; ++i) if (num_of_col[i] != 0) if ((int)fill_array[i] > m_search) m_search = (int)fill_array[i];
                        for (int j = std::max(0, m_search - 1); j < (int)result_col.size(); ++j) {
                            if (result_col[j] == -1.0) { select_col = j; break; }
                        }
                    }

                    // Placement Check
                    bool basic_fit = true;
                    for (int i = 0; i < N_ROWS_ARRAY; ++i) if (remaining_array[i] - num_of_col[i] < 0) basic_fit = false;

                    if (basic_fit && select_col < N_COLS_ARRAY) {
                        // Penalty Logic
                        std::vector<double> num_of_col_temp(N_ROWS_ARRAY, 0.0);
                        for (int c_n = ch_start; c_n < ch_end; ++c_n) {
                            int off = W_H * (c_n % ch_per_slice);
                            if (num_of_col[off] != 0) {
                                int td = select_col - (int)fill_array[off] + 1;
                                if (td > max_mux_trans) {
                                    for (int i = 0; i < W_H; ++i) num_of_col_temp[off + i] = (double)(td - max_mux_trans);
                                }
                            }
                        }

                        bool fit_penalty = true;
                        for (int i = 0; i < N_ROWS_ARRAY; ++i) if (remaining_array[i] - num_of_col[i] - num_of_col_temp[i] < 0) fit_penalty = false;

                        if (fit_penalty) {
                            for (int i = 0; i < N_ROWS_ARRAY; ++i) remaining_array[i] -= (num_of_col[i] + num_of_col_temp[i]);

                            // SUCCESSFUL ASSIGNMENT - TRACKING
                            if (std::find(current_pe_filters.begin(), current_pe_filters.end(), filter_num) == current_pe_filters.end())
                                current_pe_filters.push_back(filter_num);
                            
                            // Capture the channel slice range exactly like your Python
                            for (int c = ch_start; c < ch_end; ++c) {
                                if (std::find(current_pe_channels.begin(), current_pe_channels.end(), c) == current_pe_channels.end())
                                    current_pe_channels.push_back(c);
                            }

                            allocate_mux_tran(num_of_col, remaining_array, full_array, result_col, filter_num, filled_array_prev);
                        } else {
                            // CASE: Penalty Full
                            PE_filled_snapshots.push_back(filled_array_prev);
                            py::dict d; d["filters"] = current_pe_filters; d["channels"] = current_pe_channels;
                            PE_assignments.push_back(d);

                            count++;
                            std::fill(remaining_array.begin(), remaining_array.end(), (double)N_COLS_ARRAY);
                            std::fill(result_col.begin(), result_col.end(), -1.0);
                            std::fill(filled_array_prev.begin(), filled_array_prev.end(), 0.0);

                            current_pe_filters = {filter_num};
                            current_pe_channels = {ch_slices_num * ch_per_slice}; // Carry over failure index
                            for (int c = ch_start; c < ch_end; ++c) {
                                if (std::find(current_pe_channels.begin(), current_pe_channels.end(), c) == current_pe_channels.end())
                                    current_pe_channels.push_back(c);
                            }

                            for (int i = 0; i < N_ROWS_ARRAY; ++i) remaining_array[i] -= num_of_col[i];
                            allocate_mux_tran(num_of_col, remaining_array, full_array, result_col, filter_num, filled_array_prev);
                        }
                    } else {
                        // CASE: Basic Fit Full
                        PE_filled_snapshots.push_back(filled_array_prev);
                        py::dict d; d["filters"] = current_pe_filters; d["channels"] = current_pe_channels;
                        PE_assignments.push_back(d);

                        count++;
                        std::fill(remaining_array.begin(), remaining_array.end(), (double)N_COLS_ARRAY);
                        std::fill(result_col.begin(), result_col.end(), -1.0);
                        std::fill(filled_array_prev.begin(), filled_array_prev.end(), 0.0);

                        current_pe_filters = {filter_num};
                        current_pe_channels = {ch_slices_num * ch_per_slice};
                        for (int c = ch_start; c < ch_end; ++c) {
                                if (std::find(current_pe_channels.begin(), current_pe_channels.end(), c) == current_pe_channels.end())
                                    current_pe_channels.push_back(c);
                            }

                        for (int i = 0; i < N_ROWS_ARRAY; ++i) remaining_array[i] -= num_of_col[i];
                        allocate_mux_tran(num_of_col, remaining_array, full_array, result_col, filter_num, filled_array_prev);
                    }
                }
            }
            // End of channel slice - final save
            PE_filled_snapshots.push_back(filled_array_prev);
            py::dict d; d["filters"] = current_pe_filters; d["channels"] = current_pe_channels;
            PE_assignments.push_back(d);

            count++;
            std::fill(remaining_array.begin(), remaining_array.end(), (double)N_COLS_ARRAY);
            std::fill(result_col.begin(), result_col.end(), -1.0);
            std::fill(filled_array_prev.begin(), filled_array_prev.end(), 0.0);
            current_pe_filters.clear();
            current_pe_channels.clear();
        }
    }

    double denom = std::ceil((double)C_OUT / int(N_COLS_ARRAY / (double)r.shape(1))) * std::ceil((double)C_IN / int (N_ROWS_ARRAY / (double)W_H));
    double utilization = 100.0 * (1.0 - (double)count / denom);
    
    return py::make_tuple(PE_filled_snapshots, PE_assignments, utilization);
}

PYBIND11_MODULE(assign_PE_max_output_filter_cpp, m) {
    m.def("assign_PE_max_output_filter", &assign_PE_max_output_filter);
}