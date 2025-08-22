#include <stdexcept>
#include <iostream>
#include <chrono>
#include <sstream>
#include <iomanip>


class DotProgressBar {

public:
    // Constructor
    DotProgressBar(const long unsigned int &total_iter,
                   const std::string &unit = "iteration(s)",
                   const unsigned int &refresh_rate = 1,
                   const unsigned int &bar_length = 100) {
        if (total_iter == 0) {
            throw std::invalid_argument("Total number of iterations cannot be 0!");
        }

        _total_iter = total_iter;
        _curr_iter = 0;
        _start_time = std::chrono::system_clock::now();
        _curr_time = _start_time;
        _unit = unit;
        _refresh_rate = refresh_rate;
        _bar_length = bar_length;

        update_bar(); // print progress bar
    }


    std::string h_min_s_duration(const int &time) {
        // copy time in seconds
        int seconds = time;
        // extract hours, minutes, and seconds
        int hours = seconds /3600; // hours
        seconds -= hours *3600;
        int minutes = seconds /60; // minutes
        seconds -= minutes *60;

        std::ostringstream oss;
        oss << std::setfill('0')
            << std::setw(2) << hours << ":"
            << std::setw(2) << minutes << ":"
            << std::setw(2) << seconds;

        return oss.str();
    }


    void update_bar() {
        // calculate progress and length of the progress bar
        float progress = static_cast<float>(_curr_iter) / _total_iter;
        unsigned int length = std::round(static_cast<float>(_curr_iter) / _total_iter * _bar_length);
        // calculate ellapsed and remaining time
        int elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(_curr_time - _start_time).count();
        int remaining_time;
        if (_curr_iter == 0) {remaining_time = -3661000;}
        else {remaining_time = elapsed_time * static_cast<float>(_total_iter - _curr_iter) / _curr_iter;}
        // format times
        std::string ellapsed_h_min_s = h_min_s_duration(elapsed_time /1000);
        std::string remaining_h_min_s = h_min_s_duration(remaining_time /1000);

        // print the progress bar
        std::ostringstream oss;
        oss << "\r\033[2K"; // go to beginning of the line and clear it
        oss << std::setw(std::to_string(_total_iter).size()) << _curr_iter // print progress info
            << " / " << _total_iter << " " << _unit
            << " (" << std::setw(5) << std::fixed << std::setprecision(1) << progress *100 << " %)";
        oss << " : "; // open progress bar
        for (unsigned int i = 0; i < length; ++i) {
            oss << "\u25CF"; // completed part; full dots
        }
        for (unsigned int i = 0; i < _bar_length - length; ++i) {
            oss << "\u25CB"; // remaining part; empty dots
        }
        oss << " : "; // close progress bar
        oss << ellapsed_h_min_s // print time info
            << " | "
            << remaining_h_min_s
            << " ";

        std::cout << oss.str() << std::flush; // flush the output to ensure it's displayed immediately
    }


    void step() {
        _curr_iter += 1;
        if (_curr_iter > _total_iter) {
            std::cout << "Iteration has already finished!" << std::endl;
        }
        else if (_curr_iter == _total_iter) {
            _curr_time = std::chrono::system_clock::now();
            update_bar();
            std::cout << std::endl;
        }
        else if (_curr_iter % _refresh_rate == 0) {
            _curr_time = std::chrono::system_clock::now();
            update_bar();
        }
    }


private:
    long unsigned int _total_iter;
    long unsigned int _curr_iter;
    std::chrono::system_clock::time_point _start_time;
    std::chrono::system_clock::time_point _curr_time;
    std::string _unit;
    unsigned int _refresh_rate;
    unsigned int _bar_length;
};


