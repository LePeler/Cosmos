#include <stdexcept>
#include <iostream>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <vector>
#include <functional>
#include <cmath>


#ifndef MONITOR_H_INCLUDED
#define MONITOR_H_INCLUDED

class Monitor {

public:
    // Constructor
    Monitor(std::vector<std::string> names,
            std::function<std::vector<double>()> get_quantities,
            std::vector<std::function<bool(double)>> conditions,
            size_t check_interval,
            std::string unit = "iter")
        :
        _names(names),
        _quantities(names.size(), NAN),
        _get_quantities(get_quantities),
        _conditions(conditions),
        _curr_iter(0),
        _start_time(std::chrono::system_clock::now()),
        _curr_time(_start_time),
        _check_interval(check_interval),
        _unit(unit),
        _conditions_satisfied(names.size(), false),
        _finished(false)
    {
        draw_monitor(); // draw monitor
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


    void draw_monitor() {
        // calculate ellapsed and remaining time
        int elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(_curr_time - _start_time).count();
        // format times
        std::string ellapsed_h_min_s = h_min_s_duration(elapsed_time /1000);

        // print the monitor
        std::ostringstream oss;
        oss << "\r\033[2K"; // go to beginning of the line and clear it
        oss << _unit << " " << _curr_iter << " | ";
        for (size_t j = 0; j < _names.size(); j++) {
            oss << _names[j] << "=" << _quantities[j] << " (";
            if (_conditions_satisfied[j]) {
                oss << "\u2713";
            }
            else {
                oss << "\u2715";
            }
            oss << ") | ";
        }
        if (_finished) {
            oss << "\u2714";
        }
        else {
            oss << "\u2716";
        }
        oss << " | ";

        oss << ellapsed_h_min_s;

        std::cout << oss.str() << std::flush; // flush the output to ensure it's displayed immediately
    }


    bool operator()() {
        _curr_iter += 1;
        if (_curr_iter % _check_interval) {
            return false;
        }

        _quantities = _get_quantities();
        for (size_t j = 0; j < _names.size(); j++) {
            _conditions_satisfied[j] = _conditions[j](_quantities[j]);
        }
        _finished = true;
        for (bool satisfied : _conditions_satisfied) {
            _finished = (_finished && satisfied);
        }

        _curr_time = std::chrono::system_clock::now();
        draw_monitor();

        if (_finished) {
            std::cout << std::endl;
            std::cout << "finished after " << _curr_iter << " " << _unit << "." << std::endl;
            return true;
        }
        else {
            return false;
        }
    }


private:
    std::vector<std::string> _names;
    std::vector<double> _quantities;
    std::function<std::vector<double>()> _get_quantities;
    std::vector<std::function<bool(double)>> _conditions;
    size_t _curr_iter;
    std::chrono::system_clock::time_point _start_time;
    std::chrono::system_clock::time_point _curr_time;
    size_t _check_interval;
    std::string _unit;
    std::vector<bool> _conditions_satisfied;
    bool _finished;
};



#endif //MONITOR_H_INCLUDED

