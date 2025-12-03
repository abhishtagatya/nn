#pragma once

#include <iostream>
#include <ctime>
#include <string>

#ifndef TIME_HPP
#define TIME_HPP

std::string get_time() {
    std::time_t now = std::time(nullptr);
    std::tm* localTime = std::localtime(&now);

    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", localTime);

    return std::string(buffer);
}

#endif // TIME_HPP
