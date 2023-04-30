#pragma once
#include <fstream>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <iostream>
#include <unordered_map>
#include <sstream>

class ConfigLoader {
    private:
        std::unordered_map<std::string, std::string> m_configMap;

        template<typename T, typename std::enable_if<std::is_integral_v<T>, T>::type* = nullptr>
        T parse(const std::string &value)
        {
            return stoi(value);
        }
        template<typename T, typename std::enable_if<std::is_floating_point_v<T>, T>::type* = nullptr>
        T parse(const std::string &value)
        {
            return stof(value);
        }

    public:
        bool loadConfigFile(std::string path)
        {
            std::ifstream configFile = std::ifstream(path);
            if(!configFile.is_open())
            {
                return false;
            }

            for (std::string line; std::getline(configFile, line) ;)
            {
                std::stringstream lineStream = std::stringstream(line);
                std::string key;
                std::string value;

                std::getline(lineStream, key, ':');
                std::getline(lineStream, value);

                if (m_configMap.find(key) != m_configMap.end())
                {
                    printf("duplicate key %s already exists in config\n", key.c_str());
                }

                key.erase(remove_if(key.begin(), key.end(), isspace), key.end());
                value.erase(remove_if(value.begin(), value.end(), isspace), value.end());

                printf("%s %s\n", key.c_str(), value.c_str());
                m_configMap.insert({key, value});
            }
            configFile.close();
            return true;
        }

        template <typename T>
        bool LoadConfigParameter(T &dest, T defaultValue, const std::string key)
        {
            dest = defaultValue;
            auto it = m_configMap.find(key);
            if (it == m_configMap.end())
            {
                printf("Could not find key %s\n", key.c_str());
                return false;
            }
            const std::string &value = it->second;
            dest = parse<T>(value);
            return true;
        }
};