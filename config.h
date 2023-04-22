#pragma once
#include "config_loader.h"

struct Config
{
    inline static unsigned Height = 1080;
    inline static unsigned Width = 1920;
    inline static float Ratio = (float) Width / Height;
    inline static float Epsilon = 0.0001f;
    inline static float FrustrumMax = 1000.0f;

    static bool Load(std::string path)
    {
        ConfigLoader configLoader;
        bool loadingSuccess = configLoader.loadConfigFile(path);
        if (!loadingSuccess)
        {
            return false;
        }

        configLoader.LoadConfigParameter<unsigned>(Height, 1080, "Height");
        configLoader.LoadConfigParameter<unsigned>(Width, 1920, "Width");
        Ratio = (float) Width / Height;
        configLoader.LoadConfigParameter<float>(Epsilon, 0.0001f, "Epsilon");
        configLoader.LoadConfigParameter<float>(FrustrumMax, 1000.0f, "FrustrumMax");

        return true;
    }
};