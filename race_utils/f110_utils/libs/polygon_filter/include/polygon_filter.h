#ifndef POLYGON_FILTER_H
#define POLYGON_FILTER_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

class PolygonFilter {
public:
    PolygonFilter();
    ~PolygonFilter();

    // load map info from a YAML file path (image filename, resolution, origin, threshold, etc.)
    bool loadMapFromYAML(const std::string& yamlPath, const std::string& imagePath);

    // set the erosion kernel size and refresh the contours
    void setErosionKernelSize(int pix);

    // convert pixel coordinates to global coordinates (global = origin + pixel * resolution)
    cv::Point2d pixelToGlobal(const cv::Point& pixelPoint);

    // test whether the given global coordinate lies inside the external contour
    bool isPointInside(const cv::Point2d& globalPoint);

    // visualize the contour result (for debugging)
    void visualizeContours();

private:
    // after image processing, update erosion and contours
    void updateContours();

    cv::Mat image;             // image specified in the YAML (grayscale)
    cv::Mat erodedImage;       // image after erosion
    cv::Mat kernel;            // kernel used for erosion
    cv::Size kernelSize;       // kernel size

    // contour info
    std::vector<std::vector<cv::Point>> externalContours; // external contours
    std::vector<std::vector<cv::Point>> internalContours; // internal contours

    // map parameters (read from YAML)
    double resolution;         // resolution (real length per pixel)
    cv::Point2d origin;        // global origin corresponding to pixel (0,0)
    double free_thresh;
    double occupied_thresh;
    int negate;                // whether to invert the occupancy map (0 or 1)
    bool debug;
};

#endif // POLYGON_FILTER_H
