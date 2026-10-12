/*********************************************************************
* Software License Agreement (BSD License)
* 
*  Copyright (c) 2017-2020, Waterplus http://www.6-robot.com
*  All rights reserved.
* 
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
* 
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the WaterPlus nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
* 
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  FOOTPRINTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/
/*!******************************************************************
 @author     ZhangWanjie
 ********************************************************************/

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <sensor_msgs/LaserScan.h>
#include <algorithm>
#include <cmath>
#include <limits>

static std::string pub_topic;
static float self_filter_range;
static float self_mask_1_min_angle_deg;
static float self_mask_1_max_angle_deg;
static float self_mask_1_range;
static float self_mask_2_min_angle_deg;
static float self_mask_2_max_angle_deg;
static float self_mask_2_range;

static bool inAngularMask(float angle_deg, float min_angle_deg,
                          float max_angle_deg, float range,
                          float mask_range)
{
    return mask_range > 0.0f && range <= mask_range &&
           angle_deg >= min_angle_deg && angle_deg <= max_angle_deg;
}
class CWPHLidarFilter
{
public:
    CWPHLidarFilter();
private:
     ros::NodeHandle n;
     ros::Publisher scan_pub;
     ros::Subscriber scan_sub;
     void lidarCallback(const sensor_msgs::LaserScan::ConstPtr& scan);
};

CWPHLidarFilter::CWPHLidarFilter()
{
    scan_pub = n.advertise<sensor_msgs::LaserScan>(pub_topic,1);
    scan_sub = n.subscribe<sensor_msgs::LaserScan>("/scan_raw",1,&CWPHLidarFilter::lidarCallback,this);
}

void CWPHLidarFilter::lidarCallback(const sensor_msgs::LaserScan::ConstPtr& scan)
{
    //ROS_INFO("[wpb_home_lidar_filter]");
    int nRanges = scan->ranges.size();
    float k = nRanges/360.0f;
    sensor_msgs::LaserScan new_scan;
    new_scan.header.stamp = scan->header.stamp;
    new_scan.header.frame_id = scan->header.frame_id;
    new_scan.angle_max = scan->angle_max;
    new_scan.angle_min = scan->angle_min;
    // new_scan.angle_increment = scan->angle_increment;
    new_scan.angle_increment = M_PI/180;
    // new_scan.time_increment = scan->time_increment;
    new_scan.time_increment = scan->time_increment*k;
    new_scan.range_min = scan->range_min;
    new_scan.range_max = scan->range_max;
    new_scan.ranges.resize(360);
    new_scan.intensities.resize(360);
    for(int i=0 ; i<360 ; i++)
    {
        const int source_index = std::min(static_cast<int>(i * k), nRanges - 1);
        const float range = scan->ranges[source_index];
        const float angle_deg = (new_scan.angle_min + i * new_scan.angle_increment)
                                * 180.0f / M_PI;
        // The scanner is mounted inside the robot footprint. A small radial
        // mask removes the inner chassis; the two optional angular masks
        // remove this robot's fixed side/rear chassis returns only. The
        // forward field of view is deliberately left untouched for safety.
        const bool is_self_return = std::isfinite(range) &&
            (range <= self_filter_range ||
             inAngularMask(angle_deg, self_mask_1_min_angle_deg,
                           self_mask_1_max_angle_deg, range, self_mask_1_range) ||
             inAngularMask(angle_deg, self_mask_2_min_angle_deg,
                           self_mask_2_max_angle_deg, range, self_mask_2_range));
        new_scan.ranges[i] = is_self_return
            ? std::numeric_limits<float>::infinity()
            : range;
        new_scan.intensities[i] = scan->intensities[source_index];
    }
    scan_pub.publish(new_scan);
}

int main(int argc, char** argv)
{
    ros::init(argc,argv,"wpb_home_lidar_filter");

     ros::NodeHandle n_param("~");
    std::string strSerialPort;
    n_param.param<std::string>("pub_topic", pub_topic, "/scan_filtered");
    n_param.param<float>("self_filter_range", self_filter_range, 0.25f);
    // Disabled by default to retain the upstream filter behaviour. Robot
    // specific launch files may enable masks after observing fixed chassis
    // reflections in /scan_raw.
    n_param.param<float>("self_mask_1_min_angle_deg", self_mask_1_min_angle_deg, 0.0f);
    n_param.param<float>("self_mask_1_max_angle_deg", self_mask_1_max_angle_deg, 0.0f);
    n_param.param<float>("self_mask_1_range", self_mask_1_range, 0.0f);
    n_param.param<float>("self_mask_2_min_angle_deg", self_mask_2_min_angle_deg, 0.0f);
    n_param.param<float>("self_mask_2_max_angle_deg", self_mask_2_max_angle_deg, 0.0f);
    n_param.param<float>("self_mask_2_range", self_mask_2_range, 0.0f);

    CWPHLidarFilter lidar_filter;
    ros::spin();
}