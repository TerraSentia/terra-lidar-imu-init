// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//#include "ros/package.h"
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include "imu_processing.hpp"
#include "rclcpp/rclcpp.hpp"
#include <unistd.h>
#include <Python.h>
#include <Eigen/Core>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <algorithm>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include "preprocess.h"
#include <ikd-tree/ikd_tree.h>
#include <terra_lidar_imu_init/lidar_imu_init.h>

#ifndef DEPLOY
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
#endif

#define LASER_POINT_COV 0.001
#define MAXN 720000
#define PUBFRAME_PERIOD 20

float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;



class LaserMapping : public rclcpp::Node
{
public:
    LaserMapping()
    : Node("laser_mapping"),
      mean_acc_norm_(9.81),
      time_lag_IMU_wtr_lidar_(0.0),
      timediff_imu_wrt_lidar_(0.0),
      timediff_set_flg_(false),
      Trans_LI_cov_(3, 0.0005),
      Rot_LI_cov_(3, 0.00005), 
      last_rot_(M3D::Zero()), 
      position_last_(V3D::Zero()),
      last_odom_(V3D::Zero())
    {
        featsFromMap = PointCloudXYZI::Ptr(new PointCloudXYZI());
        feats_undistort = PointCloudXYZI::Ptr(new PointCloudXYZI());
        feats_down_body = PointCloudXYZI::Ptr(new PointCloudXYZI());
        feats_down_world_ = PointCloudXYZI::Ptr(new PointCloudXYZI());
        normvec_ = PointCloudXYZI::Ptr(new PointCloudXYZI(100000, 1));
        laserCloudOri_ = PointCloudXYZI::Ptr(new PointCloudXYZI(100000, 1));
        corr_normvect_ = PointCloudXYZI::Ptr(new PointCloudXYZI(100000, 1));
        _featsArray_ = PointCloudXYZI::Ptr(new PointCloudXYZI());
        pcl_wait_save_ = PointCloudXYZI::Ptr(new PointCloudXYZI());

        loadParams();
        initSubscribers();
        initPublishers();
        initState();
        initImuProcessor();
    }

    void calcBodyVar(Eigen::Vector3d &pb, const float range_inc,
        const float degree_inc, Eigen::Matrix3d &var) 
    {
        float range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
        float range_var = range_inc * range_inc;
        Eigen::Matrix2d direction_var;
        direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0,
                pow(sin(DEG2RAD(degree_inc)), 2);
        Eigen::Vector3d direction(pb);
        direction.normalize();
        Eigen::Matrix3d direction_hat;
        direction_hat << 0, -direction(2), direction(1), direction(2), 0,
                -direction(0), -direction(1), direction(0), 0;
        Eigen::Vector3d base_vector1(1, 1,
                                    -(direction(0) + direction(1)) / direction(2));
        base_vector1.normalize();
        Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
        base_vector2.normalize();
        Eigen::Matrix<double, 3, 2> N;
        N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1),
                base_vector1(2), base_vector2(2);
        Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
        var = direction * range_var * direction.transpose() +
            A * direction_var * A.transpose();
    }
    
    void sigHandle(int sig) {
        if (pcd_save_en_ && pcd_save_interval_ < 0){
            all_points_dir_ = string(root_dir_ + "/PCD/PCD_all" + string(".pcd"));
            pcd_writer_.writeBinary(all_points_dir_, *pcl_wait_save_);
        }
        flag_exit_ = true;
        RCLCPP_WARN(this->get_logger(), "catch sig %d", sig);
        sig_buffer_.notify_all();
    }
    
    void pointBodyToWorld(PointType const *const pi, PointType *const po) {
        V3D p_body(pi->x, pi->y, pi->z);
        V3D p_global(state_.rot_end * (state_.offset_R_L_I * p_body + state_.offset_T_L_I) + state_.pos_end);

        po->x = p_global(0);
        po->y = p_global(1);
        po->z = p_global(2);
        po->normal_x = pi->normal_x;
        po->normal_y = pi->normal_y;
        po->normal_z = pi->normal_z;
        po->intensity = pi->intensity;
    }

    template<typename T>
    void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po) {
        V3D p_body(pi[0], pi[1], pi[2]);
        V3D p_global(state_.rot_end * (state_.offset_R_L_I * p_body + state_.offset_T_L_I) + state_.pos_end);
        po[0] = p_global(0);
        po[1] = p_global(1);
        po[2] = p_global(2);
    }

    void rgbPointBodyToWorld(PointType const *const pi, PointTypeRGB *const po) {
        V3D p_body(pi->x, pi->y, pi->z);
        V3D p_global(state_.rot_end * (state_.offset_R_L_I * p_body + state_.offset_T_L_I) + state_.pos_end);
        po->x = p_global(0);
        po->y = p_global(1);
        po->z = p_global(2);
        po->r = pi->normal_x;
        po->g = pi->normal_y;
        po->b = pi->normal_z;

        float intensity = pi->intensity;
        intensity = intensity - floor(intensity);

        int reflection_map = intensity * 10000;
    }


    void pointsCacheCollect() {
        PointVector points_history;
        ikdtree_.acquire_removed_points(points_history);
        points_cache_size_ = points_history.size();
        for (int i = 0; i < points_history.size(); i++) {
            _featsArray_->push_back(points_history[i]);
        }
    }

    void lasermapFovSegment() {
        cub_needrm.clear();

        pointBodyToWorld(XAxisPoint_body_, XAxisPoint_world_);
        V3D pos_LiD = state_.pos_end;

        if (!local_map_initialized_) {
            for (int i = 0; i < 3; i++) {
                local_map_points_.vertex_min[i] = pos_LiD(i) - cube_len_ / 2.0;
                local_map_points_.vertex_max[i] = pos_LiD(i) + cube_len_ / 2.0;
            }
            local_map_initialized_ = true;
            return;
        }

        float dist_to_map_edge[3][2];
        bool need_move = false;
        for (int i = 0; i < 3; i++) {
            dist_to_map_edge[i][0] = fabs(pos_LiD(i) - local_map_points_.vertex_min[i]);
            dist_to_map_edge[i][1] = fabs(pos_LiD(i) - local_map_points_.vertex_max[i]);
            if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE ||
                dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
                need_move = true;
        }
        if (!need_move) return;
        
        BoxPointType new_local_map_points, tmp_boxpoints;
        new_local_map_points = local_map_points_;
        float mov_dist = max((cube_len_ - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9,
                            double(DET_RANGE * (MOV_THRESHOLD - 1)));
        for (int i = 0; i < 3; i++) {
            tmp_boxpoints = local_map_points_;
            if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE) {
                new_local_map_points.vertex_max[i] -= mov_dist;
                new_local_map_points.vertex_min[i] -= mov_dist;
                tmp_boxpoints.vertex_min[i] = local_map_points_.vertex_max[i] - mov_dist;
                cub_needrm.push_back(tmp_boxpoints);
            } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) {
                new_local_map_points.vertex_max[i] += mov_dist;
                new_local_map_points.vertex_min[i] += mov_dist;
                tmp_boxpoints.vertex_max[i] = local_map_points_.vertex_min[i] + mov_dist;
                cub_needrm.push_back(tmp_boxpoints);
            }
        }
        local_map_points_ = new_local_map_points;
        pointsCacheCollect();
    }

    void processPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg, bool isLivox) {
        mtx_buffer_.lock();
        scan_count_++;

        double timestamp = isLivox ? (msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9) : rclcpp::Time(msg->header.stamp).seconds();

        if (timestamp < last_timestamp_lidar_) {
            RCLCPP_WARN(this->get_logger(), "Lidar loop back, clearing buffer");
            lidar_buffer_.clear();
            time_buffer_.clear();
        }

        last_timestamp_lidar_ = timestamp;

        if (abs(last_timestamp_imu_ - last_timestamp_lidar_) > 1.0 && !timediff_set_flg_ && !imu_buffer_.empty()) {
            timediff_set_flg_ = true;
            timediff_imu_wrt_lidar_ = last_timestamp_imu_ - last_timestamp_lidar_;
            RCLCPP_INFO(this->get_logger(), "Self sync IMU and LiDAR, HARD time lag is %.10lf", timediff_imu_wrt_lidar_);
        }

        std::deque<PointCloudXYZI::Ptr> ptr;
        std::deque<double> timestamp_lidar;

        if (cut_frame_) {
            if (isLivox) {
                p_preprocess_->process_cut_frame_livox(msg, ptr, timestamp_lidar, cut_frame_num_, scan_count_);
            } else {
                p_preprocess_->process_cut_frame_pcl2(msg, ptr, timestamp_lidar, cut_frame_num_, scan_count_);
            }

            while (!ptr.empty() && !timestamp_lidar.empty()) {
                lidar_buffer_.push_back(ptr.front());
                ptr.pop_front();
                time_buffer_.push_back(timestamp_lidar.front() / 1000.0); // Convert to seconds
                timestamp_lidar.pop_front();
            }
        } else {
            PointCloudXYZI::Ptr single_ptr(new PointCloudXYZI());
            p_preprocess_->process(msg, single_ptr);
            lidar_buffer_.push_back(single_ptr);
            time_buffer_.push_back(last_timestamp_lidar_);
        }

        mtx_buffer_.unlock();
        sig_buffer_.notify_all();
    }

    // void livoxPclCb(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg) {
    //     processPointCloud(msg, true);
    // }

    void standardPclCb(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        processPointCloud(msg, false);
    }

    void imuCb(const sensor_msgs::msg::Imu::SharedPtr msg_in) {
        publish_count_++;
        mtx_buffer_.lock();

        static double IMU_period, time_msg_in, last_time_msg_in;
        static int imu_cnt = 0;
        time_msg_in = rclcpp::Time(msg_in->header.stamp).seconds();


        if (imu_cnt < 100) {
            imu_cnt++;
            mean_acc += (V3D(msg_in->linear_acceleration.x, msg_in->linear_acceleration.y, msg_in->linear_acceleration.z) -
                        mean_acc) / (imu_cnt);
            if (imu_cnt > 1) {
                IMU_period += (time_msg_in - last_time_msg_in - IMU_period) / (imu_cnt - 1);
            }
            if (imu_cnt == 99) {
                cout << endl << "Acceleration norm  : " << mean_acc.norm() << endl;
                if (IMU_period > 0.01) {
                    cout << "IMU data frequency : " << 1 / IMU_period << " Hz" << endl;
                    ROS_WARN("IMU data frequency too low. Higher than 150 Hz is recommended.");
                }
                cout << endl;
            }
        }
        last_time_msg_in = time_msg_in;


        sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));

        //IMU Time Compensation
        msg->header.stamp = rclcpp::Time().fromSec(msg->header.stamp.sec - timediff_imu_wrt_lidar_ - time_lag_IMU_wtr_lidar_);
        double timestamp = rclcpp::Time(msg->header.stamp).seconds();

        if (timestamp < last_timestamp_imu_) {
            RCLCPP_WARN(this->get_logger(), "IMU loop back, clear IMU buffer.");
            imu_buffer_.clear();
            Init_LI_->IMU_buffer_clear();
        }

        last_timestamp_imu_ = timestamp;
        imu_buffer_.push_back(msg);

        // push all IMU meas into Init_LI
        if (!imu_en_ && !data_accum_finished_)
            Init_LI_->push_ALL_IMU_CalibState(msg, mean_acc_norm_);

        mtx_buffer_.unlock();
        sig_buffer_.notify_all();
    }

    void map_incremental() 
    {
        PointVector point_to_add;
        PointVector point_no_need_downsample;
        point_to_add.reserve(feats_down_size_);
        point_no_need_downsample.reserve(feats_down_size_);
        for (int i = 0; i < feats_down_size_; i++) {
            /* transform to world frame */
            pointBodyToWorld(&(feats_down_body_->points[i]), &(feats_down_world_->points[i]));
            /* decide if need add to map */
            if (!Nearest_Points_[i].empty() && flg_EKF_inited_) 
            {
                const PointVector &points_near = Nearest_Points_[i];
                bool need_add = true;
                BoxPointType box_of_point;
                PointType downsample_result, mid_point;
                mid_point.x = floor(feats_down_world_->points[i].x / filter_size_map_min_) * filter_size_map_min_ +
                            0.5 * filter_size_map_min_;
                mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min +
                            0.5 * filter_size_map_min;
                mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min +
                            0.5 * filter_size_map_min;
                float dist = calc_dist(feats_down_world->points[i], mid_point);
                if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min &&
                    fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min &&
                    fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min) {
                    PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                    continue;
                }
                for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++) {
                    if (points_near.size() < NUM_MATCH_POINTS) break;
                    if (calc_dist(points_near[readd_i], mid_point) < dist) {
                        need_add = false;
                        break;
                    }
                }
                if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
            } else {
                PointToAdd.push_back(feats_down_world->points[i]);
            }
        }
            add_point_size = ikdtree.Add_Points(PointToAdd, true);
            ikdtree.Add_Points(PointNoNeedDownsample, false);
            add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    }

    void publish_frame_world() {
        if (scan_pub_en_) {
            PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en_ ? feats_undistort_ : feats_down_body_);
            int size = laserCloudFullRes->points.size();

            PointCloudXYZRGB::Ptr laserCloudWorldRGB(new PointCloudXYZRGB(size, 1));
            PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

            for (int i = 0; i < size; i++) {
                if (lidar_type_ == L515)
                    rgbPointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorldRGB->points[i]);
                else
                    pointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
            }

            sensor_msgs::msg::PointCloud2 laserCloudmsg;
            if (lidar_type_ == L515)
                pcl::toROSMsg(*laserCloudWorldRGB, laserCloudmsg);
            else
                pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);

            laserCloudmsg.header.stamp = rclcpp::Time(lidar_end_time_ * 1e9);
            laserCloudmsg.header.frame_id = "camera_init";
            pubLaserCloudFullRes_->publish(laserCloudmsg);
            publish_count_ -= PUBFRAME_PERIOD;
        }

        // Save map if enabled
        if (pcd_save_en_) {
            std::filesystem::create_directories(root_dir_ + "/PCD");
            int size = feats_undistort_->points.size();
            PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
            for (int i = 0; i < size; i++) {
                pointBodyToWorld(&feats_undistort_->points[i], &laserCloudWorld->points[i]);
            }

            *pcl_wait_save_ += *laserCloudWorld;
            static int scan_wait_num = 0;
            scan_wait_num++;
            if (pcl_wait_save_->size() > 0 && pcd_save_interval_ > 0 && 
                scan_wait_num >= pcd_save_interval_) {
                pcd_index_++;
                all_points_dir_ = std::string(root_dir_ + "/PCD/PCD") + 
                                  std::to_string(pcd_index_) + std::string(".pcd");
                RCLCPP_INFO(this->get_logger(), "Current scan saved to %s", all_points_dir_.c_str());
                pcd_writer_.writeBinary(all_points_dir_, *pcl_wait_save_);
                pcl_wait_save_->clear();
                scan_wait_num = 0;
            }
        }
    }

    void publish_frame_body() {
        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*feats_undistort_, laserCloudmsg);
        laserCloudmsg.header.stamp = rclcpp::Time(lidar_end_time_ * 1e9);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFullRes_body_->publish(laserCloudmsg);
    }

    void publish_effect_world() {
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effect_feat_num_, 1));
        for (int i = 0; i < effect_feat_num_; i++) {
            pointBodyToWorld(&laserCloudOri_->points[i], &laserCloudWorld->points[i]);
        }
        sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
        pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
        laserCloudFullRes3.header.stamp = rclcpp::Time(lidar_end_time_ * 1e9);
        laserCloudFullRes3.header.frame_id = "camera_init";
        pubLaserCloudEffect_->publish(laserCloudFullRes3);
    }

    void publish_map() {
        sensor_msgs::msg::PointCloud2 laserCloudMap;
        pcl::toROSMsg(*featsFromMap_, laserCloudMap);
        laserCloudMap.header.stamp = rclcpp::Time(lidar_end_time_ * 1e9);
        laserCloudMap.header.frame_id = "camera_init";
        pubLaserCloudMap_->publish(laserCloudMap);
    }

    template<typename T>
    void set_posestamp(T &out) {
        if (!imu_en) {
            out.position.x = state.pos_end(0);
            out.position.y = state.pos_end(1);
            out.position.z = state.pos_end(2);
        } else {
            // Publish LiDAR's pose and position
            V3D pos_cur_lidar = state.rot_end * state.offset_T_L_I + state.pos_end;
            out.position.x = pos_cur_lidar(0);
            out.position.y = pos_cur_lidar(1);
            out.position.z = pos_cur_lidar(2);
        }
        out.orientation.x = geoQuat.x;
        out.orientation.y = geoQuat.y;
        out.orientation.z = geoQuat.z;
        out.orientation.w = geoQuat.w;
    }

    void publish_odometry() {
        odomAftMapped_.header.frame_id = "camera_init";
        odomAftMapped_.child_frame_id = "aft_mapped";
        odomAftMapped_.header.stamp = rclcpp::Time(lidar_end_time_ * 1e9);
        set_posestamp(odomAftMapped_.pose.pose);

        pubOdomAftMapped_->publish(odomAftMapped_);

        // TF2 broadcast for ROS2
        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = odomAftMapped_.header.stamp;
        transform.header.frame_id = "camera_init";
        transform.child_frame_id = "aft_mapped";
        transform.transform.translation.x = odomAftMapped_.pose.pose.position.x;
        transform.transform.translation.y = odomAftMapped_.pose.pose.position.y;
        transform.transform.translation.z = odomAftMapped_.pose.pose.position.z;
        transform.transform.rotation = odomAftMapped_.pose.pose.orientation;
        
        tf_broadcaster_->sendTransform(transform);
    }

    void publish_path() {
        set_posestamp(msg_body_pose_.pose);
        msg_body_pose_.header.stamp = rclcpp::Time(lidar_end_time_ * 1e9);
        msg_body_pose_.header.frame_id = "camera_init";
        
        static int jjj = 0;
        jjj++;
        if (jjj % 5 == 0) { // if path is too large, the RVIZ will crash
            path_.poses.push_back(msg_body_pose_);
            pubPath_->publish(path_);
        }
    }

    void loadParams() {
        // Parameter declarations
        this->declare_parameter("max_iteration", 4);
        this->declare_parameter("point_filter_num", 2);
        this->declare_parameter("map_file_path", "");
        this->declare_parameter("common/lid_topic", "/livox/lidar");
        this->declare_parameter("common/imu_topic", "/livox/imu");
        this->declare_parameter("mapping/filter_size_surf", 0.5);
        this->declare_parameter("mapping/filter_size_map", 0.5);
        this->declare_parameter("cube_side_length", 200.0);
        this->declare_parameter("mapping/det_range", 300.f);
        this->declare_parameter("mapping/gyr_cov", 0.1);
        this->declare_parameter("mapping/acc_cov", 0.1);
        this->declare_parameter("mapping/grav_cov", 0.001);
        this->declare_parameter("mapping/b_gyr_cov", 0.0001);
        this->declare_parameter("mapping/b_acc_cov", 0.0001);
        this->declare_parameter("preprocess/blind", 1.0);
        this->declare_parameter("preprocess/lidar_type", AVIA);
        this->declare_parameter("preprocess/scan_line", 16);
        this->declare_parameter("preprocess/feature_extract_en", false);
        this->declare_parameter("initialization/cut_frame", true);
        this->declare_parameter("initialization/cut_frame_num", 1);
        this->declare_parameter("initialization/orig_odom_freq", 10);
        this->declare_parameter("initialization/online_refine_time", 20.0);
        this->declare_parameter("initialization/mean_acc_norm", 9.81);
        this->declare_parameter("initialization/data_accum_length", 300.0);
        this->declare_parameter("initialization/Rot_LI_cov", std::vector<double>());
        this->declare_parameter("initialization/Trans_LI_cov", std::vector<double>());
        this->declare_parameter("publish/path_en", true);
        this->declare_parameter("publish/scan_publish_en", true);
        this->declare_parameter("publish/dense_publish_en", true);
        this->declare_parameter("publish/scan_bodyframe_pub_en", true);
        this->declare_parameter("runtime_pos_log_enable", false);
        this->declare_parameter("pcd_save/pcd_save_en", false);
        this->declare_parameter("pcd_save/interval", -1);
        
        // Get parameters
        NUM_MAX_ITERATIONS = this->get_parameter("max_iteration").as_int();
        p_preprocess_->point_filter_num = this->get_parameter("point_filter_num").as_int();
        map_file_path_ = this->get_parameter("map_file_path").as_string();
        lid_topic_ = this->get_parameter("common/lid_topic").as_string();
        imu_topic_ = this->get_parameter("common/imu_topic").as_string();
        filter_size_surf_min_ = this->get_parameter("mapping/filter_size_surf").as_double();
        filter_size_map_min_ = this->get_parameter("mapping/filter_size_map").as_double();
        cube_len_ = this->get_parameter("cube_side_length").as_double();
        DET_RANGE = this->get_parameter("mapping/det_range").as_double();
        gyr_cov_ = this->get_parameter("mapping/gyr_cov").as_double();
        acc_cov_ = this->get_parameter("mapping/acc_cov").as_double();
        grav_cov_ = this->get_parameter("mapping/grav_cov").as_double();
        b_gyr_cov_ = this->get_parameter("mapping/b_gyr_cov").as_double();
        b_acc_cov_ = this->get_parameter("mapping/b_acc_cov").as_double();
        p_preprocess_->blind = this->get_parameter("preprocess/blind").as_double();
        lidar_type_ = this->get_parameter("preprocess/lidar_type").as_int();
        p_preprocess_->N_SCANS = this->get_parameter("preprocess/scan_line").as_int();
        p_preprocess_->feature_enabled = this->get_parameter("preprocess/feature_extract_en").as_bool();
        cut_frame_ = this->get_parameter("initialization/cut_frame").as_bool();
        cut_frame_num_ = this->get_parameter("initialization/cut_frame_num").as_int();
        orig_odom_freq_ = this->get_parameter("initialization/orig_odom_freq").as_int();
        online_refine_time_ = this->get_parameter("initialization/online_refine_time").as_double();
        mean_acc_norm_ = this->get_parameter("initialization/mean_acc_norm").as_double();
        Init_LI_->data_accum_length = this->get_parameter("initialization/data_accum_length").as_double();
        Rot_LI_cov_ = this->get_parameter("initialization/Rot_LI_cov").as_double_array();
        Trans_LI_cov_ = this->get_parameter("initialization/Trans_LI_cov").as_double_array();
        path_en_ = this->get_parameter("publish/path_en").as_bool();
        scan_pub_en_ = this->get_parameter("publish/scan_publish_en").as_bool();
        dense_pub_en_ = this->get_parameter("publish/dense_publish_en").as_bool();
        scan_body_pub_en_ = this->get_parameter("publish/scan_bodyframe_pub_en").as_bool();
        runtime_pos_log_ = this->get_parameter("runtime_pos_log_enable").as_bool();
        pcd_save_en_ = this->get_parameter("pcd_save/pcd_save_en").as_bool();
        pcd_save_interval_ = this->get_parameter("pcd_save/interval").as_int();
        
    }

    void initSubscribers()
    {
        // Initialize subscribers
        sub_pcl_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            lid_topic_, 200000, std::bind(&LaserMapping::standardPclCb, this, std::placeholders::_1));
        
        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic_, 200000, std::bind(&LaserMapping::imuCb, this, std::placeholders::_1));
    
    }

    void initPublishers()
    {
        // Initialise publishers
        pubIMU_sync_ = this->create_publisher<sensor_msgs::msg::Imu>("/livox/imu/async", 100000);
        pubLaserCloudFullRes_ = 
            this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 100000);
        pubLaserCloudFullRes_body_ = 
            this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", 100000);
        pubLaserCloudEffect_ = 
            this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected", 100000);
        pubLaserCloudMap_ = 
            this->create_publisher<sensor_msgs::msg::PointCloud2>("/Laser_map", 100000);
        pubOdomAftMapped_ = 
            this->create_publisher<nav_msgs::msg::Odometry>("/aft_mapped_to_init", 100000);
        pubPath_ = this->create_publisher<nav_msgs::msg::Path>("/path", 100000);
    }

    void initState()
    {
        // Initialize the state
        downSizeFilterSurf_.setLeafSize(filter_size_surf_min_, filter_size_surf_min_, filter_size_surf_min_);
        downSizeFilterMap_.setLeafSize(filter_size_map_min_, filter_size_map_min_, filter_size_map_min_);
    }

    void initImuProcessor()
    {
        // Initialize IMU processor
        p_imu_ = std::make_shared<ImuProcess>();
        p_preprocess_ = std::make_shared<PreProcess>();
        p_imu_->lidar_type = p_preprocess_>lidar_type = lidar_type_;
        p_imu_->imu_en = imu_en_;
        p_imu_->LI_init_done = false;
        p_imu_->set_gyr_cov(V3D(gyr_cov_, gyr_cov_, gyr_cov_));
        p_imu_->set_acc_cov(V3D(acc_cov_, acc_cov_, acc_cov_));
        p_imu_->set_R_LI_cov(V3D(VEC_FROM_ARRAY(Rot_LI_cov_)));
        p_imu_->set_T_LI_cov(V3D(VEC_FROM_ARRAY(Trans_LI_cov_)));
        p_imu_->set_gyr_bias_cov(V3D(b_gyr_cov_, b_gyr_cov_, b_gyr_cov_));
        p_imu_->set_acc_bias_cov(V3D(b_acc_cov_, b_acc_cov_, b_acc_cov_));
        
        // Set up debug logging
        // std::filesystem::create_directories(root_dir_ + "/Log");
        // std::filesystem::create_directories(root_dir_ + "/result");
        // fout_out_.open(DEBUG_FILE_DIR("mat_out.txt"), std::ios::out);
        // fout_result_.open(RESULT_FILE_DIR("Initialization_result.txt"), std::ios::out);
        
        RCLCPP_INFO(this->get_logger(), "LiDAR-IMU calibration initialized");
    }

private:
    mutex mtx_buffer_;
    condition_variable sig_buffer_;

    string root_dir_ = ROOT_DIR;
    string map_file_path_, lid_topic_, imu_topic_;

    int iterCount_ = 0, feats_down_size_ = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum_ = 0, \
    effect_feat_num_ = 0, scan_count_ = 0, publish_count_ = 0;

    double res_mean_last_ = 0.05;
    double gyr_cov_ = 0.1, acc_cov_ = 0.1, grav_cov_ = 0.0001, b_gyr_cov_ = 0.0001, b_acc_cov_ = 0.0001;
    double last_timestamp_lidar_ = 0, last_timestamp_imu_ = 0.0;
    double filter_size_surf_min_ = 0, filter_size_map_min_ = 0;
    double cube_len_ = 0, total_distance_ = 0, lidar_end_time_ = 0, first_lidar_time_ = 0.0;

    // Time Log Variables
    int kdtree_size_st_ = 0, kdtree_size_end_ = 0, add_point_size_ = 0;


    int lidar_type_, pcd_save_interval_ = -1, pcd_index_ = 0;
    bool lidar_pushed_, flag_reset_, flag_exit_ = false, flg_EKF_inited_ = true;
    bool imu_en_ = false;
    bool scan_pub_en_ = false, dense_pub_en_ = false, scan_body_pub_en_ = false;
    bool runtime_pos_log_ = false, pcd_save_en_ = false, extrinsic_est_en_ = true, path_en_ = true;

    // LI-Init Parameters
    bool cut_frame_ = true, data_accum_finished_ = false, data_accum_start_ = false, online_calib_finish_ = false, refine_print_ = false;
    int cut_frame_num_ = 1, orig_odom_freq_ = 10, frame_num_ = 0;
    double time_lag_IMU_wtr_lidar_ = 0.0, move_start_time_ = 0.0, online_calib_starts_time_ = 0.0, mean_acc_norm_ = 9.81;
    double online_refine_time_ = 20.0; //unit: s
    V3D mean_acc_ = Zero3d;
    ofstream fout_result_;


    vector<BoxPointType> cub_needrm_;
    deque<PointCloudXYZI::Ptr> lidar_buffer_;
    deque<double> time_buffer_;
    deque<sensor_msgs::msg::Imu::SharedPtr> imu_buffer_;
    vector<vector<int>> pointSearchInd_surf_;
    vector<PointVector> Nearest_Points_;
    bool point_selected_surf_[100000] = {0};
    float res_last_[100000] = {0.0};
    double total_residual_;

    std::vector<double> Trans_LI_cov_;
    std::vector<double> Rot_LI_cov_;

    //surf feature in map
    PointCloudXYZI::Ptr featsFromMap;
    PointCloudXYZI::Ptr feats_undistort;
    PointCloudXYZI::Ptr feats_down_body;
    PointCloudXYZI::Ptr feats_down_world_;
    PointCloudXYZI::Ptr normvec_;
    PointCloudXYZI::Ptr laserCloudOri_;
    PointCloudXYZI::Ptr corr_normvect_;
    PointCloudXYZI::Ptr _featsArray_;

    pcl::VoxelGrid<PointType> downSizeFilterSurf_;
    pcl::VoxelGrid<PointType> downSizeFilterMap_;

    KD_TREE ikdtree_;

    M3D last_rot_;
    V3F XAxisPoint_body_{static_cast<float>(LIDAR_SP_LEN), 0.0f, 0.0f};
    V3F XAxisPoint_world_{static_cast<float>(LIDAR_SP_LEN), 0.0f, 0.0f};
    V3D euler_cur_;
    V3D position_last_;
    V3D last_odom_;


    //estimator inputs and output;
    MeasureGroup measures_;
    StatesGroup state_;

    PointCloudXYZI::Ptr pcl_wait_save_;
    pcl::PCDWriter pcd_writer_;
    string all_points_dir_;

    nav_msgs::msg::Path path_;
    nav_msgs::msg::Odometry odomAftMapped_;
    geometry_msgs::msg::Quaternion geoQuat_;
    geometry_msgs::msg::PoseStamped msg_body_pose_;
    sensor_msgs::msg::Imu IMU_sync_;

    std::shared_ptr<Preprocess> p_preprocess_;
    std::shared_ptr<LI_Init> p_LI_init_;
    std::shared_ptr<ImuProcess> p_imu_;

    int points_cache_size_ = 0;
    BoxPointType local_map_points_;
    bool local_map_initialized_ = false;

    double timediff_imu_wrt_lidar_ = 0.0;
    bool timediff_set_flg_ = false;


    int process_increments_ = 0;
    
    VD(DIM_STATE) solution_;
    MD(DIM_STATE, DIM_STATE) G_, H_T_H_, I_STATE_;
    V3D rot_add_, T_add_, vel_add_, gyr_add_;

    StatesGroup state_propagat_;
    PointType point_ori_, point_sel_, coeff_;

    double delta_T_, delta_R_;
    bool flag_ekf_converged_, ekf_stop_flag_ = 0;



    // ROS2 specific
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_livox_pcl_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pubIMU_sync_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullRes_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullRes_body_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    // File output
    std::ofstream fout_out_;
};



float calc_dist(PointType p1, PointType p2) {
    float d = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
    return d;
}



inline void dump_lio_state_to_log(FILE *fp) {
    V3D rot_ang(Log(state.rot_end));
    fprintf(fp, "%lf ", measures_.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state.pos_end(0), state.pos_end(1), state.pos_end(2)); // Pos
    fprintf(fp, "%lf %lf %lf ", state.vel_end(0), state.vel_end(1), state.vel_end(2)); // Vel
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc
    fprintf(fp, "%lf %lf %lf ", state.bias_g(0), state.bias_g(1), state.bias_g(2));    // Bias_g
    fprintf(fp, "%lf %lf %lf ", state.bias_a(0), state.bias_a(1), state.bias_a(2));    // Bias_a
    fprintf(fp, "%lf %lf %lf ", state.gravity(0), state.gravity(1), state.gravity(2)); // Bias_a  
    fprintf(fp, "\r\n");
    fflush(fp);
}







bool sync_packages(MeasureGroup &meas) {
    if (lidar_buffer.empty() || imu_buffer.empty()){
        return false;
    }


    /** push a lidar scan **/
    if (!lidar_pushed) {
        meas.lidar = lidar_buffer.front();

        if (meas.lidar->points.size() <= 1) {
            ROS_WARN("Too few input point cloud!\n");
            lidar_buffer.pop_front();
            time_buffer.pop_front();
            return false;
        }

        meas.lidar_beg_time = time_buffer.front(); //unit:s

        if (lidar_type == L515)
            lidar_end_time = meas.lidar_beg_time;
        else
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000); //unit:s

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
        return false;


    /** push imu data, and pop from imu buffer **/
    double imu_time = imu_buffer.front()->header.stamp.sec;
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time)) {
        imu_time = imu_buffer.front()->header.stamp.sec;
        if (imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }
    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

bool sync_packages_only_lidar(MeasureGroup &meas) {
    if (lidar_buffer.empty())
        return false;

    /** push a lidar scan **/
    if (!lidar_pushed) {
        meas.lidar = lidar_buffer.front();

        if (meas.lidar->points.size() <= 1) {
            ROS_WARN("Too few input point cloud!\n");
            lidar_buffer.pop_front();
            time_buffer.pop_front();
            return false;
        }

        meas.lidar_beg_time = time_buffer.front(); //unit:s

        if (lidar_type == L515)
            lidar_end_time = meas.lidar_beg_time;
        else
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000); //unit:s

        lidar_pushed = true;
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

void fileout_calib_result() {
    fout_result.setf(ios::fixed);
    fout_result << setprecision(6)
                << "Rotation LiDAR to IMU (degree)     = " << RotMtoEuler(state.offset_R_L_I).transpose() * 57.3
                << endl;
    fout_result << "Translation LiDAR to IMU (meter)   = " << state.offset_T_L_I.transpose() << endl;
    fout_result << "Time Lag IMU to LiDAR (second)     = " << time_lag_IMU_wtr_lidar + timediff_imu_wrt_lidar << endl;
    fout_result << "Bias of Gyroscope  (rad/s)         = " << state.bias_g.transpose() << endl;
    fout_result << "Bias of Accelerometer (meters/s^2) = " << state.bias_a.transpose() << endl;
    fout_result << "Gravity in World Frame(meters/s^2) = " << state.gravity.transpose() << endl << endl;

    MD(4, 4) Transform;
    Transform.setIdentity();
    Transform.block<3, 3>(0, 0) = state.offset_R_L_I;
    Transform.block<3, 1>(0, 3) = state.offset_T_L_I;
    fout_result << "Homogeneous Transformation Matrix from LiDAR to IMU: " << endl;
    fout_result << Transform << endl << endl << endl;
}

void print_refine_result() {
    cout.setf(ios::fixed);
    cout << endl;
    printf(BOLDGREEN "[Final Result] " RESET);
    cout << setprecision(6)
         << "Rotation LiDAR to IMU    = " << RotMtoEuler(state.offset_R_L_I).transpose() * 57.3 << " deg" << endl;
    printf(BOLDGREEN "[Final Result] " RESET);
    cout << "Translation LiDAR to IMU = " << state.offset_T_L_I.transpose() << " m" << endl;
    printf(BOLDGREEN "[Final Result] " RESET);
    printf("Time Lag IMU to LiDAR    = %.8lf s \n", time_lag_IMU_wtr_lidar + timediff_imu_wrt_lidar);
    printf(BOLDGREEN "[Final Result] " RESET);
    cout << "Bias of Gyroscope        = " << state.bias_g.transpose() << " rad/s" << endl;
    printf(BOLDGREEN "[Final Result] " RESET);
    cout << "Bias of Accelerometer    = " << state.bias_a.transpose() << " m/s^2" << endl;
    printf(BOLDGREEN "[Final Result] " RESET);
    cout << "Gravity in World Frame   = " << state.gravity.transpose() << " m/s^2" << endl;
}

void printProgress(double percentage) {
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf("\033[1A\r");
    printf(BOLDMAGENTA "[Refinement] ");
    if (percentage < 1) {
        printf(BOLDYELLOW "Online Refinement: ");
        printf(YELLOW "%3d%% [%.*s%*s]\n", val, lpad, PBSTR, rpad, "");
        cout << RESET;
    } else {
        printf(BOLDGREEN " Online Refinement ");
        printf(GREEN "%3d%% [%.*s%*s]\n", val, lpad, PBSTR, rpad, "");
        cout << RESET;
    }
    fflush(stdout);
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LaserMapping>();
    
    // Setup signal handling
    std::signal(SIGINT, [](int sig) {
        rclcpp::shutdown();
    });
    
    RCLCPP_INFO(node->get_logger(), "LiDAR-IMU calibration node started");
    
    // TODO: variable definition
    // LI Init Related
    MatrixXd Jaco_rot(30000, 3);
    Jaco_rot.setZero();
    
    // Create a thread for processing data
    std::thread processing_thread([&]() {
        while (rclcpp::ok()) {
            if (node->flag_exit_) break;
            rclcpp::spin_once(node);

            if (sync_packages(node->measures_)) {
                if (node->flag_reset_) {
                    RCLCPP_WARN(node->get_logger(), "reset when rosbag play back.");
                    node->p_imu_->Reset();
                    node->flag_reset_ = false;
                    continue;
                }
            
            if (node->feats_undistort_->empty() || (node->feats_undistort_ == NULL)) {
                node->first_lidar_time_ = node->measures_.lidar_beg_time;
                node->p_imu_->first_lidar_time = node->first_lidar_time_;
                RCLCPP_WARN(node->get_logger(), "LI-Init not ready, no points stored.");
            }

            node->p_imu_->Process(node->measures_, node->state_, node->feat_undistorted_);

            node->state_propogat_ = node->state_;

            lasermapFovSegment();

            downSizeFilterSurf_.setInputCloud(node->feats_undistort_);
            downSizeFilterSurf_.filter(node->*feats_down_body_);

            node->feats_down_size_ = node->feats_down_body_->points.size();

            if (ikdtree.Root_Node == nullptr) {
                if (node->feats_down_size_ > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for (int i = 0; i < feats_down_size; i++) {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree.Build(feats_down_world->points);
                }
                continue;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();


            /*** ICP and iterated Kalman filter update ***/
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);
            euler_cur = RotMtoEuler(state.rot_end);


            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int rematch_num = 0;
            bool nearest_search_en = true;


            /*** iterated state estimation ***/
            std::vector<M3D> body_var;
            std::vector<M3D> crossmat_list;
            body_var.reserve(feats_down_size);
            crossmat_list.reserve(feats_down_size);




            for (iterCount = 0; iterCount < NUM_MAX_ITERATIONS; iterCount++) {

                laserCloudOri->clear();
                corr_normvect->clear();
                total_residual = 0.0;

                /** closest surface search and residual computation **/
                #ifdef MP_EN
                    omp_set_num_threads(MP_PROC_NUM);
                    #pragma omp parallel for
                #endif
                for (int i = 0; i < feats_down_size; i++) {
                    PointType &point_body = feats_down_body->points[i];
                    PointType &point_world = feats_down_world->points[i];
                    V3D p_body(point_body.x, point_body.y, point_body.z);
                    /// transform to world frame
                    pointBodyToWorld(&point_body, &point_world);
                    vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
                    auto &points_near = Nearest_Points[i];
                    uint8_t search_flag = 0;

                    if (nearest_search_en) {
                        /** Find the closest surfaces in the map **/
                        ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis, 5);
                        if (points_near.size() < NUM_MATCH_POINTS)
                            point_selected_surf[i] = false;
                        else
                            point_selected_surf[i] = !(pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5);
                    }

                    res_last[i] = -1000.0f;

                    if (!point_selected_surf[i] || points_near.size() < NUM_MATCH_POINTS) {
                        point_selected_surf[i] = false;
                        continue;
                    }

                    point_selected_surf[i] = false;
                    VD(4) pabcd;
                    pabcd.setZero();
                    if (esti_plane(pabcd, points_near, 0.1)) //(planeValid)
                    {
                        float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z +
                                    pabcd(3);
                        float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

                        if (s > 0.9) {
                            point_selected_surf[i] = true;
                            normvec->points[i].x = pabcd(0);
                            normvec->points[i].y = pabcd(1);
                            normvec->points[i].z = pabcd(2);
                            normvec->points[i].intensity = pd2;
                            res_last[i] = abs(pd2);
                        }
                    }
                }
                effect_feat_num = 0;
                for (int i = 0; i < feats_down_size; i++) {
                    if (point_selected_surf[i]) {
                        laserCloudOri->points[effect_feat_num] = feats_down_body->points[i];
                        corr_normvect->points[effect_feat_num] = normvec->points[i];
                        effect_feat_num++;
                    }
                }

                res_mean_last = total_residual / effect_feat_num;

                /*** Computation of Measurement Jacobian matrix H and measurents vector ***/

                MatrixXd Hsub(effect_feat_num, 12);
                MatrixXd Hsub_T_R_inv(12, effect_feat_num);
                VectorXd R_inv(effect_feat_num);
                VectorXd meas_vec(effect_feat_num);

                Hsub.setZero();
                Hsub_T_R_inv.setZero();
                meas_vec.setZero();

                for (int i = 0; i < effect_feat_num; i++) {
                    const PointType &laser_p = laserCloudOri->points[i];
                    V3D point_this_L(laser_p.x, laser_p.y, laser_p.z);

                    V3D point_this = state.offset_R_L_I * point_this_L + state.offset_T_L_I;
                    M3D var;
                    calcBodyVar(point_this, 0.02, 0.05, var);
                    var = state.rot_end * var * state.rot_end.transpose();
                    M3D point_crossmat;
                    point_crossmat << SKEW_SYM_MATRX(point_this);

                    /*** get the normal vector of closest surface/corner ***/
                    const PointType &norm_p = corr_normvect->points[i];
                    V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

                    R_inv(i) = 1000;
                    laserCloudOri->points[i].intensity = sqrt(R_inv(i));

                    /*** calculate the Measurement Jacobian matrix H ***/
                    if (imu_en) {
                        M3D point_this_L_cross;
                        point_this_L_cross << SKEW_SYM_MATRX(point_this_L);
                        V3D H_R_LI = point_this_L_cross * state.offset_R_L_I.transpose() * state.rot_end.transpose() *
                                     norm_vec;
                        V3D H_T_LI = state.rot_end.transpose() * norm_vec;
                        V3D A(point_crossmat * state.rot_end.transpose() * norm_vec);
                        Hsub.row(i) << VEC_FROM_ARRAY(A), norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(
                                H_R_LI), VEC_FROM_ARRAY(H_T_LI);
                    } else {
                        V3D A(point_crossmat * state.rot_end.transpose() * norm_vec);
                        Hsub.row(i) << VEC_FROM_ARRAY(A), norm_p.x, norm_p.y, norm_p.z, 0, 0, 0, 0, 0, 0;
                    }

                    Hsub_T_R_inv.col(i) = Hsub.row(i).transpose() * 1000;
                    /*** Measurement: distance to the closest surface/corner ***/
                    meas_vec(i) = -norm_p.intensity;
                }

                MatrixXd K(DIM_STATE, effect_feat_num);

                EKF_stop_flg = false;
                flg_EKF_converged = false;

                /*** Iterative Kalman Filter Update ***/

                H_T_H.block<12, 12>(0, 0) = Hsub_T_R_inv * Hsub;
                MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + state.cov.inverse()).inverse();
                K = K_1.block<DIM_STATE, 12>(0, 0) * Hsub_T_R_inv;
                auto vec = state_propagat - state;
                solution = K * meas_vec + vec - K * Hsub * vec.block<12, 1>(0, 0);

                //state update
                state += solution;

                rot_add = solution.block<3, 1>(0, 0);
                T_add = solution.block<3, 1>(3, 0);


                if ((rot_add.norm() * 57.3 < 0.01) && (T_add.norm() * 100 < 0.015))
                    flg_EKF_converged = true;

                deltaR = rot_add.norm() * 57.3;
                deltaT = T_add.norm() * 100;

                euler_cur = RotMtoEuler(state.rot_end);

                /*** Rematch Judgement ***/
                nearest_search_en = false;
                if (flg_EKF_converged || ((rematch_num == 0) && (iterCount == (NUM_MAX_ITERATIONS - 2)))) {
                    nearest_search_en = true;
                    rematch_num++;
                }

                /*** Convergence Judgements and Covariance Update ***/
                if (!EKF_stop_flg && (rematch_num >= 2 || (iterCount == NUM_MAX_ITERATIONS - 1))) {
                    if (flg_EKF_inited) {
                        /*** Covariance Update ***/
                        G.setZero();
                        G.block<DIM_STATE, 12>(0, 0) = K * Hsub;
                        state.cov = (I_STATE - G) * state.cov;
                        total_distance += (state.pos_end - position_last).norm();
                        position_last = state.pos_end;
                        if (!imu_en) {
                            geoQuat = tf::createQuaternionMsgFromRollPitchYaw
                                    (euler_cur(0), euler_cur(1), euler_cur(2));
                        } else {
                            //Publish LiDAR's pose, instead of IMU's pose
                            M3D rot_cur_lidar = state.rot_end * state.offset_R_L_I;
                            V3D euler_cur_lidar = RotMtoEuler(rot_cur_lidar);
                            geoQuat = tf::createQuaternionMsgFromRollPitchYaw
                                    (euler_cur_lidar(0), euler_cur_lidar(1), euler_cur_lidar(2));
                        }
                        VD(DIM_STATE) K_sum = K.rowwise().sum();
                        VD(DIM_STATE) P_diag = state.cov.diagonal();
                    }
                    EKF_stop_flg = true;
                }

                if (EKF_stop_flg) break;
            }

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);

            /*** add the feature points to map kdtree ***/
            map_incremental();

            kdtree_size_end = ikdtree.size();

            /***** Device starts to move, data accmulation begins. ****/
            if (!imu_en && !data_accum_start && state.pos_end.norm() > 0.05) {
                printf(BOLDCYAN "[Initialization] Movement detected, data accumulation starts.\n\n\n\n\n" RESET);
                data_accum_start = true;
                move_start_time = lidar_end_time;
            }

            /******* Publish points *******/
            if (scan_pub_en || pcd_save_en) publish_frame_world(pubLaserCloudFullRes);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFullRes_body);
            last_odom = state.pos_end;
            last_rot = state.rot_end;
            publish_effect_world(pubLaserCloudEffect);
            if (path_en) publish_path(pubPath);
            //publish_mavros(mavros_pose_publisher);

            frame_num++;
            V3D ext_euler = RotMtoEuler(state.offset_R_L_I);
            fout_out << euler_cur.transpose() * 57.3 << " " << state.pos_end.transpose() << " "
                     << ext_euler.transpose() * 57.3 << " " \
                     << state.offset_T_L_I.transpose() << " " << state.vel_end.transpose() << " "  \
                     << " " << state.bias_g.transpose() << " " << state.bias_a.transpose() * 0.9822 / 9.81 << " "
                     << state.gravity.transpose() << " " << total_distance << endl;

            //Broadcast every second
            if (imu_en && frame_num % orig_odom_freq * cut_frame_num == 0 && !online_calib_finish) {
                double online_calib_completeness = lidar_end_time - online_calib_starts_time;
                online_calib_completeness =
                        online_calib_completeness < online_refine_time ? online_calib_completeness : online_refine_time;
                cout << "\x1B[2J\x1B[H"; //clear the screen
                if(online_refine_time > 0.1)
                    printProgress(online_calib_completeness / online_refine_time);
                if (!refine_print && online_calib_completeness > (online_refine_time - 1e-6)) {
                    refine_print = true;
                    online_calib_finish = true;
                    cout << endl;
                    print_refine_result();
                    fout_result << "Refinement result:" << endl;
                    fileout_calib_result();
                    string path = ros::package::getPath("lidar_imu_init");
                    path += "/result/Initialization_result.txt";
                    cout << endl  << "Initialization and refinement result is written to " << endl << BOLDGREEN << path << RESET <<endl;
                }
            }


            if (!imu_en && !data_accum_finished && data_accum_start) {
                //Push Lidar's Angular velocity and linear velocity
                Init_LI->push_Lidar_CalibState(state.rot_end, state.bias_g, state.vel_end, lidar_end_time);
                //Data Accumulation Sufficience Appraisal
                data_accum_finished = Init_LI->data_sufficiency_assess(Jaco_rot, frame_num, state.bias_g,
                                                                       orig_odom_freq, cut_frame_num);

                if (data_accum_finished) {
                    Init_LI->LI_Initialization(orig_odom_freq, cut_frame_num, timediff_imu_wrt_lidar, move_start_time);

                    online_calib_starts_time = lidar_end_time;

                    //Transfer to FAST-LIO2
                    imu_en = true;
                    state.offset_R_L_I = Init_LI->get_R_LI();
                    state.offset_T_L_I = Init_LI->get_T_LI();
                    state.pos_end = -state.rot_end * state.offset_R_L_I.transpose() * state.offset_T_L_I +
                                    state.pos_end; //Body frame is IMU frame in FAST-LIO mode
                    state.rot_end = state.rot_end * state.offset_R_L_I.transpose();
                    state.gravity = Init_LI->get_Grav_L0();
                    state.bias_g = Init_LI->get_gyro_bias();
                    state.bias_a = Init_LI->get_acc_bias();


                    if (lidar_type != AVIA)
                        cut_frame_num = 2;

                    time_lag_IMU_wtr_lidar = Init_LI->get_total_time_lag(); //Compensate IMU's time in the buffer
                    for (int i = 0; i < imu_buffer.size(); i++) {
                        imu_buffer[i]->header.stamp = rclcpp::Time().fromSec(imu_buffer[i]->header.stamp.sec - time_lag_IMU_wtr_lidar);
                    }

                    p_imu->imu_en = imu_en;
                    p_imu->LI_init_done = true;
                    p_imu->set_mean_acc_norm(mean_acc_norm);
                    p_imu->set_gyr_cov(V3D(0.1, 0.1, 0.1));
                    p_imu->set_acc_cov(V3D(0.1, 0.1, 0.1));
                    p_imu->set_gyr_bias_cov(V3D(0.0001, 0.0001, 0.0001));
                    p_imu->set_acc_bias_cov(V3D(0.0001, 0.0001, 0.0001));

                    //Output Initialization result
                    fout_result << "Initialization result:" << endl;
                    fileout_calib_result();
                }
            }
        }
        // status = ros::ok();
        // rate.sleep();

    // cout << endl << REDPURPLE << "[Exit]: Exit the process." <<RESET <<endl;
    // if (!online_calib_finish) {
    //     cout << YELLOW << "[WARN]: Online refinement not finished yet." << RESET;
    //     print_refine_result();
    // }
            // Brief sleep to avoid CPU overload
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    
    // Spin the node
    rclcpp::spin(node);
    
    processing_thread.join();
    rclcpp::shutdown();
    
    RCLCPP_INFO(node->get_logger(), "LiDAR-IMU calibration node shutdown");
    return 0;
}
