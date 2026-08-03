//
// You received this file as part of Finroc
// A framework for intelligent robot control
//
// Copyright (C) AG Robotersysteme TU Kaiserslautern
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
//
//----------------------------------------------------------------------
/*!\file    libraries/human_perception/zed/mFaceIdentification.cpp
 *
 * \author  Ashita Ashok, Zhijing Lu
 *
 * \date    2023-01-06
 *
 */
//----------------------------------------------------------------------
#include "libraries/human_perception/zed/mFaceIdentification.h"

//----------------------------------------------------------------------
// External includes (system with <>, local with "")
//----------------------------------------------------------------------
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=format"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop
#else
#include <opencv2/opencv.hpp>
#endif

#include <experimental/filesystem>
#include <boost/filesystem/operations.hpp>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <filesystem>
//----------------------------------------------------------------------
// Internal includes with ""
//----------------------------------------------------------------------
#include "rrlib/coviroa/opencv_utils.h"
#include "core/file_lookup.h"
#include <cmath>
#include "rrlib/util/fileio.h"
//----------------------------------------------------------------------
// Debugging
//----------------------------------------------------------------------
#include <cassert>

//----------------------------------------------------------------------
// Namespace usage
//----------------------------------------------------------------------

//----------------------------------------------------------------------
// Namespace declaration
//----------------------------------------------------------------------
namespace finroc
{
namespace human_perception
{
namespace zed
{

//----------------------------------------------------------------------
// Forward declarations / typedefs / enums
//----------------------------------------------------------------------

//----------------------------------------------------------------------
// Const values
//----------------------------------------------------------------------
#ifdef _LIB_FINROC_PLUGINS_RUNTIME_CONSTRUCTION_ACTIONS_PRESENT_
static const runtime_construction::tStandardCreateModuleAction<mFaceIdentification> cCREATE_ACTION_FOR_M_FACEIDENTIFICATION("FaceIdentification");
#endif

//----------------------------------------------------------------------
// Implementation
//----------------------------------------------------------------------

//----------------------------------------------------------------------
// mFaceIdentification constructor
//----------------------------------------------------------------------
mFaceIdentification::mFaceIdentification(core::tFrameworkElement *parent, const std::string &name) :
  tModule(parent, name, false),
  encoder_session_(nullptr),
  last_recorded_time(std::chrono::steady_clock::now())
{
  faceDetector.load(CASCADE_CLASSIFIER_FILE_PATH);
  env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
  Ort::SessionOptions sessOptions;
  sessOptions.SetIntraOpNumThreads(1);
  sessOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  encoder_session_ = Ort::Session(env, FACE_ID_MODEL_PATH.c_str(), sessOptions);

}

//----------------------------------------------------------------------
// mFaceIdentification destructor
//----------------------------------------------------------------------
mFaceIdentification::~mFaceIdentification()
{}

//----------------------------------------------------------------------
// mFaceIdentification OnStaticParameterChange
//----------------------------------------------------------------------
void mFaceIdentification::OnStaticParameterChange()
{
}

//----------------------------------------------------------------------
// mFaceIdentification OnParameterChange
//----------------------------------------------------------------------
void mFaceIdentification::OnParameterChange()
{

}

//----------------------------------------------------------------------
// mFaceIdentification Update
//----------------------------------------------------------------------
void mFaceIdentification::Update()
{
  if (this->enable_face_id.Get())
  {
    //std::cout << "enable_face_id entered" << std::endl;
    if (in_color_images.HasChanged())
    {
      //std::cout << "received incolorimage entered" << std::endl;
      auto now = std::chrono::steady_clock::now();
      if (std::chrono::duration_cast<std::chrono::seconds>(now - last_recorded_time).count() >= 5)
      {
        //variable declaration
        data_ports::tPortDataPointer<const std::vector<rrlib::coviroa::tImage>> in_orig_img = this->in_color_images.GetPointer();
        std::vector<float> out_tensor_unknown;
        std::vector<float> out_tensor_known;
        float sim_val;
        float max_sim_val = 0;
        int max_sim_val_index = -1;

        cv::Mat prepro_input;

        //Capture new image via ZED
        cv::Mat frame = this->CaptureZedImage(in_orig_img);

        //Publish detected image to finroc port
        this->PublishImage(in_orig_img);

        //Detect face ROI in image
        std::vector<cv::Mat> face_images = this->DetectFaces(frame);

        if (face_images.size() > 0)
        {

          auto in = (face_images.at(0) - 0.5) / 0.5;
          out_tensor_unknown = this->FaceModel(in);

          boost::filesystem::path dir(KNOWN_IMAGE_PATH);
          std::vector<cv::String> image_names;
          if (boost::filesystem::is_directory(dir))
          {
            for (auto & entry : boost::make_iterator_range(boost::filesystem::directory_iterator(dir), { }))
            {
              std::string file_path = entry.path().string();
              image_names.push_back(entry.path().filename().string());
            }
          }
          size_t count = image_names.size();
          std::cout << "Count:" << count << std::endl;
          for (size_t i = 0; i < count; i++)
          {
            cv::Mat img = cv::imread(dir.string() + "/" + image_names[i]);
            if (!img.empty())
            {
              img -= 0.5;
              img /= 0.5;
              auto tensor_known = (this->FaceModel(img));
              sim_val = this->CosineSimilarity(out_tensor_unknown, tensor_known);
              if (sim_val > max_sim_val)
              {
                max_sim_val = sim_val;
                max_sim_val_index = i;
              }
            }
          }
          if (max_sim_val_index != -1)
          {
            std::filesystem::path file_path_fs(image_names[max_sim_val_index]);
            std::string file_base = file_path_fs.stem().string();
            std::cout << "The face id detects: " << file_base << std::endl;
            detected_face_id.Publish(file_base);
          }
          else
          {
            std::cout << "No match found" << std::endl;
          }
          last_recorded_time = now;
        }
      }
    }
  }
}

cv::Mat mFaceIdentification::CaptureZedImage(data_ports::tPortDataPointer<const std::vector<rrlib::coviroa::tImage>>& in_orig_img)
{
  //FINROC_LOG_PRINT(DEBUG, "Entered Capture");
  cv::Mat frame = rrlib::coviroa::AccessImageAsMat(in_orig_img->at(0));
  return frame;
}

std::vector<cv::Mat> mFaceIdentification::DetectFaces(cv::Mat & frame)
{
  //FINROC_LOG_PRINT(DEBUG, "Entered DetectFaces");
  std::vector <cv::Rect> faces;
  faceDetector.detectMultiScale(frame, faces, 1.1, 4, cv::CASCADE_SCALE_IMAGE, cv::Size(112, 112));

  int face_size = faces.size();
  //std::cout << "Detected faces : " << face_size << std::endl;
  std::vector<cv::Mat> faceROIs;

  // Get the current time in milliseconds
  auto now = std::chrono::steady_clock::now();
  auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
  auto value = now_ms.time_since_epoch();
  long duration = value.count();

  for (int i = 0; i < face_size; i++)
  {
    cv::Rect roi = faces[i];
    cv::Mat faceROI = frame(roi);

    cv::Mat resized_frame;
    cv::resize(faceROI, resized_frame, cv::Size(112, 112), 0, 0, cv::INTER_AREA);
    cv::cvtColor(resized_frame, resized_frame, cv::COLOR_BGRA2BGR);
    faceROIs.push_back(resized_frame);

    // Use both time and index to create a unique filename
    std::string imagefacename = INTERIM_CURRENT_IMAGE_PATH + std::to_string(duration) + "_" + std::to_string(i) + ".jpg";
    cv::imwrite(imagefacename, resized_frame);
  }
  return faceROIs;

}

void mFaceIdentification::PublishImage(data_ports::tPortDataPointer<const std::vector<rrlib::coviroa::tImage>>& in_orig_img)
{
  // prepare a vector for visualization images
  data_ports::tPortDataPointer<rrlib::coviroa::tImage> out_face_buff = this->out_face_image.GetUnusedBuffer();
  // visualization
  rrlib::rtti::GenericOperations<rrlib::coviroa::tImage>::DeepCopy(in_orig_img->at(0), *out_face_buff);
  out_face_image.Publish(out_face_buff);

}

std::vector<float> mFaceIdentification::FaceModel(const cv::Mat & face_image)
{
  cv::Mat input_mat = face_image;
  //FINROC_LOG_PRINT(DEBUG, "Entered Predict FaceModel");
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // Ensure the input image is in the correct format and size
  if (input_mat.channels() != 3)
  {
    //FINROC_LOG_PRINT(ERROR, "Input image must have 3 channels (BGR)");
    return std::vector<float>();
  }
  if (input_mat.rows != 112 || input_mat.cols != 112)
  {
    //FINROC_LOG_PRINT(ERROR, "Input image must be of size 112x112");
    return std::vector<float>();
  }

  // Normalize the input image
  cv::Mat input_float;
  input_mat.convertTo(input_float, CV_32FC3, 1.0 / 255.0);
  input_float = input_float.reshape(1, 3 * 112 * 112);

  // Create the input tensor
  std::array<int64_t, 4> input_shape_ {1, 3, 112, 112};
  std::vector<Ort::Value> inputs;
  inputs.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo,
                      input_float.ptr<float>(),
                      3 * 112 * 112,
                      input_shape_.data(),
                      input_shape_.size()));

  // Run the encoder session
  Ort::Value encoder_output_tensor_ {nullptr};
  std::array<const char*, 1> encoder_input_names = {"input"};
  std::array<const char*, 1> encoder_output_names = {"output"};
  encoder_session_.Run(Ort::RunOptions {nullptr},
                       encoder_input_names.data(),
                       inputs.data(),
                       1,
                       encoder_output_names.data(),
                       &encoder_output_tensor_,
                       1);

  // Get the output tensor as cv::Mat
  float* out_ptr = encoder_output_tensor_.GetTensorMutableData<float>();
  std::vector<float> output_vector(out_ptr, out_ptr + encoder_output_tensor_.GetTensorTypeAndShapeInfo().GetElementCount());
//  std::for_each(output_vector.begin(), output_vector.end(), [](float x)
//  {
//    std::cout << x << " ";
//  });
  return output_vector;
}


float  mFaceIdentification::CosineSimilarity(std::vector<float> unknown, std::vector<float> known)
{
  float dot_product = 0.0;
  float magnitude1 = 0.0;
  float magnitude2 = 0.0;

  for (size_t i = 0; i < unknown.size(); i++)
  {
    dot_product += unknown[i] * known[i];
    magnitude1 += unknown[i] * unknown[i];
    magnitude2 += known[i] * known[i];
  }

  float cosine_similarity = dot_product / (sqrt(magnitude1) * sqrt(magnitude2));
  return cosine_similarity;
}

//----------------------------------------------------------------------
// End of namespace declaration
//----------------------------------------------------------------------
}
}
}
