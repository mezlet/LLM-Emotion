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
/*!\file    libraries/human_perception/zed/mFaceIdentification.h
 *
 * \author  Ashita Ashok, Zhijing Lu
 *
 * \date    2023-01-06
 *
 * \brief Contains mFaceIdentification
 *
 * \b mFaceIdentification
 *
 * This module identifies a person based on face data.
 *
 */
//----------------------------------------------------------------------
#ifndef __libraries__human_perception__zed__mFaceIdentification_h__
#define __libraries__human_perception__zed__mFaceIdentification_h__

#include "plugins/structure/tModule.h"

//----------------------------------------------------------------------
// External includes (system with <>, local with "")
//----------------------------------------------------------------------
#include <include/onnxruntime/core/session/onnxruntime_cxx_api.h>
#include "core/file_lookup.h"
//----------------------------------------------------------------------
// Internal includes with ""
//----------------------------------------------------------------------
#include "rrlib/coviroa/tImage.h"
#include <opencv2/face/facemark.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
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
static const std::string CASCADE_CLASSIFIER_FILE_PATH = "/mnt/public/learning/trained_models/onnx/image_based/face_id/haarcascade_frontalface_alt.xml";
static const std::string FACE_ID_MODEL_PATH = "/mnt/public/learning/trained_models/onnx/image_based/face_id/faceReID.onnx";
static const std::string KNOWN_IMAGE_PATH = "/mnt/public/learning/trained_models/onnx/image_based/face_id/knownDB/";
static const std::string INTERIM_CURRENT_IMAGE_PATH = finroc::core::GetFinrocFile("libraries/human_perception/zed/include/face_id/captured/");
//----------------------------------------------------------------------
// Class declaration
//----------------------------------------------------------------------
//! SHORT_DESCRIPTION
/*!
 * This module identifies a person based on face data.
 */
class mFaceIdentification : public structure::tModule
{

//----------------------------------------------------------------------
// Ports (These are the only variables that may be declared public)
//----------------------------------------------------------------------
public:
  tInput<bool> enable_face_id;
  tInput<std::vector<rrlib::coviroa::tImage>> in_color_images;
  tOutput<rrlib::coviroa::tImage> out_face_image;
  tOutput<std::string> detected_face_id;
//----------------------------------------------------------------------
// Public methods and typedefs
//----------------------------------------------------------------------
public:

  mFaceIdentification(core::tFrameworkElement *parent, const std::string &name = "FaceIdentification");

//----------------------------------------------------------------------
// Protected methods
//----------------------------------------------------------------------
protected:

  /*! Destructor
   *
   * The destructor of modules is declared protected to avoid accidental deletion. Deleting
   * modules is already handled by the framework.
   */
  virtual ~mFaceIdentification();

//----------------------------------------------------------------------
// Private fields and methods
//----------------------------------------------------------------------
private:

  Ort::Env env;
  Ort::Session encoder_session_;
  Ort::AllocatorWithDefaultOptions ortAllocator;
//
  cv::CascadeClassifier faceDetector;

  std::chrono::steady_clock::time_point last_recorded_time;

  virtual void OnStaticParameterChange() override;

  virtual void OnParameterChange() override;

  virtual void Update() override;

  cv::Mat CaptureZedImage(data_ports::tPortDataPointer<const std::vector<rrlib::coviroa::tImage>>& in_orig_img);

  std::vector<cv::Mat> DetectFaces(cv::Mat& frame);

  void PublishImage(data_ports::tPortDataPointer<const std::vector<rrlib::coviroa::tImage>>& in_orig_img);

  std::vector<float> FaceModel(const cv::Mat& prepro_input);

  float CosineSimilarity(std::vector<float> unknown, std::vector<float> known);

};

//----------------------------------------------------------------------
// End of namespace declaration
//----------------------------------------------------------------------
}
}
}


#endif
