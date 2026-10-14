#include "LPRCamera.h"
#include <godot_cpp/classes/engine.hpp>
#include <iostream>
#include <opencv2/imgproc/types_c.h>


using namespace godot;


Ref<CoreBind::Mutex> g_mutex_cpu;
Ref<CoreBind::Mutex> g_mutex_gpu;




void LPRCamera::_bind_methods()
{
	ADD_SIGNAL(MethodInfo("plate_detected", PropertyInfo(Variant::ARRAY, "result")));
	ADD_SIGNAL(MethodInfo("camera_open_failed"));
	ADD_SIGNAL(MethodInfo("camera_opened"));


	ClassDB::bind_method(D_METHOD("get_frame"), &LPRCamera::GetFrame);
	ClassDB::bind_method(D_METHOD("get_fps"), &LPRCamera::GetFps);
	ClassDB::bind_method(D_METHOD("release"), &LPRCamera::Release);
	ClassDB::bind_method(D_METHOD("is_opened"), &LPRCamera::IsOpened);

	ClassDB::bind_method(D_METHOD("set_camera_id", "camera_id"), &LPRCamera::SetCameraId);
	ClassDB::bind_method(D_METHOD("get_camera_id"), &LPRCamera::GetCameraId);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "camera_id"), "set_camera_id", "get_camera_id");

	ClassDB::bind_method(D_METHOD("set_stream_url", "stream_url"), &LPRCamera::SetStreamUrl);
	ClassDB::bind_method(D_METHOD("get_stream_url"), &LPRCamera::GetStreamUrl);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "stream_url", PROPERTY_HINT_LINK), "set_stream_url", "get_stream_url");

	ClassDB::bind_method(D_METHOD("set_default_texture", "default_texture"), &LPRCamera::SetDefaultTexture);
	ClassDB::bind_method(D_METHOD("get_default_texture"), &LPRCamera::GetDefaultTexture);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "default_texture", PROPERTY_HINT_NONE, "", 6U,
		"Texture2D"), "set_default_texture", "get_default_texture");
	
	ClassDB::bind_method(D_METHOD("set_disable_lpr", "disable_lpr"), &LPRCamera::SetDisableLPR);
	ClassDB::bind_method(D_METHOD("get_disable_lpr"), &LPRCamera::GetDisableLPR);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "disable_lpr"), "set_disable_lpr", "get_disable_lpr");

	ClassDB::bind_method(D_METHOD("set_model", "model_path"), &LPRCamera::SetModelPath);
	ClassDB::bind_method(D_METHOD("get_model"), &LPRCamera::GetModelPath);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "model_path", PROPERTY_HINT_FILE), "set_model", "get_model");

	ClassDB::bind_method(D_METHOD("set_model_score_threshold", "model_score_threshold"), &LPRCamera::SetModelScoreThreshold);
	ClassDB::bind_method(D_METHOD("get_model_score_threshold"), &LPRCamera::GetModelScoreThreshold);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "model_score_threshold", PROPERTY_HINT_RANGE, "0,1,0.01"),
		"set_model_score_threshold", "get_model_score_threshold");

	ClassDB::bind_method(D_METHOD("set_model_nms_threshold", "model_nms_threshold"), &LPRCamera::SetModelNMSThreshold);
	ClassDB::bind_method(D_METHOD("get_model_nms_threshold"), &LPRCamera::GetModelNMSThreshold);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "model_nms_threshold", PROPERTY_HINT_RANGE, "0,1,0.01"),
		"set_model_nms_threshold", "get_model_nms_threshold");

	ClassDB::bind_method(D_METHOD("set_enable_cuda", "enable_cuda"), &LPRCamera::SetEnableCuda);
	ClassDB::bind_method(D_METHOD("get_enable_cuda"), &LPRCamera::GetEnableCuda);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable_cuda"), "set_enable_cuda", "get_enable_cuda");

	ClassDB::bind_method(D_METHOD("set_tessdata", "tessdata"), &LPRCamera::SetTessData);
	ClassDB::bind_method(D_METHOD("get_tessdata"), &LPRCamera::GetTessData);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "tessdata", PROPERTY_HINT_FILE), "set_tessdata", "get_tessdata");

	ClassDB::bind_method(D_METHOD("set_max_fps", "max_fps"), &LPRCamera::SetMaxFps);
	ClassDB::bind_method(D_METHOD("get_max_fps"), &LPRCamera::GetMaxFps);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_fps"), "set_max_fps", "get_max_fps");

}

void LPRCamera::_notification(int pWhat)
{
	if (Engine::get_singleton()->is_editor_hint()) return;
	
	switch (pWhat)
	{
		case NOTIFICATION_READY:
		{
			if (!disableLPR)
			{
				InitNet();
				InitOCR();
			}
			isWorking = false;
			g_mutex_cpu.instantiate();
			g_mutex_gpu.instantiate();
			worker.instantiate();
			this->connect("visibility_changed", callable_mp(this, &LPRCamera::VisibilityChangedHandle));
		} break;
		case NOTIFICATION_EXIT_TREE:
		{
			if (!disableLPR)
			{
				DeleteNet();
				DeleteOCR();
			}
			isWorking = false;
			if (worker->is_started())
				worker->wait_to_finish();
			worker.unref();
			Release();
		} break;
	}
}


void LPRCamera::VisibilityChangedHandle()
{
	if (is_visible())
	{
		if (!capture.isOpened())
		{
			OpenCameraInThread();
		}
		if (!disableLPR)
		{
			InitNet();
			InitOCR();
		}
	}
	else
	{
		isWorking = false;
		if (worker->is_alive())
			worker->wait_to_finish();
		if (capture.isOpened())
		{
			capture.release();
		}
		if (!disableLPR)
		{
			DeleteOCR();
		}
	}
}


void LPRCamera::ProcessWorker()
{
	bool isDefault = true;
	int64 startTime, total = 0;
	int durationMs, frameCount = 0;
	call_deferred("set_texture", defaultTexture);
	while (isWorking)
	{
		try
		{
			startTime = static_cast<double>(cv::getTickCount());
			if (isOpenCameraInThread)
				OpenCameraInThreadWorker();
			if (!capture.read(frame) || frame.empty())
			{
				if (!isDefault)
				{
					call_deferred("set_texture", defaultTexture);
					isDefault = true;
				}
				std::this_thread::sleep_for(std::chrono::seconds(1));
				continue;
			}
			isDefault = false;
			call_deferred("set_texture", ImageTexture::create_from_image(cvMatToImage(frame)));
			if (isDetect)
			{
				if (cv::getTickCount() - startTickOut < detectTickOut)
					isDetect = !Detect(frame);
				else
				{
					isDetect = false;
					call_deferred("emit_signal", "plate_detected", Array());
				}
			}
			else if (autoDetect && !disableLPR)
				Detect(frame);

			durationMs = (int)((cv::getTickCount() - startTime) * 1000 / cv::getTickFrequency());
			if (durationMs < T)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(T - durationMs));
				fps = maxFps;
				total = 0;
				frameCount = 0;
			}
			else
			{
				total += durationMs;
				frameCount++;
				if (frameCount >= 10)
				{
					fps = frameCount * 1000 / total;
					total = 0;
					frameCount = 0;
				}
					
			}
		}
		catch (std::exception& e)
		{
			print_error(this->to_string(), e.what());
		}
	}
}



LPRCamera::LPRCamera()
{
}

LPRCamera::~LPRCamera()
{
}


void LPRCamera::SetDefaultTexture(Ref<Texture2D> newTexture)
{
	defaultTexture = newTexture;
}


Ref<Texture2D> LPRCamera::GetDefaultTexture()
{
	return defaultTexture;
}


void LPRCamera::SetDisableLPR(bool value)
{
	disableLPR = value;
}

bool LPRCamera::GetDisableLPR()
{
	return disableLPR;
}


int LPRCamera::GetFps()
{
	if (!capture.isOpened() || frame.empty()) {
		return 0;
	}
	return fps;
}


void LPRCamera::SetCameraId(int newId)
{
	if (Engine::get_singleton()->is_editor_hint())
	{
		cameraId = newId;
		streamUrl = "";
		return;
	}

	if (newId >= 0 && newId != cameraId)
	{
		Release();
		cameraId= newId;
		OpenCameraInThread();
	}
	else
	{
		Release();
	}
}


int LPRCamera::GetCameraId()
{
    return cameraId;
}


void LPRCamera::SetStreamUrl(String newUrl)
{
	if (Engine::get_singleton()->is_editor_hint())
	{
		streamUrl = newUrl;
		cameraId = -1;
		return;
	}

	if (newUrl != streamUrl)
	{
		Release();
		streamUrl = newUrl;
		OpenCameraInThread();
	}
	else if (newUrl == "")
	{
		Release();
	}
}


String LPRCamera::GetStreamUrl()
{
	return streamUrl;
}


void LPRCamera::OpenCameraInThread()
{
	isOpenCameraInThread = true;
	if (!isWorking)
	{
		if (worker->is_started())
			worker->wait_to_finish();
		isWorking = true;
		worker->start(callable_mp(this, &LPRCamera::ProcessWorker));
	}
}

void LPRCamera::OpenCameraInThreadWorker()
{
	isOpenCameraInThread = false;
	if (capture.isOpened())
		capture.release();
	if (cameraId >= 0)
	{
		if (!capture.open(cameraId))
		{
			call_deferred("emit_signal", "camera_open_failed");
			isWorking = false;
			print_error(this->to_string(), "Camera opend failed! cameraId: \"", cameraId, "\"");
		}
		else
		{
			call_deferred("emit_signal", "camera_opened");
			isWorking = true;
		}
	}
	else if (streamUrl != "")
	{
		if (!capture.open(streamUrl.utf8().get_data()))
		{
			call_deferred("emit_signal", "camera_open_failed");
			isWorking = false;
			print_error(this->to_string(), "Camera opend failed! Stream URL: \"", streamUrl, "\"");
		}
		else
		{
			call_deferred("emit_signal", "camera_opened");
			isWorking = true;
		}
	}
}

void LPRCamera::DetectNumberPlate(float timeout)
{
	isDetect = true;
	detectTickOut = (double)(1000 * timeout) * cv::getTickFrequency();
	startTickOut = cv::getTickCount();
}

void LPRCamera::SetAutoDetect(bool newValue)
{
	autoDetect = newValue;
}

bool LPRCamera::GetAutoDetect()
{
	return autoDetect;
}

Ref<Image> LPRCamera::cvMatToImage(cv::Mat& mat)
{
	try
	{
		if (mat.empty())
		{
			print_error(this->to_string(), "cvMatToImage input is empty!");
			return Ref<Image>();
		}
		cv::Mat rgb;
		if (cudaEnable)
		{
			cv::cuda::GpuMat gpuImg, gpuRGB;
			gpuImg.upload(mat);
			cv::cuda::cvtColor(gpuImg, gpuRGB, cv::COLOR_BGR2RGB);
			gpuRGB.download(rgb);
		}
		else 
		{
			cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
		}
		
		int num = rgb.rows * rgb.cols * rgb.channels();
		PackedByteArray bytes;
		bytes.resize(num);
		if (rgb.isContinuous())
		{
			std::memcpy(bytes.ptrw(), rgb.ptr(), num);
		}
		else
		{
			for (int i = 0; i < sizeof(rgb); i++)
			{
				std::memcpy(bytes.ptrw(), rgb.ptr(i), num);
			}
		}
		return Image::create_from_data(rgb.cols, rgb.rows, false, Image::FORMAT_RGB8, bytes);
	}
	catch (std::exception& e)
	{
		print_error(to_string(), e.what());
	}
	catch (...)
	{
		print_error(to_string(), "Unknown or disable exceptions");
	}
	return Ref<Image>();
	
}


Ref<Image> LPRCamera::GetFrame()
{
	return cvMatToImage(frame);
}


void LPRCamera::Release()
{
	capture.release();
	frame.release();
	cameraId= -1;
	streamUrl = "";
}


bool LPRCamera::IsOpened()
{
	return capture.isOpened();
}



void LPRCamera::SetModelPath(String newPath)
{
	if (modelPath != newPath)
	{
		modelPath = newPath;
		InitNet();
	}
}

String LPRCamera::GetModelPath()
{
	return modelPath;
}


void LPRCamera::SetModelScoreThreshold(float newValue)
{
	modelScoreThreshold = newValue;
}

float LPRCamera::GetModelScoreThreshold()
{
	return modelScoreThreshold;
}

void LPRCamera::SetModelNMSThreshold(float newValue)
{
	modelNMSThreshold = newValue;
}

float LPRCamera::GetModelNMSThreshold()
{
	return modelNMSThreshold;
}

void LPRCamera::SetEnableCuda(bool newValue)
{
	if (cudaEnable != newValue)
	{
		cudaEnable = newValue;
		InitNet();
	}
}

bool LPRCamera::GetEnableCuda()
{
	return cudaEnable;
}


void LPRCamera::SetTessData(String newPath)
{
	tessData = newPath;
}


String LPRCamera::GetTessData()
{
	return tessData;
}


void LPRCamera::SetMaxFps(int value)
{
	maxFps = value;
	T = (int)(1000 / maxFps);
}

int LPRCamera::GetMaxFps()
{
	return maxFps;
}

void LPRCamera::InitNet()
{
	if (netInitializated)
		return;
	net = cv::dnn::readNetFromONNX(modelPath.utf8().get_data());
	if (cudaEnable)
	{
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	}
	else
	{
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
	netInitializated = true;
}

void LPRCamera::DeleteNet()
{
	netInitializated = false;
}

bool LPRCamera::Detect(cv::Mat& imgInput)
{
	if (imgInput.empty()) return false;
	cv::Mat image = imgInput.clone();
	static cv::Mat blob;
	static std::vector<cv::Mat> outputs;
	if (cudaEnable) g_mutex_gpu->lock(); else g_mutex_cpu->lock();
	cv::dnn::blobFromImage(image, blob, 1.0 / 255.0, modelShape, cv::Scalar(), true, false);
	net.setInput(blob);
	net.forward(outputs, net.getUnconnectedOutLayersNames());
	if (cudaEnable) g_mutex_gpu->unlock(); else g_mutex_cpu->unlock();
	int rows = outputs[0].size[2];
	int dimensions = outputs[0].size[1];

	outputs[0] = outputs[0].reshape(1, dimensions);
	if (cudaEnable)
	{
		cv::cuda::GpuMat gpuMat(outputs[0]);
		cv::cuda::transpose(gpuMat, gpuMat);
		gpuMat.download(outputs[0]);
	}
	else
	{
		cv::transpose(outputs[0], outputs[0]);
	}

	float* data = (float*)outputs[0].data;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;

	for (int i = 0; i < rows; ++i) {
		float* classesScores = data + 4;

		cv::Mat scores(1, classesCount, CV_32FC1, classesScores);
		cv::Point classId;
		double maxClassScore;

		minMaxLoc(scores, nullptr, &maxClassScore, nullptr, &classId);

		if (maxClassScore > modelScoreThreshold && maxClassScore < 1) {
			float x_factor = (float)image.cols / (float)modelShape.width;
			float y_factor = (float)image.rows / (float)modelShape.height;
			int left = int((data[0] - 0.5 * data[2]) * x_factor);
			int top = int((data[1] - 0.5 * data[3]) * y_factor);
			int width = int(data[2] * x_factor);
			int height = int(data[3] * y_factor);

			if (0 <= left && 0 <= width && left + width <= image.cols && 0 <= top && 0 <= height && top + height <= image.rows)
			{
				confidences.push_back(float(maxClassScore));
				boxes.push_back(cv::Rect(left, top, width, height));
			}
		}
		data += dimensions;
	}
	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);
	Array result = Array();
	for (int idx : nms_result)
	{
		cv::Mat cropImg = image(boxes[idx]);
		if (cropImg.empty()) continue;
		cv::Mat reImg;
		char* text = OCR(cropImg, reImg);
		Dictionary dicData;
		dicData.set("text", text);
		dicData.set("image", cvMatToImage(cropImg));
		dicData.set("score", confidences[idx]);
		result.append(dicData);
	}
	if (result.size() > 0)
	{
		call_deferred("emit_signal", "plate_detected", result);
		return true;
	}
	return false;
}


void LPRCamera::InitOCR()
{
	if (ocrInitializated)
		return;
	if (tessData.is_empty())
	{
		print_error(this->to_string(), "tessdata is empty");
		return;
	}
	else
	{
		std::string path, file, dir, filename;
		path = std::string(tessData.utf8().get_data());
		size_t pos = path.find_last_of("\\/");
		dir = (std::string::npos == pos) ? "" : path.substr(0, pos);
		file = path.substr(pos + 1);
		filename = file.substr(0, file.find_last_of('.'));
		if (ocrAPI->Init((dir == "") ? NULL : dir.c_str(), filename.c_str(), tesseract::OEM_DEFAULT))
		{
			print_error(this->to_string(), "Could not initialize tesseract OCR with data path \"", tessData, " \"");
			ocrInitializated = false;
			return;
		}
	}
	ocrInitializated = true;
	ocrAPI->SetVariable("tessedit_char_whitelist", "ABCDEFHIJKLMNOPQRSTUVWXYZ0123456789-.");
	ocrAPI->SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
}


void LPRCamera::DeleteOCR()
{
	if (ocrInitializated)
	{
		ocrAPI->Clear();
		ocrAPI->End();
		delete ocrAPI;
		ocrInitializated = false;
	}
}


char *LPRCamera::OCR(cv::Mat& input, cv::Mat& reInput)
{
	cv::Mat image = input.clone();
	cv::Mat gray, blurred, edged;
	cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
	cv::Canny(blurred, edged, 75, 200);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(edged.clone(), contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
	std::sort(contours.begin(), contours.end(),
		[](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2) {
			return cv::contourArea(c1, false) > cv::contourArea(c2, false);
		});

	std::vector<cv::Point> screenCnt;
	for (size_t i = 0; i < std::min(contours.size(), (size_t)5); i++) {
		double peri = cv::arcLength(contours[i], true);
		std::vector<cv::Point> approx;
		cv::approxPolyDP(contours[i], approx, 0.02 * peri, true);
		if (approx.size() == 4) {
			screenCnt = approx;
			break;
		}
	}

	if (!screenCnt.empty()) {
		std::vector<cv::Point2f> pts;
		for (auto& p : screenCnt) pts.push_back(cv::Point2f(p.x, p.y));
		std::vector<cv::Point2f> rect(4);
		std::vector<float> s, diff;
		for (auto& p : pts) {
			s.push_back(p.x + p.y);
			diff.push_back(p.x - p.y);
		}
		
		rect[0] = pts[min_element(s.begin(), s.end()) - s.begin()];
		rect[2] = pts[max_element(s.begin(), s.end()) - s.begin()];
		rect[1] = pts[max_element(diff.begin(), diff.end()) - diff.begin()];
		rect[3] = pts[min_element(diff.begin(), diff.end()) - diff.begin()];
		cv::Point2f tl = rect[0], tr = rect[1], br = rect[2], bl = rect[3];
		
		float widthA = norm(br - bl);
		float widthB = norm(tr - tl);
		int maxWidth = (int)std::max(widthA, widthB);

		float heightA = norm(tr - br);
		float heightB = norm(tl - bl);
		int maxHeight = (int)std::max(heightA, heightB);
		std::vector<cv::Point2f> dst = {
			cv::Point2f(0, 0),
			cv::Point2f(maxWidth - 1, 0),
			cv::Point2f(maxWidth - 1, maxHeight - 1),
			cv::Point2f(0, maxHeight - 1)
		};

		cv::Mat M = cv::getPerspectiveTransform(rect, dst);
		cv::Mat warped, warped_gray;
		cv::warpPerspective(image, warped, M, cv::Size(maxWidth, maxHeight));
		//if (maxHeight > maxWidth)
		warped.copyTo(reInput);
		if (!ocrInitializated) return "";
		cvtColor(warped, warped_gray, cv::COLOR_BGR2GRAY);
		ocrAPI->SetImage(warped_gray.data, warped_gray.cols, warped_gray.rows, warped_gray.channels(), warped_gray.step1());
		ocrAPI->Recognize(0);
		return ocrAPI->GetUTF8Text();
	}
	return "";
}