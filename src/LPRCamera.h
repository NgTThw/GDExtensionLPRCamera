#ifndef LPR_CAMERA_H_
#define LPR_CAMERA_H_



#include <godot_cpp/godot.hpp>
#include <godot_cpp/classes/texture_rect.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/classes/image_texture.hpp>
#include <godot_cpp/classes/thread.hpp>
#include <godot_cpp/classes/mutex.hpp>
#include <opencv2/text/erfilter.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/text/ocr.hpp>
#include <opencv2/highgui.hpp>
#include <tesseract/baseapi.h>



// global Mutex for cv::dnn::Net Thread-safe
extern godot::Ref<godot::CoreBind::Mutex> g_mutex_cpu;
extern godot::Ref<godot::CoreBind::Mutex> g_mutex_gpu;



namespace godot {

	class LPRCamera : public TextureRect
	{
		GDCLASS(LPRCamera, TextureRect)

	private:
		int cameraId = -1;
		String streamUrl = "";
		cv::VideoCapture capture = cv::VideoCapture();
		cv::Mat frame;
		Ref<Texture2D> defaultTexture;
		bool disableLPR = true;
		int fps = 0;
		int maxFps = 30;
		int T = 33;
		bool isOpenCameraInThread = false;
		bool isDetect = false;
		double detectTickOut = 0;
		double startTickOut = 0;

		bool cudaEnable = true;
		String modelPath = "license_plate_detector.onnx";
		cv::Size modelShape = cv::Size(640, 640);
		int classesCount = 1;
		float modelScoreThreshold = 0.5;
		float modelNMSThreshold = 0.5;

		String tessData = "lp.traineddata";

		cv::dnn::Net net;
		tesseract::TessBaseAPI* ocrAPI = new tesseract::TessBaseAPI();

		Ref<Thread> worker;
		bool isWorking = false;
		void ProcessWorker();
		void VisibilityChangedHandle();

		bool netInitializated = false;
		bool ocrInitializated = false;
		bool autoDetect = false;

	protected:
		static void _bind_methods();
		Ref<Image> cvMatToImage(cv::Mat& mat);
		void InitNet();
		bool Detect(cv::Mat& imgInput);
		void _notification(int pWhat);
		void DeleteNet();

		void InitOCR();
		char *OCR(cv::Mat& input, cv::Mat& reInput);
		void DeleteOCR();

		void OpenCameraInThread();
		void OpenCameraInThreadWorker();

	public:
		LPRCamera();
		~LPRCamera();
		Ref<Image> GetFrame();
		int GetFps();
		void Release();
		bool IsOpened();
		void SetDisableLPR(bool newValue);
		bool GetDisableLPR();
		void SetMaxFps(int newValue);
		int GetMaxFps();

		void SetCameraId(int newID);
		int GetCameraId();
		void SetStreamUrl(String newUrl);
		String GetStreamUrl();
		void SetDefaultTexture(Ref<Texture2D> newTexture);
		Ref<Texture2D> GetDefaultTexture();

		void SetModelPath(String newPath);
		String GetModelPath();
		void SetModelScoreThreshold(float newValue);
		float GetModelScoreThreshold();
		void SetModelNMSThreshold(float newValue);
		float GetModelNMSThreshold();
		void SetEnableCuda(bool newValue);
		bool GetEnableCuda();

		void SetTessData(String newPath);
		String GetTessData();

		void DetectNumberPlate(float timeout);
		void SetAutoDetect(bool newValue);
		bool GetAutoDetect();
	};
}

#endif // PR_CAMERA_H_
