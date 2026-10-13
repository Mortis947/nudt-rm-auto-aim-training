// first-homework/Mortis/src/main.cpp
// 绿球识别与运动追踪（Week1 作业）
//
// 目标：从视频中检测绿色小球，用圆形框标出位置，用箭头表示运动方向，
//       箭头长度表示速度大小、颜色随速度变化，并标注球的大小。
//
// 处理流水线（每帧执行）：
//   1. 绿色分割：BGR -> HSV，阈值提取绿色区域，开+闭运算去噪
//   2. 候选筛选：轮廓 -> 面积/圆度/长宽比过滤，区分绿球与绿色圆柱
//   3. 跨帧匹配：形状打分 + 上一帧速度预测位置，抑制目标跳变
//   4. 遮挡外推：短时丢失用速度外推位置，避免贴边时中断
//   5. 可视化：检测圆 + 速度箭头(随速度变色) + 球大小/速度文本 + 运动轨迹

#include <opencv2/opencv.hpp>
#include <opencv2/freetype.hpp>

#include <algorithm>
#include <cmath>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// 可调参数（换视频时主要改这里）
// ---------------------------------------------------------------------------

// 绿色 HSV 范围（8 位 HSV 图像中，OpenCV 的 H 通常为 [0,179]；范围需按实际视频调试）
const cv::Scalar kLowerGreen(35, 45, 40);
const cv::Scalar kUpperGreen(95, 255, 255);

// 轮廓面积范围：绿球在这个视频里比较大，按像素面积卡一个合理区间
constexpr double kMinArea = 7000.0;
constexpr double kMaxArea = 32000.0;

// 形状打分参数：与上面的筛选阈值分开管理，便于调参
constexpr double kScoreCircularityStart = 0.68;
constexpr double kScoreCircularityRange = 0.12;
constexpr double kTargetArea = 12000.0;
constexpr double kAreaScoreTolerance = 10000.0;
constexpr double kDistancePenalty = 0.01;
constexpr double kRadiusPenalty = 0.02;
constexpr double kMaxRadiusChangeRatio = 0.45;

// 正常帧的形状阈值：圆度（=4πA/P²，理想圆为 1）和长宽比（w/h）
constexpr double kMinCircularity = 0.68;
constexpr double kMinAspect = 0.85;
constexpr double kMaxAspect = 1.20;

// 贴边帧的放宽阈值：球被画面边缘裁切后形状会变形，放宽避免漏检
constexpr double kBorderMinCircularity = 0.45;
constexpr double kBorderMinAspect = 0.55;
constexpr double kBorderMaxAspect = 1.45;
constexpr int kBorderMargin = 8;

// 短时遮挡：连续丢失帧数不超过该值时用速度外推，否则重置
constexpr int kMaxLostFrames = 8;

// 速度箭头：长度 = clamp(速度 × kArrowScale, kArrowMinLen, kArrowMaxLen)
constexpr double kArrowScale = 4.5;
constexpr double kArrowMinLen = 25.0;
constexpr double kArrowMaxLen = 140.0;

// 运动轨迹：保留最近 N 帧位置绘制"彗尾"
constexpr int kMaxTrail = 30;

// 速度变色参考值（像素/帧）：速度达到该值时箭头显示为红色
constexpr double kFastSpeedRef = 30.0;

// 速度低于此阈值时不画箭头，避免静止目标仍显示一个虚假的方向
constexpr double kMinArrowSpeed = 1.0;

// 小于该速度变化量时，状态文本显示 stable，以减少轻微抖动造成的闪烁
constexpr double kStableSpeedDeltaPps = 2.0;

// 元数据无法提供有效 FPS 时采用的显示/输出假设值；此时速度只是近似值
constexpr double kDefaultFps = 30.0;

// 速度平滑系数（EMA）：越小越平滑，用于抑制箭头方向/长度的帧间抖动
constexpr double kSmoothAlpha = 0.35;

// 黑体（SimHei）候选路径：优先 Windows 字体（WSL 可访问），失败回退到内置字体
const char * const kFontPaths[] = {
  "/mnt/c/Windows/Fonts/simhei.ttf",
  "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
};

// ---------------------------------------------------------------------------
// 数据结构
// ---------------------------------------------------------------------------

// 单个绿色轮廓的几何信息
struct BallCandidate {
  cv::Point2f center;
  float radius = 0.0F;
  double area = 0.0;
  double circularity = 0.0;
  double aspect_ratio = 0.0;
  cv::Rect bounding_box;
};

// 当前帧的跟踪状态
struct BallTrack {
  cv::Point2f center;
  cv::Point2f velocity;         // 相邻帧真实位移（像素/帧），用于位置预测
  cv::Point2f smooth_velocity;  // EMA 平滑后的速度（像素/帧），用于箭头显示
  float radius = 0.0F;
  double speed = 0.0;           // 平滑速度大小（像素/帧）
  double confidence = 0.0;      // 检测置信度（0~1），来自形状打分
  cv::Rect bounding_box;
  int lost_frames = 0;
  bool initialized = false;
};

// ---------------------------------------------------------------------------
// 工具函数
// ---------------------------------------------------------------------------

// 判断轮廓是否贴近画面边缘（贴边时球会被裁切，需要放宽形状约束）
bool isNearFrameBorder(const cv::Rect & box, const cv::Size & frame_size, int margin = kBorderMargin)
{
  return box.x <= margin || box.y <= margin ||
         box.x + box.width >= frame_size.width - margin ||
         box.y + box.height >= frame_size.height - margin;
}

// 把点限制在画面内（外推时避免越界）
cv::Point2f clampPointToFrame(const cv::Point2f & p, const cv::Size & frame_size)
{
  const float x = std::clamp(p.x, 0.0F, static_cast<float>(frame_size.width - 1));
  const float y = std::clamp(p.y, 0.0F, static_cast<float>(frame_size.height - 1));
  return cv::Point2f(x, y);
}

// 检测绿球候选：HSV 分割 + 形态学 + 轮廓几何筛选
std::vector<BallCandidate> detectBallCandidates(const cv::Mat & frame)
{
  cv::Mat hsv;
  cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

  cv::Mat mask;
  cv::inRange(hsv, kLowerGreen, kUpperGreen, mask);

  // 开运算去噪点，闭运算填空洞。结构元素固定，因此只创建一次并复用。
  static const cv::Mat kernel =
    cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  std::vector<BallCandidate> candidates;
  candidates.reserve(contours.size());

  for (const auto & contour : contours) {
    const double area = cv::contourArea(contour);
    if (area < kMinArea || area > kMaxArea) {
      continue;
    }

    const double perimeter = cv::arcLength(contour, true);
    if (perimeter <= 0.0) {
      continue;
    }

    const cv::Rect box = cv::boundingRect(contour);
    if (box.height <= 0) {
      continue;
    }

    const double aspect_ratio = static_cast<double>(box.width) / box.height;
    const double circularity = 4.0 * CV_PI * area / (perimeter * perimeter);
    const bool near_border = isNearFrameBorder(box, frame.size());

    // 圆度 + 长宽比区分绿球（圆）和绿色圆柱（细长）
    if (near_border) {
      if (circularity < kBorderMinCircularity || aspect_ratio < kBorderMinAspect ||
          aspect_ratio > kBorderMaxAspect) {
        continue;
      }
    } else {
      if (circularity < kMinCircularity || aspect_ratio < kMinAspect ||
          aspect_ratio > kMaxAspect) {
        continue;
      }
    }

    cv::Point2f center;
    float radius = 0.0F;
    cv::minEnclosingCircle(contour, center, radius);

    candidates.push_back(BallCandidate{center, radius, area, circularity, aspect_ratio, box});
  }

  return candidates;
}

// 形状打分：分数越高越像"球"
double computeShapeScore(const BallCandidate & c)
{
  // 分数归一化到 [0, 1]；这些参数集中定义，避免和筛选阈值分散在不同位置。
  const double circularity_score = std::clamp(
    (c.circularity - kScoreCircularityStart) / kScoreCircularityRange, 0.0, 1.0);
  const double aspect_score = std::clamp(
    1.0 - std::abs(c.aspect_ratio - 1.0) / 0.20, 0.0, 1.0);
  const double area_score = std::clamp(
    1.0 - std::abs(c.area - kTargetArea) / kAreaScoreTolerance, 0.0, 1.0);
  return 0.45 * circularity_score + 0.35 * aspect_score + 0.20 * area_score;
}

// 从候选中选出最可能的目标：形状分 + 跨帧运动连续性
std::optional<BallCandidate> chooseBestCandidate(
  const std::vector<BallCandidate> & candidates, const BallTrack & previous,
  double * confidence = nullptr)
{
  if (candidates.empty()) {
    return std::nullopt;
  }

  double best_score = -std::numeric_limits<double>::infinity();
  const BallCandidate * best = nullptr;

  for (const auto & c : candidates) {
    double score = computeShapeScore(c);

    if (previous.initialized) {
      // 用上一帧速度预测当前位置
      const cv::Point2f predicted = previous.center + previous.velocity;
      const double distance = cv::norm(c.center - predicted);
      const double max_distance = std::max(previous.radius * 1.8, previous.speed * 3.5 + 35.0);

      // 距离太远 => 跳到了别的绿色物体，拒绝
      if (distance > max_distance) {
        continue;
      }
      score -= distance * kDistancePenalty;

      // 半径突变过大 => 误匹配，拒绝
      const double radius_gap = std::abs(c.radius - previous.radius);
      if (radius_gap > previous.radius * kMaxRadiusChangeRatio) {
        continue;
      }
      score -= radius_gap * kRadiusPenalty;
    }

    if (score > best_score) {
      best_score = score;
      best = &c;
    }
  }

  if (best == nullptr) {
    return std::nullopt;
  }
  if (confidence != nullptr) {
    *confidence = computeShapeScore(*best);
  }
  return *best;
}

// 生成速度状态文本（速度 + 加减速趋势）
std::string buildStatusText(double speed_pps, double delta_pps)
{
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(1) << "speed: " << speed_pps << " px/s";
  if (std::abs(delta_pps) < kStableSpeedDeltaPps) {
    ss << "  stable";
  } else if (delta_pps > 0.0) {
    ss << "  accelerating";
  } else {
    ss << "  decelerating";
  }
  return ss.str();
}

// 速度 → 箭头颜色：绿(慢) → 黄(中) → 红(快)
cv::Scalar speedToColor(double speed)
{
  const double t = std::clamp(speed / kFastSpeedRef, 0.0, 1.0);
  const int red = static_cast<int>(255.0 * std::min(1.0, 2.0 * t));
  const int green = static_cast<int>(255.0 * std::min(1.0, 2.0 * (1.0 - t)));
  return cv::Scalar(0, green, red);  // BGR
}

// 惰性加载黑体（SimHei）；成功返回字体对象，失败返回空指针（回退到内置字体）
cv::Ptr<cv::freetype::FreeType2> getFont()
{
  static const cv::Ptr<cv::freetype::FreeType2> font = []() {
    auto f = cv::freetype::createFreeType2();
    for (const char * path : kFontPaths) {
      std::error_code ec;
      if (std::filesystem::exists(path, ec) && !ec) {
        f->loadFontData(path, 0);
        return f;
      }
    }
    return cv::Ptr<cv::freetype::FreeType2>();
  }();
  return font;
}

// 统一文字绘制：优先黑体，失败回退 OpenCV 内置字体（org 均为文本左上角）
void drawText(cv::Mat & frame, const std::string & text, cv::Point org,
  int font_height, cv::Scalar color, int thickness)
{
  const auto font = getFont();
  if (font) {
    font->putText(frame, text, org, font_height, color, thickness, cv::LINE_AA, false);
  } else {
    const double scale = static_cast<double>(font_height) / 28.0;
    cv::putText(frame, text, org + cv::Point(0, font_height), cv::FONT_HERSHEY_DUPLEX,
      scale, color, thickness, cv::LINE_AA);
  }
}

// 绘制 YOLO 风格标签：目标框上方背景色块 + 「类名 置信度」白字
void drawYoloLabel(cv::Mat & frame, const std::string & label, const cv::Rect & box,
  cv::Scalar bg_color, int font_height)
{
  const auto font = getFont();
  const int thickness = -1;  // 实心填充

  cv::Size text_size;
  if (font) {
    int baseline = 0;
    text_size = font->getTextSize(label, font_height, thickness, &baseline);
  } else {
    int baseline = 0;
    text_size = cv::getTextSize(label, cv::FONT_HERSHEY_DUPLEX,
      static_cast<double>(font_height) / 28.0, 2, &baseline);
  }

  const int pad_x = 8;
  const int pad_y = 4;
  cv::Point tl(box.x, box.y - text_size.height - pad_y * 2);
  if (tl.y < 0) {
    tl.y = box.y + 2;  // 贴顶边时放到框内，避免越界
  }
  cv::rectangle(frame, cv::Rect(tl.x, tl.y, text_size.width + pad_x * 2,
    text_size.height + pad_y * 2), bg_color, cv::FILLED);

  if (font) {
    font->putText(frame, label, cv::Point(tl.x + pad_x, tl.y + pad_y), font_height,
      cv::Scalar(255, 255, 255), thickness, cv::LINE_AA, false);
  } else {
    cv::putText(frame, label, cv::Point(tl.x + pad_x, tl.y + pad_y + text_size.height),
      cv::FONT_HERSHEY_DUPLEX, static_cast<double>(font_height) / 28.0,
      cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
  }
}

// 绘制运动轨迹（彗尾）：旧点偏绿、新点偏红
void drawTrail(cv::Mat & frame, const std::deque<cv::Point2f> & trail)
{
  if (trail.size() < 2) {
    return;
  }
  for (size_t i = 1; i < trail.size(); ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(trail.size() - 1);
    const cv::Scalar color(0, static_cast<int>(255.0 * (1.0 - t)), static_cast<int>(255.0 * t));
    cv::line(frame, trail[i - 1], trail[i], color, 2, cv::LINE_AA);
  }
}

// 绘制检测圆、速度箭头和状态文本
void drawTrackOverlay(cv::Mat & frame, const BallTrack & current, const BallTrack & previous,
  double fps, int frame_index, bool predicted)
{
  cv::circle(frame, current.center, static_cast<int>(std::round(current.radius)),
    cv::Scalar(0, 255, 255), 2);
  cv::circle(frame, current.center, 4, cv::Scalar(0, 0, 255), -1);

  const double speed_pps = current.speed * fps;
  const double prev_speed_pps = previous.speed * fps;
  const double delta_pps = speed_pps - prev_speed_pps;

  // 只有速度和方向都可信时才画箭头；静止时不伪造一个向右的方向。
  const cv::Point2f arrow = current.smooth_velocity;
  const double arrow_norm = cv::norm(arrow);
  if (current.speed >= kMinArrowSpeed && arrow_norm > 1e-6) {
    const double len = std::clamp(current.speed * kArrowScale, kArrowMinLen, kArrowMaxLen);
    const cv::Point2f dir = arrow * static_cast<float>(len / arrow_norm);
    const cv::Point2f end = current.center + dir;
    cv::arrowedLine(frame, current.center, end, speedToColor(current.speed),
      3, cv::LINE_AA, 0, 0.28);
  }

  // YOLO 风格标签：类名 + 置信度，贴在目标框上方（外推时用灰色背景区分）。
  {
    std::ostringstream label_ss;
    label_ss << "green ball " << std::fixed << std::setprecision(2) << current.confidence;
    const cv::Scalar label_bg = predicted ? cv::Scalar(110, 110, 110) : cv::Scalar(0, 180, 0);
    drawYoloLabel(frame, label_ss.str(), current.bounding_box, label_bg, 24);
  }

  // 其余文字固定绘制在左上角，避免遮挡运动中的小球。
  constexpr int kTextLeft = 20;
  constexpr int kTextLine = 40;
  int text_y = 32;

  drawText(frame, "frame: " + std::to_string(frame_index),
    cv::Point(kTextLeft, text_y), 22, cv::Scalar(255, 255, 255), -1);
  text_y += kTextLine;

  // 外推期间的速度只是沿用上次观测值；重获目标的第一帧也不计算跨越丢失区间的瞬时速度。
  std::string status_text;
  if (predicted) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1)
       << "speed: ~" << speed_pps << " px/s  predicted";
    status_text = ss.str();
  } else if (previous.lost_frames > 0) {
    status_text = "speed: recalibrating";
  } else {
    status_text = buildStatusText(speed_pps, delta_pps);
  }
  drawText(frame, status_text, cv::Point(kTextLeft, text_y), 24,
    cv::Scalar(255, 255, 255), -1);
  text_y += kTextLine;

  // 球的大小标注（图像中的像素半径，不是现实世界长度）
  const std::string size_text =
    "r: " + std::to_string(static_cast<int>(std::round(current.radius))) + " px";
  drawText(frame, size_text, cv::Point(kTextLeft, text_y), 24,
    cv::Scalar(255, 255, 0), -1);
}

}  // namespace

int main(int argc, char ** argv)
{
  const std::filesystem::path default_input = "first-homework/first-homework.mp4";
  const std::filesystem::path default_output =
    "first-homework/Mortis/first-homework-annotated.mp4";

  const std::filesystem::path input_path = argc > 1 ? argv[1] : default_input;
  const std::filesystem::path output_path = argc > 2 ? argv[2] : default_output;

  // 防止输入、输出指向同一路径，避免打开写入器时覆盖输入视频。
  try {
    const auto absolute_input = std::filesystem::absolute(input_path).lexically_normal();
    const auto absolute_output = std::filesystem::absolute(output_path).lexically_normal();
    if (absolute_input == absolute_output) {
      std::cerr << "输入视频和输出视频不能是同一个路径。" << std::endl;
      return 1;
    }
  } catch (const std::filesystem::filesystem_error & e) {
    std::cerr << "检查文件路径失败: " << e.what() << std::endl;
    return 1;
  }

  cv::VideoCapture capture(input_path.string());
  if (!capture.isOpened()) {
    std::cerr << "无法打开输入视频: " << input_path << std::endl;
    return 1;
  }

  const double reported_fps = capture.get(cv::CAP_PROP_FPS);
  const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
  const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));

  if (width <= 0 || height <= 0) {
    std::cerr << "输入视频的宽度或高度无效: " << width << "x" << height << std::endl;
    return 1;
  }

  const bool fps_is_valid = std::isfinite(reported_fps) && reported_fps > 0.0;
  const double fps = fps_is_valid ? reported_fps : kDefaultFps;
  if (!fps_is_valid) {
    std::cerr << "警告：无法从视频读取有效 FPS，将按 " << kDefaultFps
              << " FPS 处理；显示的 px/s 速度可能不准确。" << std::endl;
  }

  // 仅当路径包含目录时创建父目录；输出为当前目录下的文件名时不传空路径。
  const std::filesystem::path output_parent = output_path.parent_path();
  if (!output_parent.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(output_parent, ec);
    if (ec) {
      std::cerr << "无法创建输出目录: " << output_parent
                << "，原因: " << ec.message() << std::endl;
      return 1;
    }
  }

  cv::VideoWriter writer(output_path.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
    fps, cv::Size(width, height));
  if (!writer.isOpened()) {
    std::cerr << "无法创建输出视频: " << output_path
              << "。请检查路径、写入权限以及当前 OpenCV 的视频编码器支持。" << std::endl;
    return 1;
  }

  BallTrack previous;
  std::deque<cv::Point2f> trail;
  int frame_index = 0;

  for (cv::Mat frame; capture.read(frame); ++frame_index) {
    if (frame.empty()) {
      std::cerr << "警告：第 " << frame_index << " 帧为空，停止处理。" << std::endl;
      break;
    }

    const auto candidates = detectBallCandidates(frame);
    double confidence = 0.0;
    const auto selected = chooseBestCandidate(candidates, previous, &confidence);

    if (selected.has_value()) {
      BallTrack current;
      current.center = selected->center;
      current.radius = selected->radius;
      current.bounding_box = selected->bounding_box;
      current.confidence = confidence;
      current.initialized = true;
      current.lost_frames = 0;

      // 若上一帧是外推位置，不能把“重获目标时的位置差”当成普通相邻帧位移。
      // 重获的第一帧先重置速度估计，下一帧再恢复正常计算，避免出现虚假的速度尖峰。
      if (previous.initialized && previous.lost_frames == 0) {
        const cv::Point2f raw_velocity = current.center - previous.center;
        current.velocity = raw_velocity;
        current.smooth_velocity =
          static_cast<float>(kSmoothAlpha) * raw_velocity +
          static_cast<float>(1.0 - kSmoothAlpha) * previous.smooth_velocity;
      } else {
        current.velocity = cv::Point2f(0.0F, 0.0F);
        current.smooth_velocity = cv::Point2f(0.0F, 0.0F);
      }
      current.speed = cv::norm(current.smooth_velocity);

      trail.push_back(current.center);
      if (trail.size() > kMaxTrail) {
        trail.pop_front();
      }
      drawTrail(frame, trail);
      drawTrackOverlay(frame, current, previous, fps, frame_index, false);
      previous = current;
    } else if (previous.initialized && previous.lost_frames < kMaxLostFrames) {
      // 短时丢失：按上一帧速度外推。注意这只是预测，不是实际检测结果。
      BallTrack predicted = previous;
      predicted.center = clampPointToFrame(previous.center + previous.velocity, frame.size());
      predicted.bounding_box.x = static_cast<int>(std::round(predicted.center.x - predicted.radius));
      predicted.bounding_box.y = static_cast<int>(std::round(predicted.center.y - predicted.radius));
      predicted.bounding_box.width = static_cast<int>(std::round(predicted.radius * 2.0F));
      predicted.bounding_box.height = static_cast<int>(std::round(predicted.radius * 2.0F));
      predicted.lost_frames = previous.lost_frames + 1;

      trail.push_back(predicted.center);
      if (trail.size() > kMaxTrail) {
        trail.pop_front();
      }
      drawTrail(frame, trail);
      drawTrackOverlay(frame, predicted, previous, fps, frame_index, true);
      previous = predicted;
    } else {
      trail.clear();
      drawText(frame, "green ball lost", cv::Point(20, 36), 28, cv::Scalar(0, 0, 255), -1);
      previous = BallTrack{};
    }

    writer.write(frame);
  }

  if (frame_index == 0) {
    std::cerr << "输入视频没有可读取的帧，未生成有效输出。" << std::endl;
    return 1;
  }

  std::cout << "已写入标注视频: " << output_path
            << "（处理帧数: " << frame_index << ")" << std::endl;
  return 0;
}
