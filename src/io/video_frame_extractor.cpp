/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "video_frame_extractor.hpp"
#include "core/include/core/logger.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <stb_image_write.h>

namespace lfs::io {

    class VideoFrameExtractor::Impl {
    public:
        bool extract(const Params& params, std::string& error) {
            AVFormatContext* fmt_ctx = nullptr;
            AVCodecContext* codec_ctx = nullptr;
            SwsContext* sws_ctx = nullptr;
            AVFrame* frame = nullptr;
            AVFrame* rgb_frame = nullptr;
            AVPacket* packet = nullptr;

            try {
                // Open input file
                if (avformat_open_input(&fmt_ctx, params.video_path.string().c_str(), nullptr, nullptr) < 0) {
                    error = "Failed to open video file";
                    return false;
                }

                if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
                    error = "Failed to find stream info";
                    avformat_close_input(&fmt_ctx);
                    return false;
                }

                // Find video stream
                int video_stream_idx = -1;
                for (unsigned int i = 0; i < fmt_ctx->nb_streams; i++) {
                    if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                        video_stream_idx = i;
                        break;
                    }
                }

                if (video_stream_idx == -1) {
                    error = "No video stream found";
                    avformat_close_input(&fmt_ctx);
                    return false;
                }

                AVStream* video_stream = fmt_ctx->streams[video_stream_idx];

                // Find decoder
                const AVCodec* codec = avcodec_find_decoder(video_stream->codecpar->codec_id);
                if (!codec) {
                    error = "Unsupported codec";
                    avformat_close_input(&fmt_ctx);
                    return false;
                }

                codec_ctx = avcodec_alloc_context3(codec);
                if (!codec_ctx) {
                    error = "Failed to allocate codec context";
                    avformat_close_input(&fmt_ctx);
                    return false;
                }

                if (avcodec_parameters_to_context(codec_ctx, video_stream->codecpar) < 0) {
                    error = "Failed to copy codec parameters";
                    avcodec_free_context(&codec_ctx);
                    avformat_close_input(&fmt_ctx);
                    return false;
                }

                if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
                    error = "Failed to open codec";
                    avcodec_free_context(&codec_ctx);
                    avformat_close_input(&fmt_ctx);
                    return false;
                }

                // Create output directory
                std::filesystem::create_directories(params.output_dir);

                // Calculate frame extraction parameters
                double video_fps = av_q2d(video_stream->r_frame_rate);
                int64_t total_frames = video_stream->nb_frames;
                if (total_frames == 0) {
                    total_frames = static_cast<int64_t>(fmt_ctx->duration * video_fps / AV_TIME_BASE);
                }

                int frame_step = 1;
                if (params.mode == ExtractionMode::FPS) {
                    frame_step = static_cast<int>(video_fps / params.fps);
                    if (frame_step < 1)
                        frame_step = 1;
                } else {
                    frame_step = params.frame_interval;
                }

                // Allocate frames
                frame = av_frame_alloc();
                rgb_frame = av_frame_alloc();
                packet = av_packet_alloc();

                if (!frame || !rgb_frame || !packet) {
                    error = "Failed to allocate frame/packet";
                    throw std::runtime_error(error);
                }

                // Setup RGB conversion
                rgb_frame->format = AV_PIX_FMT_RGB24;
                rgb_frame->width = codec_ctx->width;
                rgb_frame->height = codec_ctx->height;
                if (av_frame_get_buffer(rgb_frame, 0) < 0) {
                    error = "Failed to allocate RGB frame buffer";
                    throw std::runtime_error(error);
                }

                sws_ctx = sws_getContext(
                    codec_ctx->width, codec_ctx->height, codec_ctx->pix_fmt,
                    codec_ctx->width, codec_ctx->height, AV_PIX_FMT_RGB24,
                    SWS_BILINEAR, nullptr, nullptr, nullptr);

                if (!sws_ctx) {
                    error = "Failed to create scaling context";
                    throw std::runtime_error(error);
                }

                // Extract frames
                int frame_count = 0;
                int saved_count = 0;

                while (av_read_frame(fmt_ctx, packet) >= 0) {
                    if (packet->stream_index == video_stream_idx) {
                        if (avcodec_send_packet(codec_ctx, packet) == 0) {
                            while (avcodec_receive_frame(codec_ctx, frame) == 0) {
                                if (frame_count % frame_step == 0) {
                                    // Convert to RGB
                                    sws_scale(sws_ctx, frame->data, frame->linesize, 0, codec_ctx->height,
                                              rgb_frame->data, rgb_frame->linesize);

                                    // Save frame
                                    std::filesystem::path filename = params.output_dir /
                                                                     (std::string("frame_") + std::to_string(saved_count + 1) +
                                                                      (params.format == ImageFormat::PNG ? ".png" : ".jpg"));

                                    // Write image using stb_image_write
                                    bool write_success = false;
                                    if (params.format == ImageFormat::PNG) {
                                        write_success = stbi_write_png(filename.string().c_str(), rgb_frame->width, rgb_frame->height, 3,
                                                                       rgb_frame->data[0], rgb_frame->linesize[0]) != 0;
                                    } else {
                                        write_success = stbi_write_jpg(filename.string().c_str(), rgb_frame->width, rgb_frame->height, 3,
                                                                       rgb_frame->data[0], params.jpg_quality) != 0;
                                    }

                                    if (!write_success) {
                                        LOG_ERROR("Failed to write frame {}", saved_count + 1);
                                    }

                                    saved_count++;

                                    if (params.progress_callback) {
                                        int estimated_total = static_cast<int>(total_frames / frame_step);
                                        params.progress_callback(saved_count, estimated_total);
                                    }
                                }
                                frame_count++;
                            }
                        }
                    }
                    av_packet_unref(packet);
                }

                // Flush decoder
                avcodec_send_packet(codec_ctx, nullptr);
                while (avcodec_receive_frame(codec_ctx, frame) == 0) {
                    if (frame_count % frame_step == 0) {
                        sws_scale(sws_ctx, frame->data, frame->linesize, 0, codec_ctx->height,
                                  rgb_frame->data, rgb_frame->linesize);

                        std::filesystem::path filename = params.output_dir /
                                                         (std::string("frame_") + std::to_string(saved_count + 1) +
                                                          (params.format == ImageFormat::PNG ? ".png" : ".jpg"));

                        // Write image using stb_image_write
                        bool write_success = false;
                        if (params.format == ImageFormat::PNG) {
                            write_success = stbi_write_png(filename.string().c_str(), rgb_frame->width, rgb_frame->height, 3,
                                                           rgb_frame->data[0], rgb_frame->linesize[0]) != 0;
                        } else {
                            write_success = stbi_write_jpg(filename.string().c_str(), rgb_frame->width, rgb_frame->height, 3,
                                                           rgb_frame->data[0], params.jpg_quality) != 0;
                        }

                        if (!write_success) {
                            LOG_ERROR("Failed to write frame {}", saved_count + 1);
                        }

                        saved_count++;

                        if (params.progress_callback) {
                            int estimated_total = static_cast<int>(total_frames / frame_step);
                            params.progress_callback(saved_count, estimated_total);
                        }
                    }
                    frame_count++;
                }

                LOG_INFO("Extracted {} frames from video", saved_count);

                // Cleanup
                sws_freeContext(sws_ctx);
                av_frame_free(&frame);
                av_frame_free(&rgb_frame);
                av_packet_free(&packet);
                avcodec_free_context(&codec_ctx);
                avformat_close_input(&fmt_ctx);

                return true;

            } catch (const std::exception& e) {
                // Cleanup on error
                if (sws_ctx)
                    sws_freeContext(sws_ctx);
                if (frame)
                    av_frame_free(&frame);
                if (rgb_frame)
                    av_frame_free(&rgb_frame);
                if (packet)
                    av_packet_free(&packet);
                if (codec_ctx)
                    avcodec_free_context(&codec_ctx);
                if (fmt_ctx)
                    avformat_close_input(&fmt_ctx);

                error = e.what();
                return false;
            }
        }
    };

    VideoFrameExtractor::VideoFrameExtractor() : impl_(new Impl()) {}
    VideoFrameExtractor::~VideoFrameExtractor() { delete impl_; }

    bool VideoFrameExtractor::extract(const Params& params, std::string& error) {
        return impl_->extract(params, error);
    }

} // namespace lfs::io
