/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "video_frame_extractor.hpp"
#include "core/include/core/logger.hpp"
#include "nvcodec_image_loader.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <cuda_runtime.h>
#include <stb_image_write.h>

#include <fstream>

namespace lfs::io {

    namespace {
        constexpr int JPEG_BATCH_SIZE = 32;

        void write_jpeg_to_file(const std::filesystem::path& path,
                                const std::vector<uint8_t>& data) {
            std::ofstream file(path, std::ios::binary);
            if (file) {
                file.write(reinterpret_cast<const char*>(data.data()),
                           static_cast<std::streamsize>(data.size()));
            }
        }
    } // namespace

    class VideoFrameExtractor::Impl {
    public:
        bool extract(const Params& params, std::string& error) {
            AVFormatContext* fmt_ctx = nullptr;
            AVCodecContext* codec_ctx = nullptr;
            SwsContext* sws_ctx = nullptr;
            AVFrame* frame = nullptr;
            AVPacket* packet = nullptr;

            uint8_t* gpu_batch_buffer = nullptr;
            uint8_t* cpu_contiguous_buffer = nullptr;
            std::unique_ptr<NvCodecImageLoader> nvcodec;

            try {
                if (avformat_open_input(&fmt_ctx, params.video_path.string().c_str(),
                                        nullptr, nullptr) < 0) {
                    error = "Failed to open video file";
                    return false;
                }

                if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
                    error = "Failed to find stream info";
                    avformat_close_input(&fmt_ctx);
                    return false;
                }

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

                std::filesystem::create_directories(params.output_dir);

                const int width = codec_ctx->width;
                const int height = codec_ctx->height;
                const size_t frame_size = static_cast<size_t>(width) * height * 3;

                double video_fps = av_q2d(video_stream->r_frame_rate);
                int64_t total_frames = video_stream->nb_frames;
                if (total_frames == 0) {
                    total_frames =
                        static_cast<int64_t>(fmt_ctx->duration * video_fps / AV_TIME_BASE);
                }

                int frame_step = 1;
                if (params.mode == ExtractionMode::FPS) {
                    frame_step = static_cast<int>(video_fps / params.fps);
                    if (frame_step < 1)
                        frame_step = 1;
                } else {
                    frame_step = params.frame_interval;
                }

                frame = av_frame_alloc();
                packet = av_packet_alloc();
                if (!frame || !packet) {
                    error = "Failed to allocate frame/packet";
                    throw std::runtime_error(error);
                }

                cpu_contiguous_buffer = new uint8_t[frame_size];

                sws_ctx = sws_getContext(width, height, codec_ctx->pix_fmt, width, height,
                                         AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr, nullptr,
                                         nullptr);
                if (!sws_ctx) {
                    error = "Failed to create scaling context";
                    throw std::runtime_error(error);
                }

                const bool use_gpu_jpeg =
                    params.format == ImageFormat::JPG && NvCodecImageLoader::is_available();

                if (use_gpu_jpeg) {
                    NvCodecImageLoader::Options opts;
                    nvcodec = std::make_unique<NvCodecImageLoader>(opts);

                    cudaError_t cuda_err = cudaMalloc(&gpu_batch_buffer,
                                                      JPEG_BATCH_SIZE * frame_size);
                    if (cuda_err != cudaSuccess) {
                        LOG_WARN("Failed to allocate GPU batch buffer, falling back to CPU");
                    }
                }

                const bool gpu_encoding_enabled = use_gpu_jpeg && gpu_batch_buffer != nullptr;
                if (gpu_encoding_enabled) {
                    LOG_INFO("Using GPU batch JPEG encoding (batch size: {})", JPEG_BATCH_SIZE);
                } else if (params.format == ImageFormat::JPG) {
                    LOG_INFO("Using CPU JPEG encoding");
                } else {
                    LOG_INFO("Using CPU PNG encoding");
                }

                int frame_count = 0;
                int saved_count = 0;

                std::vector<void*> batch_gpu_ptrs;
                std::vector<std::filesystem::path> batch_filenames;
                int batch_idx = 0;

                auto flush_jpeg_batch = [&]() {
                    if (batch_gpu_ptrs.empty())
                        return;

                    auto encoded = nvcodec->encode_batch_rgb_to_jpeg(
                        batch_gpu_ptrs, width, height, params.jpg_quality);

                    for (size_t i = 0; i < encoded.size(); i++) {
                        if (!encoded[i].empty()) {
                            write_jpeg_to_file(batch_filenames[i], encoded[i]);
                        }
                    }

                    batch_gpu_ptrs.clear();
                    batch_filenames.clear();
                    batch_idx = 0;
                };

                auto process_frame = [&]() {
                    uint8_t* dst_data[1] = {cpu_contiguous_buffer};
                    int dst_linesize[1] = {width * 3};
                    sws_scale(sws_ctx, frame->data, frame->linesize, 0, height,
                              dst_data, dst_linesize);

                    std::filesystem::path filename =
                        params.output_dir /
                        (std::string("frame_") + std::to_string(saved_count + 1) +
                         (params.format == ImageFormat::PNG ? ".png" : ".jpg"));

                    if (gpu_encoding_enabled) {
                        void* dst_ptr = gpu_batch_buffer + batch_idx * frame_size;
                        cudaMemcpy(dst_ptr, cpu_contiguous_buffer, frame_size,
                                   cudaMemcpyHostToDevice);

                        batch_gpu_ptrs.push_back(dst_ptr);
                        batch_filenames.push_back(filename);
                        batch_idx++;

                        if (batch_idx >= JPEG_BATCH_SIZE) {
                            flush_jpeg_batch();
                        }
                    } else if (params.format == ImageFormat::JPG) {
                        stbi_write_jpg(filename.string().c_str(), width, height, 3,
                                       cpu_contiguous_buffer, params.jpg_quality);
                    } else {
                        stbi_write_png(filename.string().c_str(), width, height, 3,
                                       cpu_contiguous_buffer, width * 3);
                    }

                    saved_count++;

                    if (params.progress_callback) {
                        int estimated_total = static_cast<int>(total_frames / frame_step);
                        params.progress_callback(saved_count, estimated_total);
                    }
                };

                while (av_read_frame(fmt_ctx, packet) >= 0) {
                    if (packet->stream_index == video_stream_idx) {
                        if (avcodec_send_packet(codec_ctx, packet) == 0) {
                            while (avcodec_receive_frame(codec_ctx, frame) == 0) {
                                if (frame_count % frame_step == 0) {
                                    process_frame();
                                }
                                frame_count++;
                            }
                        }
                    }
                    av_packet_unref(packet);
                }

                avcodec_send_packet(codec_ctx, nullptr);
                while (avcodec_receive_frame(codec_ctx, frame) == 0) {
                    if (frame_count % frame_step == 0) {
                        process_frame();
                    }
                    frame_count++;
                }

                if (gpu_encoding_enabled) {
                    flush_jpeg_batch();
                }

                LOG_INFO("Extracted {} frames from video", saved_count);

                sws_freeContext(sws_ctx);
                av_frame_free(&frame);
                av_packet_free(&packet);
                avcodec_free_context(&codec_ctx);
                avformat_close_input(&fmt_ctx);
                delete[] cpu_contiguous_buffer;
                if (gpu_batch_buffer)
                    cudaFree(gpu_batch_buffer);

                return true;

            } catch (const std::exception& e) {
                if (sws_ctx)
                    sws_freeContext(sws_ctx);
                if (frame)
                    av_frame_free(&frame);
                if (packet)
                    av_packet_free(&packet);
                if (codec_ctx)
                    avcodec_free_context(&codec_ctx);
                if (fmt_ctx)
                    avformat_close_input(&fmt_ctx);
                delete[] cpu_contiguous_buffer;
                if (gpu_batch_buffer)
                    cudaFree(gpu_batch_buffer);

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
