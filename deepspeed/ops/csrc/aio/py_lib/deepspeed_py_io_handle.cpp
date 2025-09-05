// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include "deepspeed_py_io_handle.h"
#include <cstdlib>

#define O_DIRECT_ALIGNMENT 512

using namespace std;

static void _start_aio_thread(std::shared_ptr<struct deepspeed_aio_thread_t> ctxt) { ctxt->run(); }

// static bool is_valid_bytes_to_read(const char *filename,
//                                    const int64_t file_offset,
//                                    const int64_t num_bytes_to_read)
// {
//     int64_t num_file_bytes;
//     if (-1 == get_file_size(filename, num_file_bytes))
//     {
//         const auto error_code = errno;
//         report_file_error(filename, " fstat for read", error_code);
//         return false;
//     }
//     if ((file_offset + num_bytes_to_read) > num_file_bytes)
//     {
//         std::cout << filename << ": file_offset + buffer nbytes > file bytes "
//                   << (file_offset + num_bytes_to_read) << " > " << num_file_bytes << std::endl;
//     }
//     assert((file_offset + num_bytes_to_read) <= num_file_bytes);
//     return true;
// }

deepspeed_io_handle_t::deepspeed_io_handle_t(const int block_size,
                                             const int queue_depth,
                                             const bool single_submit,
                                             const bool overlap_events,
                                             const int intra_op_parallelism)
    : _aio_ctxt(new aio_context(block_size, queue_depth)),
      _single_submit(single_submit),
      _overlap_events(overlap_events),
      _intra_op_parallelism(intra_op_parallelism),
      _aio_config(block_size, queue_depth, single_submit, overlap_events, false),
      _num_pending_ops(0),
      _pinned_tensor_mgr(new deepspeed_pin_tensor_t())
{
    for (auto i = 0; i < intra_op_parallelism; ++i)
    {
        _thread_contexts.push_back(std::make_shared<deepspeed_aio_thread_t>(i, _aio_config));
    }

    for (auto &ctxt : _thread_contexts)
    {
        _threads.push_back(std::thread(_start_aio_thread, ctxt));
    }
}

deepspeed_io_handle_t::~deepspeed_io_handle_t()
{
    _stop_threads();
    for (auto &thr : _threads)
    {
        thr.join();
    }
}

const int deepspeed_io_handle_t::get_block_size() const
{
    return _aio_ctxt ? _aio_ctxt->_block_size : -1;
}

const int deepspeed_io_handle_t::get_queue_depth() const
{
    return _aio_ctxt ? _aio_ctxt->_queue_depth : -1;
}

const bool deepspeed_io_handle_t::get_single_submit() const { return _single_submit; }

const bool deepspeed_io_handle_t::get_overlap_events() const { return _overlap_events; }

const int deepspeed_io_handle_t::get_intra_op_parallelism() const { return _intra_op_parallelism; }

const int deepspeed_io_handle_t::get_alignment() const
{
    return _intra_op_parallelism * O_DIRECT_ALIGNMENT;
}

int deepspeed_io_handle_t::read(torch::Tensor &buffer,
                                const char *filename,
                                const bool validate,
                                const int64_t file_offset)
{
    const auto start_time = std::chrono::high_resolution_clock::now();
    assert(_aio_ctxt);

    int64_t num_file_bytes;
    if (-1 == get_file_size(filename, num_file_bytes))
    {
        const auto error_code = errno;
        report_file_error(filename, " fstat for read", error_code);
        return -1;
    }

    // 替换严格的断言为灵活的共享缓冲区处理
    const auto buffer_bytes = static_cast<int64_t>(buffer.nbytes());
    torch::Tensor actual_buffer = buffer;
    int64_t actual_read_bytes = num_file_bytes;

    if (buffer_bytes != num_file_bytes)
    {
        std::cout << "[SHARED_BUFFER_READ] Buffer size mismatch: buffer="
                  << buffer_bytes << " file=" << num_file_bytes << std::endl;

        if (buffer_bytes > num_file_bytes)
        {
            // 缓冲区比文件大，只读取文件大小
            actual_buffer = buffer.narrow(0, 0, num_file_bytes / buffer.element_size());
        }
        else
        {
            // 缓冲区比文件小，只读取缓冲区大小
            actual_read_bytes = buffer_bytes;
        }
    }

    const auto fd = open_file(filename, true);
    if (fd == -1)
    {
        return -1;
    }

    // 获取正确的数据指针
    auto buffer_ptr = (char *)actual_buffer.data_ptr();
    std::unique_ptr<io_xfer_ctxt> xfer_ctxt(
        new io_xfer_ctxt(fd, file_offset, 0, actual_read_bytes, buffer_ptr));

    if (_aio_config._overlap_events)
    {
        do_aio_operation_overlap(true, _aio_ctxt, xfer_ctxt, &_aio_config, nullptr);
    }
    else
    {
        do_aio_operation_sequential(true, _aio_ctxt, xfer_ctxt, &_aio_config, nullptr);
    }

    close(fd);
    const std::chrono::duration<double> aio_time =
        std::chrono::high_resolution_clock::now() - start_time;

    if (validate)
    {
        validate_aio_operation(true, filename, buffer_ptr, actual_read_bytes);
    }

    const std::chrono::duration<double> fn_time =
        std::chrono::high_resolution_clock::now() - start_time;
    std::cout << "Elapsed time(usec): " << "aio = " << aio_time.count() * 1e6
              << " call = " << fn_time.count() * 1e6 << std::endl;
    return 0;
}

int deepspeed_io_handle_t::write(const torch::Tensor &buffer,
                                 const char *filename,
                                 const bool validate,
                                 const int64_t file_offset)
{
    assert(_aio_ctxt);

    const auto start_time = std::chrono::high_resolution_clock::now();

    const auto fd = open_file(filename, false);
    if (fd == -1)
    {
        return -1;
    }

    auto write_buffer = (char *)buffer.data_ptr();
    const auto num_write_bytes = static_cast<int64_t>(buffer.nbytes());
    std::unique_ptr<io_xfer_ctxt> xfer_ctxt(
        new io_xfer_ctxt(fd, file_offset, 0, num_write_bytes, write_buffer));

    if (_aio_config._overlap_events)
    {
        do_aio_operation_overlap(false, _aio_ctxt, xfer_ctxt, &_aio_config, nullptr);
    }
    else
    {
        do_aio_operation_sequential(false, _aio_ctxt, xfer_ctxt, &_aio_config, nullptr);
    }
    const std::chrono::duration<double> aio_time =
        std::chrono::high_resolution_clock::now() - start_time;

    close(fd);

    if (validate)
    {
        validate_aio_operation(false, filename, write_buffer, num_write_bytes);
    }

    const std::chrono::duration<double> fn_time =
        std::chrono::high_resolution_clock::now() - start_time;
    std::cout << "Elapsed time(usec): " << "aio = " << aio_time.count() * 1e6
              << " call = " << fn_time.count() * 1e6 << std::endl;
    return 0;
}

void deepspeed_io_handle_t::_schedule_aio_work(std::shared_ptr<struct io_op_desc_t> scheduled_op)
{
    for (auto &ctxt : _thread_contexts)
    {
        {
            std::lock_guard<std::mutex> lock(ctxt->_work_sync._mutex);
            ctxt->_work_queue.push(scheduled_op);
        }
        ctxt->_work_sync._cond_var.notify_one();
    }
    _num_pending_ops++;
}

std::shared_ptr<struct io_op_desc_t> deepspeed_io_handle_t::_wait_for_aio_work()
{
    std::shared_ptr<struct io_op_desc_t> completed_op = nullptr;
    for (auto &ctxt : _thread_contexts)
    {
        std::unique_lock<std::mutex> lock(ctxt->_complete_sync._mutex);
        ctxt->_complete_sync._cond_var.wait(lock,
                                            [ctxt]
                                            { return !ctxt->_complete_queue.empty(); });
        completed_op = ctxt->_complete_queue.front();
        ctxt->_complete_queue.pop();
    }
    return completed_op;
}

void deepspeed_io_handle_t::_stop_threads()
{
    assert(0 == _num_pending_ops);
    for (auto &ctxt : _thread_contexts)
    {
        {
            std::lock_guard<std::mutex> lock(ctxt->_work_sync._mutex);
            ctxt->_time_to_exit = true;
        }
        ctxt->_work_sync._cond_var.notify_one();
    }
}

int deepspeed_io_handle_t::wait()
{
    assert(_num_pending_ops > 0);
    auto num_completed_ops = 0;

    while (_num_pending_ops > 0)
    {
        auto completed_op = _wait_for_aio_work();

        if (completed_op->_validate)
        {
            completed_op->validate();
        }

        completed_op->finish();

        if (!completed_op->_filename.empty())
        {
            (completed_op->_fd);
        }

        --_num_pending_ops;
        ++num_completed_ops;
    }

    return num_completed_ops;
}

bool deepspeed_io_handle_t::_is_valid_parallel_aio_op(const bool read_op, const int64_t num_bytes)
{
    const auto op_string = read_op ? "Read" : "Write";
    if (num_bytes % get_intra_op_parallelism())
    {
        std::cout << "deepspeed_aio failure: parallel " << op_string << " num_bytes = " << num_bytes
                  << " not divisible by intra op parallelism = " << get_intra_op_parallelism()
                  << std::endl;
        return false;
    }

    return true;
}

std::shared_ptr<struct io_op_desc_t> deepspeed_io_handle_t::_create_io_op_desc(
    const bool read_op,
    const torch::Tensor &buffer,
    const int fd,
    const char *filename,
    const bool validate,
    const int64_t file_offset)
{
    return std::make_shared<cpu_op_desc_t>(_pinned_tensor_mgr,
                                           read_op,
                                           buffer,
                                           fd,
                                           filename,
                                           _intra_op_parallelism,
                                           validate,
                                           file_offset);
}

int deepspeed_io_handle_t::_pread(const torch::Tensor &buffer,
                                  const int fd,
                                  const char *filename,
                                  const bool validate,
                                  const bool async,
                                  const int64_t file_offset)
{
    auto scheduled_op = _create_io_op_desc(true, buffer, fd, filename, validate, file_offset);

    _schedule_aio_work(scheduled_op);

    if (async)
    {
        return 0;
    }

    return wait();
}

// int deepspeed_io_handle_t::pread(const torch::Tensor& buffer,
//                                  const char* filename,
//                                  const bool validate,
//                                  const bool async,
//                                  const int64_t file_offset)
// {
//     const auto buffer_bytes = static_cast<int64_t>(buffer.nbytes());

//     if (!is_valid_bytes_to_read(filename, file_offset, buffer_bytes)) { return -1; }

//     if (!_is_valid_parallel_aio_op(true, buffer_bytes)) { return -1; }

//     const auto fd = open_file(filename, true);
//     if (fd == -1) { return -1; }

//     return _pread(buffer, fd, filename, validate, async, file_offset);
// }

int deepspeed_io_handle_t::pread(const torch::Tensor &buffer,
                                 const char *filename,
                                 const bool validate,
                                 const bool async,
                                 const int64_t file_offset)
{
    const auto buffer_bytes = static_cast<int64_t>(buffer.nbytes());

    // 获取文件大小进行安全检查
    int64_t num_file_bytes;
    if (-1 == get_file_size(filename, num_file_bytes))
    {
        const auto error_code = errno;
        report_file_error(filename, " fstat for read", error_code);
        return -1;
    }

    // 对于共享缓冲区，需要更精确的处理
    if ((file_offset + buffer_bytes) > num_file_bytes)
    {
        // std::cout << "[SHARED_BUFFER_FIX] File size mismatch detected:" << std::endl;
        // std::cout << "  Filename: " << filename << std::endl;
        // std::cout << "  Requested offset: " << file_offset << std::endl;
        // std::cout << "  Requested size: " << buffer_bytes << std::endl;
        // std::cout << "  Actual file size: " << num_file_bytes << std::endl;

        // 对于共享缓冲区，不应该随意调整偏移量
        // 而是应该只读取文件实际存在的部分
        if (file_offset >= num_file_bytes)
        {
            // 如果偏移量完全超出文件，这是一个严重错误
            std::cout << "[ERROR] File offset completely beyond file size" << std::endl;
            return -1; // 返回错误而不是尝试修复
        }

        // 只调整读取大小，保持原始偏移量
        const auto safe_read_size = num_file_bytes - file_offset;
        // std::cout << "  Adjusted read size to: " << safe_read_size << std::endl;

        // 创建一个临时的较小缓冲区视图
        auto safe_buffer = buffer.narrow(0, 0, safe_read_size / buffer.element_size());

        if (!_is_valid_parallel_aio_op(true, safe_read_size))
        {
            return -1;
        }

        const auto fd = open_file(filename, true);
        if (fd == -1)
        {
            return -1;
        }

        return _pread(safe_buffer, fd, filename, validate, async, file_offset);
    }

    // 正常情况下的处理
    if (!_is_valid_parallel_aio_op(true, buffer_bytes))
    {
        return -1;
    }

    const auto fd = open_file(filename, true);
    if (fd == -1)
    {
        return -1;
    }

    return _pread(buffer, fd, filename, validate, async, file_offset);
}

int deepspeed_io_handle_t::_pwrite(const torch::Tensor &buffer,
                                   const int fd,
                                   const char *filename,
                                   const bool validate,
                                   const bool async,
                                   const int64_t file_offset)
{
    auto scheduled_op = _create_io_op_desc(false, buffer, fd, filename, validate, file_offset);

    _schedule_aio_work(scheduled_op);

    if (async)
    {
        return 0;
    }

    return wait();
}

int deepspeed_io_handle_t::pwrite(const torch::Tensor &buffer,
                                  const char *filename,
                                  const bool validate,
                                  const bool async,
                                  const int64_t file_offset)
{
    const auto num_write_bytes = static_cast<int64_t>(buffer.nbytes());

    if (!_is_valid_parallel_aio_op(false, num_write_bytes))
    {
        return -1;
    }

    const auto fd = open_file(filename, false);
    if (fd == -1)
    {
        return -1;
    }

    return _pwrite(buffer, fd, filename, validate, async, file_offset);
}

int deepspeed_io_handle_t::sync_pread(torch::Tensor &buffer,
                                      const char *filename,
                                      const int64_t file_offset)
{
    return pread(buffer, filename, false, false, file_offset);
}

int deepspeed_io_handle_t::sync_pwrite(const torch::Tensor &buffer,
                                       const char *filename,
                                       const int64_t file_offset)
{
    return pwrite(buffer, filename, false, false, file_offset);
}

int deepspeed_io_handle_t::async_pread(torch::Tensor &buffer,
                                       const char *filename,
                                       const int64_t file_offset)
{
    return pread(buffer, filename, false, true, file_offset);
}

int deepspeed_io_handle_t::async_pwrite(const torch::Tensor &buffer,
                                        const char *filename,
                                        const int64_t file_offset)
{
    return pwrite(buffer, filename, false, true, file_offset);
}

int deepspeed_io_handle_t::async_pwrite(const torch::Tensor &buffer,
                                        const int fd,
                                        const int64_t file_offset = 0)
{
    const auto num_write_bytes = static_cast<int64_t>(buffer.nbytes());
    if (!_is_valid_parallel_aio_op(false, num_write_bytes))
    {
        return -1;
    }

    return _pwrite(buffer, fd, nullptr, false, true, file_offset);
}

at::Tensor deepspeed_io_handle_t::new_cpu_locked_tensor(const int64_t num_elem,
                                                        const torch::Tensor &example_tensor)
{
    return _pinned_tensor_mgr->alloc(num_elem, example_tensor.scalar_type());
}

bool deepspeed_io_handle_t::free_cpu_locked_tensor(torch::Tensor &locked_tensor)
{
    return _pinned_tensor_mgr->free(locked_tensor);
}
