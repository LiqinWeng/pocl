/* OpenCL device using the spine-runtime (spert) tile scheduler (derived from
   the pthread device).

   Copyright (c) 2011-2021 PoCL developers
                 2026 SpacemiT

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <algorithm>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <vector>

#include <spert.hpp>

#include "common.h"
#include "common_driver.h"
#include "common_utils.h"
#include "pocl_builtin_kernels.h"
#include "pocl_cl.h"
#include "pocl_mem_management.h"
#include "pocl_runtime_config.h"
#include "pocl_util.h"
#include "spert_scheduler.h"
#include "utlist.h"

/* External functions declared in spert_scheduler.h */

void spert_stream_init(pocl_spert_scheduler_data *SchedData) {
  spert::Stream *Stream = new spert::Stream();
  if (!Stream->valid()) {
    POCL_MSG_ERR("spert: failed to create a Stream (no CC cores available?)\n");
    delete Stream;
    SchedData->spert_stream = nullptr;
    return;
  }
  SchedData->spert_stream = Stream;
  POCL_MSG_PRINT_INFO("spert: stream created with %u CC cores\n",
                      Stream->core_count());
}

void spert_stream_uninit(pocl_spert_scheduler_data *SchedData) {
  delete static_cast<spert::Stream *>(SchedData->spert_stream);
  SchedData->spert_stream = nullptr;
}

unsigned spert_stream_core_count(pocl_spert_scheduler_data *SchedData) {
  spert::Stream *Stream =
      static_cast<spert::Stream *>(SchedData->spert_stream);
  return Stream ? Stream->core_count() : 1;
}

/* Internal functions */

/* The tile function submitted to the spine-runtime. Each tile runs
 * WgsPerTile consecutive work-groups (in linear WG index order) to
 * completion; tiles never use cooperative scheduling primitives. */
static void runWGRangeOnTile(spert::Context *Ctx, kernel_run_command *RunCmd,
                             size_t LocalMemSize, size_t WgsPerTile,
                             size_t TotalWGs) {
  kernel_run_command *K = RunCmd;
  pocl_kernel_metadata_t *Meta = K->kernel->meta;
  const size_t TileIdx = Ctx->program_id(0);
  const size_t Begin = TileIdx * WgsPerTile;
  const size_t End = std::min(TotalWGs, Begin + WgsPerTile);

  if (Begin >= End)
    return;

  const size_t NumArgs = Meta->num_args + Meta->num_locals + 1;
  std::vector<void *> Arguments(NumArgs);
  std::vector<void *> Arguments2(NumArgs);

  /* Each tile executes its work-groups sequentially, so a single local
   * memory buffer per tile is sufficient. */
  char *LocalMem =
      (char *)pocl_aligned_malloc(MAX_EXTENDED_ALIGNMENT, LocalMemSize);
  struct pocl_context PC;

  pocl_setup_kernel_arg_array_with_locals(Arguments.data(), Arguments2.data(),
                                          K, LocalMem, LocalMemSize);
  memcpy(&PC, &K->pc, sizeof(struct pocl_context));

  /* LIMITATION: printf() from kernels is not supported by the spert driver
   * (MVP): no per-tile printf buffers are allocated. Kernels must not call
   * printf. */
  PC.printf_buffer = NULL;
  PC.printf_buffer_position = NULL;
  PC.printf_buffer_capacity = 0;

  pocl_cpu_setup_rm_and_ftz(K->device, K->kernel->program);

  const size_t RowSize = K->pc.num_groups[0];
  const size_t SliceSize = K->pc.num_groups[0] * K->pc.num_groups[1];
  uint32_t ExecutionFailed = 0;
  for (size_t I = Begin; I < End; ++I) {
    /* linear WG index -> 3D index (same mapping as the pthread driver) */
    const size_t GZ = I / SliceSize;
    const size_t GY = (I % SliceSize) / RowSize;
    const size_t GX = (I % SliceSize) % RowSize;
    K->workgroup((uint8_t *)Arguments.data(), (uint8_t *)&PC, GX, GY, GZ);
    ExecutionFailed |= PC.execution_failed;
  }
  POCL_ATOMIC_OR(RunCmd->execution_failed, ExecutionFailed);

  pocl_free_kernel_arg_array_with_locals(Arguments.data(), Arguments2.data(),
                                         K);
  pocl_aligned_free(LocalMem);
}

static void finalizeKernelCommand(kernel_run_command *RunCmd) {
  pocl_free_kernel_arg_array(RunCmd);

  pocl_release_dlhandle_cache(RunCmd->cmd->command.run.device_data);

  if (RunCmd->execution_failed)
    POCL_UPDATE_EVENT_FAILED_MSG(CL_FAILED, RunCmd->cmd->sync.event.event,
                                 "NDRange Kernel        ");
  else
    POCL_UPDATE_EVENT_COMPLETE_MSG(RunCmd->cmd->sync.event.event,
                                   "NDRange Kernel        ");
  free_kernel_run_command(RunCmd);
}

static kernel_run_command *
prepareKernelCommand(pocl_spert_scheduler_data *SchedData,
                     _cl_command_node *Cmd) {

  kernel_run_command *RunCmd;
  cl_kernel Kernel = Cmd->command.run.kernel;
  struct pocl_context *PC = &Cmd->command.run.pc;
  cl_program Program = Kernel->program;
  cl_uint DevI = Cmd->program_device_i;

  size_t NumGroups = PC->num_groups[0] * PC->num_groups[1] * PC->num_groups[2];

  // if WGs == 0, return early to avoid having to uninitialize anything
  if (NumGroups == 0) {
    pocl_update_event_running(Cmd->sync.event.event);

    POCL_UPDATE_EVENT_COMPLETE_MSG(Cmd->sync.event.event,
                                   "NDRange Kernel        ");

    return NULL;
  }

  /* initialize the program gvars if required */
  if (pocl_driver_build_gvar_init_kernel(Program, DevI, Cmd->device,
                                         pocl_cpu_gvar_init_callback) != 0) {
    pocl_update_event_running(Cmd->sync.event.event);
    POCL_UPDATE_EVENT_FAILED_MSG(CL_FAILED, Cmd->sync.event.event,
                                 "CPU: failed to compile GVar init kernel");
    return NULL;
  }

  char *SavedName = NULL;
  pocl_sanitize_builtin_kernel_name(Kernel, &SavedName);
  void *ci = pocl_check_kernel_dlhandle_cache(Cmd, CL_TRUE, CL_TRUE);
  Cmd->command.run.device_data = ci;
  pocl_restore_builtin_kernel_name(Kernel, SavedName);
  if (ci == NULL) {
    pocl_update_event_running(Cmd->sync.event.event);
    POCL_UPDATE_EVENT_FAILED_MSG(CL_FAILED, Cmd->sync.event.event,
                                 "CPU: failed to compile kernel");
    return NULL;
  }

  RunCmd = new_kernel_run_command();
  RunCmd->data = SchedData;
  RunCmd->kernel = Kernel;
  RunCmd->device = Cmd->device;
  RunCmd->pc = *PC;
  RunCmd->cmd = Cmd;
  RunCmd->pc.printf_buffer = NULL;
  RunCmd->pc.printf_buffer_capacity = 0;
  RunCmd->pc.printf_buffer_position = NULL;
  RunCmd->pc.global_var_buffer = (uchar *)Program->gvar_storage[DevI];
  RunCmd->workgroup =
      reinterpret_cast<pocl_workgroup_func>(Cmd->command.run.wg);
  RunCmd->kernel_args = Cmd->command.run.arguments;
  RunCmd->next = NULL;
  RunCmd->ref_count = 0;
  RunCmd->execution_failed = 0;

  pocl_setup_kernel_arg_array(RunCmd);

  pocl_update_event_running(Cmd->sync.event.event);

  return RunCmd;
}

static void execCommand(pocl_spert_scheduler_data *SchedData,
                        kernel_run_command *RunCmd) {
  const size_t TotalWGs = (size_t)RunCmd->pc.num_groups[0] *
                          RunCmd->pc.num_groups[1] * RunCmd->pc.num_groups[2];
  spert::Stream *Stream =
      static_cast<spert::Stream *>(SchedData->spert_stream);
  if (Stream == nullptr) {
    POCL_MSG_ERR("spert: no stream available, failing the kernel launch\n");
    POCL_ATOMIC_OR(RunCmd->execution_failed, 1);
    return;
  }

  /* A single spert launch is limited to max_tiles_per_launch tiles; split
   * the WG grid into chunks of WgsPerTile work-groups, each chunk executed
   * sequentially inside one tile. */
  const size_t MaxTiles = SchedData->max_tiles_per_launch;
  const size_t WgsPerTile = (TotalWGs + MaxTiles - 1) / MaxTiles;
  const size_t NumTiles = (TotalWGs + WgsPerTile - 1) / WgsPerTile;
  assert(NumTiles <= MaxTiles);

  spert::Future Fut = Stream->launch(spert::Grid((uint32_t)NumTiles),
                                     runWGRangeOnTile, RunCmd,
                                     SchedData->local_mem_size, WgsPerTile,
                                     TotalWGs);
  spert::Status St = Fut.sync();
  if (St != spert::Status::Ok) {
    POCL_MSG_ERR("spert: tile launch failed with status %s\n",
                 spert::to_string(St));
    POCL_ATOMIC_OR(RunCmd->execution_failed, 1);
  }
}

static int runSingleCommand(pocl_spert_scheduler_data *SchedData) {
  _cl_command_node *Cmd;
  kernel_run_command *RunCmd;

  POCL_LOCK(SchedData->wq_lock_fast);
  int DoExit = 0;

RETRY:
  DoExit = SchedData->meta_thread_shutdown_requested;

  /* execute a command if available */
  Cmd = SchedData->work_queue;
  if (Cmd) {
    DL_DELETE(SchedData->work_queue, Cmd);
    POCL_UNLOCK(SchedData->wq_lock_fast);

    assert(pocl_command_is_ready(Cmd->sync.event.event));

    if (Cmd->type == CL_COMMAND_NDRANGE_KERNEL) {
      assert((void *)SchedData == (void *)Cmd->device->data);
      RunCmd = prepareKernelCommand(SchedData, Cmd);
      if (RunCmd) {
        /* blocks until all tiles of the launch have finished */
        execCommand(SchedData, RunCmd);
        finalizeKernelCommand(RunCmd);
      }
    } else {
      pocl_exec_command(Cmd);
    }

    POCL_LOCK(SchedData->wq_lock_fast);
  }

  /* if no command was available, sleep */
  if ((Cmd == NULL) && (DoExit == 0)) {
    POCL_WAIT_COND(SchedData->wake_meta_thread, SchedData->wq_lock_fast);
    goto RETRY;
  }

  POCL_UNLOCK(SchedData->wq_lock_fast);

  return DoExit;
}

void *SpertDriverThread(void *Dev) {
  int DoExit = 0;
  cl_device_id Device = (cl_device_id)Dev;
  pocl_spert_scheduler_data *SchedData =
      (pocl_spert_scheduler_data *)Device->data;

  while (1) {
    DoExit = runSingleCommand(SchedData);
    if (DoExit) {
      return NULL;
    }
  }
}
