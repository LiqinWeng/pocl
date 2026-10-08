/* OpenCL device using the spine-runtime (spert) tile scheduler (derived from
   the pthread device).

   Copyright (c) 2011-2012 Universidad Rey Juan Carlos
                 2011-2019 Pekka Jääskeläinen
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
#define __USE_GNU

#include <assert.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#ifndef _MSC_VER
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

#include "common.h"
#include "common_utils.h"
#include "config.h"
#include "devices.h"
#include "pocl_mem_management.h"
#include "pocl_util.h"
#include "spert.h"
#include "spert_scheduler.h"

/* Initializes scheduler. Must be called before any kernel enqueue */
static void spert_scheduler_init (cl_device_id device);
static void spert_scheduler_uninit (cl_device_id device);

void pocl_spert_init_device_ops(struct pocl_device_ops *ops) {
  pocl_pthread_init_device_ops(ops);

  ops->device_name = "cpu-spert-libspert";

  /* implementation that differs from pthread */
  ops->probe = pocl_spert_probe;
  ops->uninit = pocl_spert_uninit;
  ops->reinit = pocl_spert_reinit;
  ops->init = pocl_spert_init;

  ops->submit = pocl_spert_submit;
  ops->notify = pocl_spert_notify;
}

unsigned int pocl_spert_probe(struct pocl_device_ops *ops) {
  int env_count = pocl_device_get_env_count(ops->device_name);

  pocl_cpu_probe ();

  /* for backwards compatibility */
  if (env_count <= 0)
    env_count = pocl_device_get_env_count("spert");

  /* No env specified -> stay hidden; spert is opt-in (like basic, unlike
   * pthread which defaults to 1). */
  if (env_count < 0)
    env_count = 0;

  /* disallow more than one; it's possible, but makes no sense */
  if (env_count > 1) {
    POCL_MSG_WARN("Not using redundant spert devices\n");
    env_count = 1;
  }

  return env_count;
}

static cl_bool spert_available = CL_TRUE;
static cl_bool spert_unavailable = CL_FALSE;

cl_int pocl_spert_init(unsigned j, cl_device_id device, const char *parameters) {
  device->available = &spert_unavailable;
  cl_int err = pocl_cpu_init_common(device, j);
  if (err) {
    POCL_MSG_ERR("pocl_cpu_init_common failed\n");
    return CL_INVALID_DEVICE;
  }

  pocl_spert_scheduler_data *dd = calloc (1, sizeof (pocl_spert_scheduler_data));
  device->data = (void *)dd;

  /* subdevices not supported ATM */
  device->max_sub_devices = 0;
  device->num_partition_properties = 0;
  device->num_partition_types = 0;
  /* system mem as global memory */
  device->global_mem_id = 0;

  pocl_init_dlhandle_cache();
  pocl_init_kernel_run_command_manager();
  spert_scheduler_init(device);

  device->max_compute_units = spert_stream_core_count (dd);

  POCL_MSG_PRINT_INFO ("spert device %u initialized\n", j);
  device->available = &spert_available;
  return err;
}

cl_int
pocl_spert_uninit (unsigned J, cl_device_id Device)
{
  spert_scheduler_uninit (Device);
  return CL_SUCCESS;
}

cl_int
pocl_spert_reinit (unsigned J, cl_device_id Device, const char *Parameters)
{
  spert_scheduler_init (Device);
  return CL_SUCCESS;
}

static void
spert_scheduler_init (cl_device_id device)
{
  pocl_spert_scheduler_data *dd = (pocl_spert_scheduler_data *)device->data;
  POCL_INIT_LOCK (dd->wq_lock_fast);
  dd->work_queue = NULL;

  POCL_INIT_COND (dd->wake_meta_thread);

  /* safety margin - aligning pointers later (in kernel arg setup)
   * may require more local memory than actual local mem size.
   * TODO fix this */
  dd->local_mem_size = device->local_mem_size
                       + device->max_parameter_size * MAX_EXTENDED_ALIGNMENT;
  dd->local_mem_size = (1 + (dd->local_mem_size - 1) / MAX_EXTENDED_ALIGNMENT)
                       * MAX_EXTENDED_ALIGNMENT;

  /* Hardware constraint of the spine-runtime: a single launch may contain
   * at most 512 tiles. */
  dd->max_tiles_per_launch = 512;

  /* check out the CC cores (creates the spert::Stream) */
  dd->spert_stream = NULL;
  spert_stream_init (dd);

  dd->meta_thread_shutdown_requested = 0;
  /* create one meta thread per device to serve as an async interface thread. */
  POCL_CREATE_THREAD (dd->meta_thread, SpertDriverThread, device);
}

static void
spert_scheduler_uninit (cl_device_id device)
{
  pocl_spert_scheduler_data *dd = (pocl_spert_scheduler_data *)device->data;

  POCL_LOCK (dd->wq_lock_fast);
  dd->meta_thread_shutdown_requested = 1;
  POCL_BROADCAST_COND (dd->wake_meta_thread);
  POCL_UNLOCK (dd->wq_lock_fast);

  POCL_JOIN_THREAD (dd->meta_thread);

  POCL_DESTROY_LOCK (dd->wq_lock_fast);
  POCL_DESTROY_COND (dd->wake_meta_thread);

  dd->meta_thread_shutdown_requested = 0;
  dd->work_queue = NULL;

  spert_stream_uninit (dd);
}

/* The spert device doesn't support subdevices, so push_command can use
 * cond_signal */
static void
spert_scheduler_push_command (_cl_command_node *cmd)
{
  cl_device_id device = cmd->device;
  pocl_spert_scheduler_data *dd = (pocl_spert_scheduler_data *)device->data;
  POCL_LOCK (dd->wq_lock_fast);
  DL_APPEND (dd->work_queue, cmd);
  POCL_SIGNAL_COND (dd->wake_meta_thread);
  POCL_UNLOCK (dd->wq_lock_fast);
}

void pocl_spert_submit(_cl_command_node *node, cl_command_queue cq) {
  node->state = POCL_COMMAND_READY;
  /* If a wait-list dependency has failed, this command must not run: fail it via
   * the error cascade. There is no point waiting for it to become otherwise
   * ready first. pocl_update_event_failed takes the cq + event locks itself, so
   * unlock the event we hold and return without unlocking again. */
  if (node->sync.event.event->failed_dependency) {
    POCL_UNLOCK_OBJ(node->sync.event.event);
    pocl_update_event_failed(CL_FAILED, NULL, 0, node->sync.event.event, NULL);
    return;
  }
  if (pocl_command_is_ready(node->sync.event.event)) {
    pocl_update_event_submitted(node->sync.event.event);
    spert_scheduler_push_command(node);
  }
  POCL_UNLOCK_OBJ(node->sync.event.event);
  return;
}

void pocl_spert_notify(cl_device_id device, cl_event event, cl_event finished) {
  cl_bool wake_thread = CL_FALSE;
  _cl_command_node *node = event->command;

  /* Fail the command if the notifier finished failed, or if any wait-list
   * dependency has already failed (recorded on the event). In either case the
   * command MUST NOT execute, so there is no point checking readiness first. */
  if (finished->status < CL_COMPLETE || event->failed_dependency)
  {
    /* Unlock the finished event in order to prevent a lock order violation
     * with the command queue that will be locked during
     * pocl_update_event_failed.
     */
    pocl_unlock_events_inorder (event, finished);
    pocl_update_event_failed (CL_FAILED, NULL, 0, event, NULL);
    /* Lock events in this order to avoid a lock order violation between
     * the finished/notifier and event/wait events.
     */
    pocl_lock_events_inorder (finished, event);
    return;
  }

  if (node->state != POCL_COMMAND_READY)
    {
      POCL_MSG_PRINT_EVENTS (
        "spert: command related to the notified event %lu not ready\n",
        event->id);
      return;
    }

  if (pocl_command_is_ready(node->sync.event.event)) {
    if (event->status == CL_QUEUED) {
      pocl_update_event_submitted(event);
      wake_thread = CL_TRUE;
    }
  }
  if (wake_thread != CL_FALSE)
    {
      spert_scheduler_push_command (node);
    }
  return;
}
