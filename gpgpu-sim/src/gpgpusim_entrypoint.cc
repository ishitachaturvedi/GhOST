// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ivan Sham,
// Andrew Turner, Ali Bakhoda, The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "gpgpusim_entrypoint.h"
#include <stdio.h>

#include "../libcuda/gpgpu_context.h"
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_ir.h"
#include "cuda-sim/ptx_parser.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu-sim/icnt_wrapper.h"
#include "option_parser.h"
#include "stream_manager.h"
#include <iostream>

using namespace std;

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

static int sg_argc = 3;
static const char *sg_argv[] = {"", "-config", "gpgpusim.config"};

#include <vector>
#include <fstream>
using std::vector;

// long total_times_in_cycle = 0;
// vector<vector<vector<int>>>stallData;
// vector<vector<int>>act_warp;
// vector<vector<int>>issued_warp;
// vector<vector<vector<int>>> str_status; 
// vector<int>warp_issue;
// vector<int>icnt_pressure;
// vector<int>warps_cannot_be_issued;
// vector<unsigned>warp_inst_num;
// vector<int> indep_pc_num_push_all_stalling_inst;

// vector<int> stall_consolidated;

// long long inst_counter = 0;
// long long max_active;
// long long actw;
// long long max_warps_act;
// long long cycles_passed = 0;
// long long max_sid;
// long long num_of_schedulers;
// long long numstall = 12;
// long long print_on = 0;
// long long going_from_shader_to_mem = 0;
// long long present_ongoing_cycle = 0;
// long long stall_cycles = 0;
// long long stall_cycles_1_sched = 0;
// long long stall_cycles_2_sched = 0;
// long long stall_cycles_3_sched = 0;
// long long stall_cycles_4_sched = 0;
// int stall_happened = 0;
// int warps_found = 0;
// long long tot_icnt_buffer = 0;
// long long tot_inst_exec = 0;
// long long tot_cycles_exec_all_SM = 0;
// long long tot_inst_ret = 0;
// int total_warps = 0;

// int tot_mem_dep_checks = 0;
// int tot_mem_dep_true = 0;
// int tot_mem_dep_false = 0;
// int time_bw_inst = 0; 
// int reached_barrier = 0;

// int tot_issues_ILP = 0;
// int tot_issues_OOO = 0;

// // Stats collection
// long long mem_data_stall = 0;
// long long comp_data_stall = 0;
// long long ibuffer_stall = 0;
// long long comp_str_stall = 0;
// long long mem_str_stall = 0;
// long long mem_data_stall_kernel = 0;
// long long tot_inst_OOO_because_dep = 0;
// long long ibuffer_stall_kernel = 0;
// long long comp_str_stall_kernel = 0;
// long long mem_str_stall_kernel = 0;
// long long waiting_warp = 0;
// long long SM_num = 0;
// long long idle = 0;
// long long ooo_opp_kernel = 0;
// long WAR_and_WAW_stalls = 0;
// bool WAR_or_WAW_found = 0;
// long long NO_INST_ISSUE = 0;
// long long opp_for_ooo = 0;
// long long opp_for_mem = 0;

// bool stalled_on_DEB_dependence = 0;
// long long cannot_issue_warp_DEB_dep = 0;
// long long cannot_issue_warp_DEB_dep_kernel = 0;

// long long tot_cycles_wb = 0;
// long long tot_cycles_wb_done = 0;
// long long inst_issued_sid_0 = 0;

// int inst_counter_warp = 0;
// int inst_counter_warp_dec = 0;

// long long sync_inst_collision = 0;
// long long sync_inst_ibuffer_coll = 0;
// long long mem_inst_collision = 0;
// long long mem_inst_ibuffer_coll = 0;

// long long tot_in_order_stall = 0;
// long long memory_inst_stuck = 0;
// long inst_exec_total = 0;

// long gpu_sim_insn_test = 0;
// long gpu_sim_insn_test_run = 0;

// bool replay_stall = 0;

// long total_mem_inst = 0;
// long total_dep_found = 0;
// long total_inst_going_for_renaming = 0;
// long total_inst_renamed = 0;
// long total_backend_stall_cycles = 0;
// long mem_inst_in_flight = 0;
// long tot_mem_inst_in_flight = 0;
// long independent_instruction_count = 0;
// int whole_sched_stall = 0;

// writing the warp issued order to file
// read data from warp file to execute sched order
vector<int> warp_sched_order;
long long warp_issued_counter = 0;
long long DEB_BUFFER_SIZE = 1;

void *gpgpu_sim_thread_sequential(void *ctx_ptr) {
  gpgpu_context *ctx = (gpgpu_context *)ctx_ptr;
  // at most one kernel running at a time
  bool done;
  do {
    sem_wait(&(ctx->the_gpgpusim->g_sim_signal_start));
    done = true;
    if (ctx->the_gpgpusim->g_the_gpu->get_more_cta_left()) {
      done = false;
      ctx->the_gpgpusim->g_the_gpu->init();
      while (ctx->the_gpgpusim->g_the_gpu->active()) {
        ctx->the_gpgpusim->g_the_gpu->cycle();
        ctx->the_gpgpusim->g_the_gpu->deadlock_check();
      }
      ctx->the_gpgpusim->g_the_gpu->print_stats();
      ctx->the_gpgpusim->g_the_gpu->update_stats();
      ctx->print_simulation_time();
    }
    sem_post(&(ctx->the_gpgpusim->g_sim_signal_finish));
  } while (!done);
  sem_post(&(ctx->the_gpgpusim->g_sim_signal_exit));
  return NULL;
}

static void termination_callback() {

  // std::cout <<"TOTAL CYCLES TAKEN "<<cycles_passed<<"\n";
  // cout <<"gpu_sim_insn_test "<<gpu_sim_insn_test<<"\n";
  // cout <<"total_mem_inst "<<total_mem_inst<<"\n";
  // cout <<"total_dep_found "<<total_dep_found<<"\n";
  // cout <<"total_inst_going_for_renaming "<<total_inst_going_for_renaming<<"\n";
  // cout <<"total_inst_renamed "<<total_inst_renamed<<"\n";
  // cout <<"tot_mem_dep_checks "<<tot_mem_dep_checks<<"\n";
  // cout <<"tot_mem_dep_true "<<tot_mem_dep_true<<"\n";
  // cout <<"tot_mem_dep_false "<<tot_mem_dep_false<<"\n";
  // cout <<"tot_issues_ILP "<<tot_issues_ILP<<"\n";
  // cout <<"stall_cycles "<<stall_cycles<<"\n";
  // cout <<"stall_cycles_1_sched "<<stall_cycles_1_sched<<"\n";
  // cout <<"stall_cycles_2_sched "<<stall_cycles_2_sched<<"\n";
  // cout <<"stall_cycles_3_sched "<<stall_cycles_3_sched<<"\n";
  // cout <<"stall_cycles_4_sched "<<stall_cycles_4_sched<<"\n";
  // cout <<"total_times_in_cycle "<<total_times_in_cycle<<"\n";
  // cout <<"tot_issues_OOO "<<tot_issues_OOO<<"\n";
  // cout <<"tot_in_order_stall "<<tot_in_order_stall<<"\n";
  // cout <<"tot_inst_OOO_because_dep "<<tot_inst_OOO_because_dep<<"\n";
  // cout <<"WAR_and_WAW_stalls "<<WAR_and_WAW_stalls<<"\n";
  // cout <<"tot_inst_exec "<<tot_inst_exec<<"\n";

  // cout <<"STALL_STATS_PER_SCHED\n";

  // cout <<"SM sched mem_data comp_data ibuffer comp_str mem_str waiting_warp_stall idle_stall total_cycles total_stalls\n";
  // for(int i = 0; i<= SM_num; i++)
  // {
  //   for(int j=0; j<4; j++)
  //   {
  //     cout <<i<<" "<<j<<" ";
  //     for(int k = 0;k<numstall; k++)
  //     {
  //       cout <<stallData[i][j][k]<<" ";
  //     }
  //     cout <<"\n";
  //   }
  // }

  printf("GPGPU-Sim: *** exit detected ***\n");
  fflush(stdout);
}

void *gpgpu_sim_thread_concurrent(void *ctx_ptr) {
  gpgpu_context *ctx = (gpgpu_context *)ctx_ptr;
  atexit(termination_callback);

  // // Per Shader
  // stallData.resize(500,
  //   // Per scheduler
  //   vector<vector<int>>(4,
  //     // Per Stall
  //     vector<int>(numstall,0)));

  //   // Per Shader
  // str_status.resize(500,
  //   // Per Sched
  //   vector<vector<int>>(4,
  //     // Per str
  //     vector<int>(8,0)));

  // warp_inst_num.resize(500,0);

  // act_warp.resize(500, vector<int>(300,0));
  // issued_warp.resize(500, vector<int>(4,0));
  // warp_issue.resize(64,0);
  // icnt_pressure.resize(500,0);
  // warps_cannot_be_issued.resize(100,0);
  // indep_pc_num_push_all_stalling_inst.resize(100,0);
  // stall_consolidated.resize(20,0);
  
  // concurrent kernel execution simulation thread
  do {
    if (g_debug_execution >= 3) {
      printf(
          "GPGPU-Sim: *** simulation thread starting and spinning waiting for "
          "work ***\n");
      fflush(stdout);
    }
    while (ctx->the_gpgpusim->g_stream_manager->empty_protected() &&
           !ctx->the_gpgpusim->g_sim_done)
      ;
    if (g_debug_execution >= 3) {
      printf("GPGPU-Sim: ** START simulation thread (detected work) **\n");
      ctx->the_gpgpusim->g_stream_manager->print(stdout);
      fflush(stdout);
    }
    pthread_mutex_lock(&(ctx->the_gpgpusim->g_sim_lock));
    ctx->the_gpgpusim->g_sim_active = true;
    pthread_mutex_unlock(&(ctx->the_gpgpusim->g_sim_lock));
    bool active = false;
    bool sim_cycles = false;
    ctx->the_gpgpusim->g_the_gpu->init();
    do {
      // check if a kernel has completed
      // launch operation on device if one is pending and can be run

      // Need to break this loop when a kernel completes. This was a
      // source of non-deterministic behaviour in GPGPU-Sim (bug 147).
      // If another stream operation is available, g_the_gpu remains active,
      // causing this loop to not break. If the next operation happens to be
      // another kernel, the gpu is not re-initialized and the inter-kernel
      // behaviour may be incorrect. Check that a kernel has finished and
      // no other kernel is currently running.
      if (ctx->the_gpgpusim->g_stream_manager->operation(&sim_cycles) &&
          !ctx->the_gpgpusim->g_the_gpu->active())
        break;

      // functional simulation
      if (ctx->the_gpgpusim->g_the_gpu->is_functional_sim()) {
        kernel_info_t *kernel =
            ctx->the_gpgpusim->g_the_gpu->get_functional_kernel();
        assert(kernel);
        ctx->the_gpgpusim->gpgpu_ctx->func_sim->gpgpu_cuda_ptx_sim_main_func(
            *kernel);
        ctx->the_gpgpusim->g_the_gpu->finish_functional_sim(kernel);
      }

      // performance simulation
      if (ctx->the_gpgpusim->g_the_gpu->active()) {
        ctx->the_gpgpusim->g_the_gpu->cycle();
        sim_cycles = true;
        ctx->the_gpgpusim->g_the_gpu->deadlock_check();
      } else {
        if (ctx->the_gpgpusim->g_the_gpu->cycle_insn_cta_max_hit()) {
          ctx->the_gpgpusim->g_stream_manager->stop_all_running_kernels();
          ctx->the_gpgpusim->g_sim_done = true;
          ctx->the_gpgpusim->break_limit = true;
        }
      }

      active = ctx->the_gpgpusim->g_the_gpu->active() ||
               !(ctx->the_gpgpusim->g_stream_manager->empty_protected());

    } while (active && !ctx->the_gpgpusim->g_sim_done);
    if (g_debug_execution >= 3) {
      printf("GPGPU-Sim: ** STOP simulation thread (no work) **\n");
      fflush(stdout);
    }
    if (sim_cycles) {
      ctx->the_gpgpusim->g_the_gpu->print_stats();
      ctx->the_gpgpusim->g_the_gpu->update_stats();
      ctx->print_simulation_time();
    }
    pthread_mutex_lock(&(ctx->the_gpgpusim->g_sim_lock));
    ctx->the_gpgpusim->g_sim_active = false;
    pthread_mutex_unlock(&(ctx->the_gpgpusim->g_sim_lock));
  } while (!ctx->the_gpgpusim->g_sim_done);

  printf("GPGPU-Sim: *** simulation thread exiting ***\n");
  fflush(stdout);

  if (ctx->the_gpgpusim->break_limit) {
    printf(
        "GPGPU-Sim: ** break due to reaching the maximum cycles (or "
        "instructions) **\n");
    exit(1);
  }

  sem_post(&(ctx->the_gpgpusim->g_sim_signal_exit));
  return NULL;
}

void gpgpu_context::synchronize() {
  printf("GPGPU-Sim: synchronize waiting for inactive GPU simulation\n");
  the_gpgpusim->g_stream_manager->print(stdout);
  fflush(stdout);
  //    sem_wait(&g_sim_signal_finish);
  bool done = false;
  do {
    pthread_mutex_lock(&(the_gpgpusim->g_sim_lock));
    done = (the_gpgpusim->g_stream_manager->empty() &&
            !the_gpgpusim->g_sim_active) ||
           the_gpgpusim->g_sim_done;
    pthread_mutex_unlock(&(the_gpgpusim->g_sim_lock));
  } while (!done);
  printf("GPGPU-Sim: detected inactive GPU simulation thread\n");
  fflush(stdout);
  //    sem_post(&g_sim_signal_start);
}

void gpgpu_context::exit_simulation() {
  the_gpgpusim->g_sim_done = true;
  printf("GPGPU-Sim: exit_simulation called\n");
  fflush(stdout);
  sem_wait(&(the_gpgpusim->g_sim_signal_exit));
  printf("GPGPU-Sim: simulation thread signaled exit\n");
  fflush(stdout);
}

gpgpu_sim *gpgpu_context::gpgpu_ptx_sim_init_perf() {
  srand(1);
  print_splash();
  func_sim->read_sim_environment_variables();
  ptx_parser->read_parser_environment_variables();
  option_parser_t opp = option_parser_create();

  ptx_reg_options(opp);
  func_sim->ptx_opcocde_latency_options(opp);

  icnt_reg_options(opp);
  the_gpgpusim->g_the_gpu_config = new gpgpu_sim_config(this);
  the_gpgpusim->g_the_gpu_config->reg_options(
      opp);  // register GPU microrachitecture options

  option_parser_cmdline(opp, sg_argc, sg_argv);  // parse configuration options
  fprintf(stdout, "GPGPU-Sim: Configuration options:\n\n");
  option_parser_print(opp, stdout);
  // Set the Numeric locale to a standard locale where a decimal point is a
  // "dot" not a "comma" so it does the parsing correctly independent of the
  // system environment variables
  assert(setlocale(LC_NUMERIC, "C"));
  the_gpgpusim->g_the_gpu_config->init();

  the_gpgpusim->g_the_gpu =
      new exec_gpgpu_sim(*(the_gpgpusim->g_the_gpu_config), this);
  the_gpgpusim->g_stream_manager = new stream_manager(
      (the_gpgpusim->g_the_gpu), func_sim->g_cuda_launch_blocking);

  the_gpgpusim->g_simulation_starttime = time((time_t *)NULL);

  sem_init(&(the_gpgpusim->g_sim_signal_start), 0, 0);
  sem_init(&(the_gpgpusim->g_sim_signal_finish), 0, 0);
  sem_init(&(the_gpgpusim->g_sim_signal_exit), 0, 0);

  return the_gpgpusim->g_the_gpu;
}

void gpgpu_context::start_sim_thread(int api) {
  if (the_gpgpusim->g_sim_done) {
    the_gpgpusim->g_sim_done = false;
    if (api == 1) {
      pthread_create(&(the_gpgpusim->g_simulation_thread), NULL,
                     gpgpu_sim_thread_concurrent, (void *)this);
    } else {
      pthread_create(&(the_gpgpusim->g_simulation_thread), NULL,
                     gpgpu_sim_thread_sequential, (void *)this);
    }
  }
}

void gpgpu_context::print_simulation_time() {
  time_t current_time, difference, d, h, m, s;
  current_time = time((time_t *)NULL);
  difference = MAX(current_time - the_gpgpusim->g_simulation_starttime, 1);

  d = difference / (3600 * 24);
  h = difference / 3600 - 24 * d;
  m = difference / 60 - 60 * (h + 24 * d);
  s = difference - 60 * (m + 60 * (h + 24 * d));

  fflush(stderr);
  printf(
      "\n\ngpgpu_simulation_time = %u days, %u hrs, %u min, %u sec (%u sec)\n",
      (unsigned)d, (unsigned)h, (unsigned)m, (unsigned)s, (unsigned)difference);
  printf("gpgpu_simulation_rate = %u (inst/sec)\n",
         (unsigned)(the_gpgpusim->g_the_gpu->gpu_tot_sim_insn / difference));
  const unsigned cycles_per_sec =
      (unsigned)(the_gpgpusim->g_the_gpu->gpu_tot_sim_cycle / difference);
  printf("gpgpu_simulation_rate = %u (cycle/sec)\n", cycles_per_sec);
  printf("gpgpu_silicon_slowdown = %ux\n",
         the_gpgpusim->g_the_gpu->shader_clock() * 1000 / cycles_per_sec);
  fflush(stdout);
}

int gpgpu_context::gpgpu_opencl_ptx_sim_main_perf(kernel_info_t *grid) {
  the_gpgpusim->g_the_gpu->launch(grid);
  sem_post(&(the_gpgpusim->g_sim_signal_start));
  sem_wait(&(the_gpgpusim->g_sim_signal_finish));
  return 0;
}

//! Functional simulation of OpenCL
/*!
 * This function call the CUDA PTX functional simulator
 */
int cuda_sim::gpgpu_opencl_ptx_sim_main_func(kernel_info_t *grid) {
  // calling the CUDA PTX simulator, sending the kernel by reference and a flag
  // set to true, the flag used by the function to distinguish OpenCL calls from
  // the CUDA simulation calls which it is needed by the called function to not
  // register the exit the exit of OpenCL kernel as it doesn't register entering
  // in the first place as the CUDA kernels does
  gpgpu_cuda_ptx_sim_main_func(*grid, true);
  return 0;
}
