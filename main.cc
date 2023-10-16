// developed by Mahmoud Khairy, Purdue Univ
// abdallm@purdue.edu

#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <time.h>
#include <vector>

#include "gpgpu_context.h"
#include "abstract_hardware_model.h"
#include "cuda-sim/cuda-sim.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu-sim/icnt_wrapper.h"
#include "gpgpusim_entrypoint.h"
#include "option_parser.h"
#include "../ISA_Def/trace_opcode.h"
#include "trace_driven.h"
#include "../trace-parser/trace_parser.h"
#include "accelsim_version.h"

using namespace std;

#include <vector>
#include <fstream>
using std::vector;

long total_times_in_cycle = 0;
vector<vector<vector<int>>>stallData;
vector<vector<int>>act_warp;
vector<vector<int>>issued_warp;
vector<vector<vector<int>>> str_status; 
vector<int>warp_issue;
vector<int>icnt_pressure;
vector<int>warps_cannot_be_issued;
vector<unsigned>warp_inst_num;
vector<int> indep_pc_num_push_all_stalling_inst;

vector<int> stall_consolidated;

long long inst_counter = 0;
long long max_active;
long long actw;
long long max_warps_act;
long long cycles_passed = 0;
long long max_sid;
long long num_of_schedulers;
long long numstall = 13;
long long print_on = 0;
long long going_from_shader_to_mem = 0;
long long present_ongoing_cycle = 0;
long long stall_cycles = 0;
long long stall_cycles_1_sched = 0;
long long stall_cycles_2_sched = 0;
long long stall_cycles_3_sched = 0;
long long stall_cycles_4_sched = 0;
int stall_happened = 0;
int whole_sched_stall = 0;
int warps_found = 0;
long long tot_icnt_buffer = 0;
long long tot_inst_exec = 0;
long long tot_cycles_exec_all_SM = 0;
long long tot_inst_ret = 0;
int total_warps = 0;

int tot_mem_dep_checks = 0;
int tot_mem_dep_true = 0;
int tot_mem_dep_false = 0;
int time_bw_inst = 0; 
int reached_barrier = 0;

int tot_issues_ILP = 0;
int tot_issues_OOO = 0;

long tot_mem_inst_in_flight = 0;
long mem_inst_in_flight = 0;

// Stats collection
long long mem_data_stall = 0;
long long comp_data_stall = 0;
long long ibuffer_stall = 0;
long long comp_str_stall = 0;
long long mem_str_stall = 0;
long long mem_data_stall_kernel = 0;
long long tot_inst_OOO_because_dep = 0;
long long ibuffer_stall_kernel = 0;
long long comp_str_stall_kernel = 0;
long long mem_str_stall_kernel = 0;
long long waiting_warp = 0;
long long SM_num = 0;
long long idle = 0;
long long ooo_opp_kernel = 0;
long WAR_and_WAW_stalls = 0;
bool WAR_or_WAW_found = 0;
long long NO_INST_ISSUE = 0;
long long opp_for_ooo = 0;
long long opp_for_mem = 0;

bool stalled_on_DEB_dependence = 0;
long long cannot_issue_warp_DEB_dep = 0;
long long cannot_issue_warp_DEB_dep_kernel = 0;

long long tot_cycles_wb = 0;
long long tot_cycles_wb_done = 0;
long long inst_issued_sid_0 = 0;

int inst_counter_warp = 0;
int inst_counter_warp_dec = 0;
int branch_insts = 0;

long long sync_inst_collision = 0;
long long sync_inst_ibuffer_coll = 0;
long long mem_inst_collision = 0;
long long mem_inst_ibuffer_coll = 0;

long long tot_in_order_stall = 0;
long long memory_inst_stuck = 0;
long inst_exec_total = 0;

long gpu_sim_insn_test = 0;
long gpu_sim_insn_test_run = 0;

bool replay_stall = 0;

long total_mem_inst = 0;
long total_dep_found = 0;
long total_inst_going_for_renaming = 0;
long total_inst_renamed = 0;

// writing the warp issued order to file
// read data from warp file to execute sched order
vector<int> warp_sched_order;
long long warp_issued_counter = 0;
long long DEB_BUFFER_SIZE = 1;
long independent_instruction_count = 0;
long independent_instruction_stuck_count = 0;
std::vector<int>indep_inst_found_loc;
std::vector<int>indep_instruction_streams_found_loc;
long stalled_inst_tot = 0;
long total_backend_stall_cycles_and_front_end = 0;
long total_backend_stall_cycles_not_front_end = 0;
long none_stalls = 0;
long total_backend_not_stall_cycles_but_front_end = 0;

/* TO DO:
 * NOTE: the current version of trace-driven is functionally working fine,
 * but we still need to improve traces compression and simulation speed.
 * This includes:
 *
 * 1- Prefetch concurrent thread that prefetches traces from disk (to not be
 * limited by disk speed)
 *
 * 2- traces compression format a) cfg format and remove
 * thread/block Id from the head and b) using zlib library to save in binary format
 *
 * 3- Efficient memory improvement (save string not objects - parse only 10 in
 * the buffer)
 *
 * 4- Seeking capability - thread scheduler (save tb index and warp
 * index info in the traces header)
 *
 * 5- Get rid off traces intermediate files -
 * change the tracer
 */

gpgpu_sim *gpgpu_trace_sim_init_perf_model(int argc, const char *argv[],
                                           gpgpu_context *m_gpgpu_context,
                                           class trace_config *m_config);

trace_kernel_info_t *create_kernel_info( kernel_trace_t* kernel_trace_info,
		                      gpgpu_context *m_gpgpu_context, class trace_config *config,
							  trace_parser *parser);


int main(int argc, const char **argv) {
  std::cout << "Accel-Sim [build " << g_accelsim_version << "]";
  gpgpu_context *m_gpgpu_context = new gpgpu_context();
  trace_config tconfig;

  gpgpu_sim *m_gpgpu_sim =
      gpgpu_trace_sim_init_perf_model(argc, argv, m_gpgpu_context, &tconfig);
  m_gpgpu_sim->init();

  trace_parser tracer(tconfig.get_traces_filename());

  tconfig.parse_config();

  // Per Shader
  stallData.resize(500,
    // Per scheduler
    vector<vector<int>>(4,
      // Per Stall
      vector<int>(numstall,0)));

    // Per Shader
  str_status.resize(500,
    // Per Sched
    vector<vector<int>>(4,
      // Per str
      vector<int>(8,0)));

  warp_inst_num.resize(500,0);

  act_warp.resize(500, vector<int>(300,0));
  issued_warp.resize(500, vector<int>(4,0));
  warp_issue.resize(64,0);
  icnt_pressure.resize(500,0);
  warps_cannot_be_issued.resize(100,0);
  indep_pc_num_push_all_stalling_inst.resize(100,0);
  stall_consolidated.resize(20,0);
  indep_inst_found_loc.resize(32,0);
  indep_instruction_streams_found_loc.resize(32,0);

  // for each kernel
  // load file
  // parse and create kernel info
  // launch
  // while loop till the end of the end kernel execution
  // prints stats
  bool concurrent_kernel_sm =  m_gpgpu_sim->getShaderCoreConfig()->gpgpu_concurrent_kernel_sm;
  unsigned window_size = concurrent_kernel_sm ? m_gpgpu_sim->get_config().get_max_concurrent_kernel() : 1;
  assert(window_size > 0);
  std::vector<trace_command> commandlist = tracer.parse_commandlist_file();
  std::vector<unsigned long> busy_streams;
  std::vector<trace_kernel_info_t*> kernels_info;
  kernels_info.reserve(window_size);

  unsigned i = 0;
  while (i < commandlist.size() || !kernels_info.empty()) {
    trace_kernel_info_t *kernel_info = NULL;
    if (commandlist[i].m_type == command_type::cpu_gpu_mem_copy) {
      size_t addre, Bcount;
      tracer.parse_memcpy_info(commandlist[i].command_string, addre, Bcount);
      std::cout << "launching memcpy command : " << commandlist[i].command_string << std::endl;
      m_gpgpu_sim->perf_memcpy_to_gpu(addre, Bcount);
      i++;
      continue;
    } else if (commandlist[i].m_type == command_type::kernel_launch) {
      // Read trace header info for window_size number of kernels
      while (kernels_info.size() < window_size && i < commandlist.size()) {
        kernel_trace_t* kernel_trace_info = tracer.parse_kernel_info(commandlist[i].command_string);
        kernel_info = create_kernel_info(kernel_trace_info, m_gpgpu_context, &tconfig, &tracer);
        kernels_info.push_back(kernel_info);
        std::cout << "Header info loaded for kernel command : " << commandlist[i].command_string << std::endl;
        i++;
      }
      
      // Launch all kernels within window that are on a stream that isn't already running
      for (auto k : kernels_info) {
        bool stream_busy = false;
        for (auto s: busy_streams) {
          if (s == k->get_cuda_stream_id())
            stream_busy = true;
        }
        if (!stream_busy && m_gpgpu_sim->can_start_kernel() && !k->was_launched()) {
          std::cout << "launching kernel name: " << k->get_name() << " uid: " << k->get_uid() << std::endl;
          m_gpgpu_sim->launch(k);
          k->set_launched();
          busy_streams.push_back(k->get_cuda_stream_id());
        }
      }
    }
    else if (kernels_info.empty())
    	assert(0 && "Undefined Command");

    bool active = false;
    bool sim_cycles = false;
    unsigned finished_kernel_uid = 0;

    do {
      if (!m_gpgpu_sim->active())
        break;

      // performance simulation
      if (m_gpgpu_sim->active()) {
        m_gpgpu_sim->cycle();
        sim_cycles = true;
        m_gpgpu_sim->deadlock_check();
      } else {
        if (m_gpgpu_sim->cycle_insn_cta_max_hit()) {
          m_gpgpu_context->the_gpgpusim->g_stream_manager
              ->stop_all_running_kernels();
          break;
        }
      }

      active = m_gpgpu_sim->active();
      finished_kernel_uid = m_gpgpu_sim->finished_kernel();
    } while (active && !finished_kernel_uid);

    // cleanup finished kernel
    if (finished_kernel_uid) {
      trace_kernel_info_t* k = NULL;
      for (unsigned j = 0; j < kernels_info.size(); j++) {
        k = kernels_info.at(j);
        if (k->get_uid() == finished_kernel_uid) {
          for (int l = 0; l < busy_streams.size(); l++) {
            if (busy_streams.at(l) == k->get_cuda_stream_id()) {
              busy_streams.erase(busy_streams.begin()+l);
              break;
            }
          }
          tracer.kernel_finalizer(k->get_trace_info());
          delete k->entry();
          delete k;
          kernels_info.erase(kernels_info.begin()+j);
          break;
        }
      }
      assert(k);
      m_gpgpu_sim->print_stats();
    }

    if (sim_cycles) {
      m_gpgpu_sim->update_stats();
      m_gpgpu_context->print_simulation_time();
    }

    if (m_gpgpu_sim->cycle_insn_cta_max_hit()) {
      printf("GPGPU-Sim: ** break due to reaching the maximum cycles (or "
             "instructions) **\n");
      m_gpgpu_sim->update_stats();
      m_gpgpu_sim->print_stats();
      fflush(stdout);
      break;
    }
  }

  std::cout <<"TOTAL CYCLES TAKEN "<<cycles_passed<<"\n";
  cout <<"gpu_sim_insn_test "<<gpu_sim_insn_test<<"\n";
  cout <<"total_mem_inst "<<total_mem_inst<<"\n";
  cout <<"total_dep_found "<<total_dep_found<<"\n";
  cout <<"total_inst_going_for_renaming "<<total_inst_going_for_renaming<<"\n";
  cout <<"total_inst_renamed "<<total_inst_renamed<<"\n";
  cout <<"tot_mem_dep_checks "<<tot_mem_dep_checks<<"\n";
  cout <<"tot_mem_dep_true "<<tot_mem_dep_true<<"\n";
  cout <<"tot_mem_dep_false "<<tot_mem_dep_false<<"\n";
  //cout <<"tot_issues_ILP "<<tot_issues_ILP<<"\n";
  cout <<"stall_cycles "<<stall_cycles<<"\n";
  cout <<"stall_cycles_1_sched "<<stall_cycles_1_sched<<"\n";
  cout <<"stall_cycles_2_sched "<<stall_cycles_2_sched<<"\n";
  cout <<"stall_cycles_3_sched "<<stall_cycles_3_sched<<"\n";
  cout <<"stall_cycles_4_sched "<<stall_cycles_4_sched<<"\n";
  cout <<"total_times_in_cycle "<<total_times_in_cycle<<"\n";
  cout <<"tot_issues_OOO "<<tot_issues_OOO<<"\n";
  cout <<"tot_in_order_stall "<<tot_in_order_stall<<"\n";
  cout <<"tot_inst_OOO_because_dep "<<tot_inst_OOO_because_dep<<"\n";
  cout <<"WAR_and_WAW_stalls "<<WAR_and_WAW_stalls<<"\n";
  cout <<"tot_inst_exec "<<tot_inst_exec<<" independent_instruction_count "<<independent_instruction_count<<" "<<(float)(independent_instruction_count)/(float)tot_inst_exec*100<<"\n";
  cout <<"stalled_inst_tot "<<stalled_inst_tot<<" independent_instruction_stuck_count "<<independent_instruction_stuck_count<<" branch_insts "<<" "<<branch_insts<<"\n";
  cout<<"STALLING_VALUES "<<whole_sched_stall<<" "<<total_times_in_cycle<<" "<<((((float)whole_sched_stall/(float)total_times_in_cycle))*100)<<"\n";
  //cout<<"total_sched_cycles "<<total_times_in_cycle<<" total_backend_stall_cycles "<<total_backend_stall_cycles<<" per "<<(float)total_backend_stall_cycles/(float)total_times_in_cycle*100<<"\n";

  std::cout<<"INDEP_INST_LOCATION ";
  for(int i=0;i<32;i++)
  {
    std::cout <<i<<":"<<indep_inst_found_loc[i]<<" ; ";
  }

  std::cout<<"\n";
  std::cout<<"INDEP_INST_STREAMS_FOUND ";
  for(int i=0;i<32;i++)
  {
    std::cout <<i<<":"<<indep_instruction_streams_found_loc[i]<<" ; ";
  }

  std::cout<<"\n";
    cout<<"total_sched_cycles "<<total_times_in_cycle<<" "<<total_backend_stall_cycles_and_front_end<<" "<<total_backend_not_stall_cycles_but_front_end<<" "<<total_backend_stall_cycles_not_front_end<<" "<<none_stalls<<"\n";

  cout <<"STALL_STATS_PER_SCHED\n";

  cout <<"SM sched mem_data comp_data ibuffer comp_str mem_str waiting_warp_stall idle_stall total_cycles total_stalls total_issues\n";
  for(int i = 0; i<= SM_num; i++)
  {
    for(int j=0; j<4; j++)
    {
      cout <<i<<" "<<j<<" ";
      for(int k = 0;k<numstall; k++)
      {
        cout <<stallData[i][j][k]<<" ";
      }
      cout <<"\n";
    }
  }

  // we print this message to inform the gpgpu-simulation stats_collect script
  // that we are done
  printf("GPGPU-Sim: *** simulation thread exiting ***\n");
  printf("GPGPU-Sim: *** exit detected ***\n");
  fflush(stdout);

  return 0;
}


trace_kernel_info_t *create_kernel_info( kernel_trace_t* kernel_trace_info,
		                      gpgpu_context *m_gpgpu_context, class trace_config *config,
							  trace_parser *parser){

  gpgpu_ptx_sim_info info;
  info.smem = kernel_trace_info->shmem;
  info.regs = kernel_trace_info->nregs;
  dim3 gridDim(kernel_trace_info->grid_dim_x, kernel_trace_info->grid_dim_y, kernel_trace_info->grid_dim_z);
  dim3 blockDim(kernel_trace_info->tb_dim_x, kernel_trace_info->tb_dim_y, kernel_trace_info->tb_dim_z);
  trace_function_info *function_info =
      new trace_function_info(info, m_gpgpu_context);
  function_info->set_name(kernel_trace_info->kernel_name.c_str());
  trace_kernel_info_t *kernel_info =
      new trace_kernel_info_t(gridDim, blockDim, function_info,
    		  parser, config, kernel_trace_info);

  return kernel_info;
}

gpgpu_sim *gpgpu_trace_sim_init_perf_model(int argc, const char *argv[],
                                           gpgpu_context *m_gpgpu_context,
                                           trace_config *m_config) {
  srand(1);
  print_splash();

  option_parser_t opp = option_parser_create();

  m_gpgpu_context->ptx_reg_options(opp);
  m_gpgpu_context->func_sim->ptx_opcocde_latency_options(opp);

  icnt_reg_options(opp);

  m_gpgpu_context->the_gpgpusim->g_the_gpu_config =
      new gpgpu_sim_config(m_gpgpu_context);
  m_gpgpu_context->the_gpgpusim->g_the_gpu_config->reg_options(
      opp); // register GPU microrachitecture options
  m_config->reg_options(opp);

  option_parser_cmdline(opp, argc, argv); // parse configuration options
  fprintf(stdout, "GPGPU-Sim: Configuration options:\n\n");
  option_parser_print(opp, stdout);
  // Set the Numeric locale to a standard locale where a decimal point is a
  // "dot" not a "comma" so it does the parsing correctly independent of the
  // system environment variables
  assert(setlocale(LC_NUMERIC, "C"));
  m_gpgpu_context->the_gpgpusim->g_the_gpu_config->init();

  m_gpgpu_context->the_gpgpusim->g_the_gpu = new trace_gpgpu_sim(
      *(m_gpgpu_context->the_gpgpusim->g_the_gpu_config), m_gpgpu_context);

  m_gpgpu_context->the_gpgpusim->g_stream_manager =
      new stream_manager((m_gpgpu_context->the_gpgpusim->g_the_gpu),
                         m_gpgpu_context->func_sim->g_cuda_launch_blocking);

  m_gpgpu_context->the_gpgpusim->g_simulation_starttime = time((time_t *)NULL);

  return m_gpgpu_context->the_gpgpusim->g_the_gpu;
}
