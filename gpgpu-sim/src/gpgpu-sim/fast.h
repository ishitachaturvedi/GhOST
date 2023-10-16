#include<vector>
using std::vector;
#include <stdio.h>
#include <fstream>

extern bool replay_stall;

extern long total_mem_inst;
extern long total_dep_found;
extern long total_inst_going_for_renaming;
extern long total_inst_renamed;

extern long total_times_in_cycle;
//initialise vector of vectors as stallData[Warp #][stall #]
extern vector<vector<vector<int>>>stallData;
extern vector<vector<int>>act_warp;
extern vector<vector<int>>issued_warp;
extern vector<vector<vector<int>>> str_status;
extern vector<int>warp_issue;
extern vector<int>icnt_pressure;
extern vector<unsigned>warp_inst_num;
extern vector<int>warps_cannot_be_issued;
extern vector<int> indep_pc_num_push_all_stalling_inst;
extern vector<int>stall_consolidated;

extern vector<int>indep_inst_found_loc;
extern vector<int>indep_instruction_streams_found_loc;

extern int branch_insts;

extern int time_bw_inst;
extern int reached_barrier;

extern int tot_mem_dep_checks;
extern int tot_mem_dep_true;
extern int tot_mem_dep_false;
//extern int tot_issues_ILP;
extern int tot_issues_OOO;

extern long long max_active;
extern long long actw;
extern long long max_warps_act;
extern long long max_sid;
extern long long num_of_schedulers;
extern long long numstall;
extern long long cycles_passed;
extern long long inst_counter;
extern long long print_on;
extern long long going_from_shader_to_mem;
extern long long stall_cycles;
extern long long stall_cycles_1_sched;
extern long long stall_cycles_2_sched;
extern long long stall_cycles_3_sched;
extern long long stall_cycles_4_sched;
extern int stall_happened;
extern int whole_sched_stall;
extern int warps_found;
extern long long tot_icnt_buffer;
extern long long tot_inst_exec;
extern long long tot_cycles_exec_all_SM;
extern long long tot_inst_ret;
extern int total_warps;
extern long independent_instruction_count;
extern long independent_instruction_stuck_count;
extern long stalled_inst_tot;

// Stats collection
extern long long  mem_data_stall;
extern long long  comp_data_stall;
extern long long  ibuffer_stall;
extern long long  comp_str_stall;
extern long long  mem_str_stall;
extern long long  waiting_warp;
extern long long  idle;
extern long long  SM_num;
extern long long  tot_inst_OOO_because_dep;
extern long long ooo_opp_kernel;
extern long WAR_and_WAW_stalls;
extern bool WAR_or_WAW_found;
extern bool print_stall_data;
extern long SHADER_ICNT_PUSH;
extern long long NO_INST_ISSUE;
extern long long opp_for_ooo;
extern long long opp_for_mem;

// MEM STATS COLLECTION

extern long long present_ongoing_cycle;
extern long long opcode_tracer;

extern long long DEB_BUFFER_SIZE;

extern bool stalled_on_DEB_dependence;
extern long long cannot_issue_warp_DEB_dep;
extern long long cannot_issue_warp_DEB_dep_kernel;

extern int inst_counter_warp;
extern int inst_counter_warp_dec;

extern long long tot_cycles_wb;
extern long long tot_cycles_wb_done;
extern long long inst_issued_sid_0;

extern int control_hazard_count;
extern int control_hazard_count_kernel;

extern long long sync_inst_collision;
extern long long sync_inst_ibuffer_coll;
extern long long mem_inst_collision;
extern long long mem_inst_ibuffer_coll;

extern long long tot_in_order_stall;
extern long long memory_inst_stuck;

extern long inst_exec_total;
extern long gpu_sim_insn_test;
extern long gpu_sim_insn_test_run;

extern long tot_mem_inst_in_flight;
extern long mem_inst_in_flight;

extern long total_backend_stall_cycles_and_front_end;
extern long total_backend_stall_cycles_not_front_end;
extern long none_stalls;
extern long total_backend_not_stall_cycles_but_front_end;
