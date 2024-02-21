// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, Andrew Turner,
// Ali Bakhoda, Vijay Kandiah, Nikos Hardavellas, 
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern 
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
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

#ifndef SHADER_H
#define SHADER_H

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <bitset>
#include <deque>
#include <list>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include <memory>	
#include <iostream>

//#include "../cuda-sim/ptx.tab.h"

#include "../abstract_hardware_model.h"
#include "delayqueue.h"
#include "dram.h"
#include "gpu-cache.h"
#include "mem_fetch.h"
#include "scoreboard.h"
#include "stack.h"
#include "stats.h"
#include "traffic_breakdown.h"

#define NO_OP_FLAG 0xFF

/* READ_PACKET_SIZE:
   bytes: 6 address (flit can specify chanel so this gives up to ~2GB/channel,
   so good for now), 2 bytes   [shaderid + mshrid](14 bits) + req_size(0-2 bits
   if req_size variable) - so up to 2^14 = 16384 mshr total
 */

#define READ_PACKET_SIZE 8

// WRITE_PACKET_SIZE: bytes: 6 address, 2 miscelaneous.
#define WRITE_PACKET_SIZE 8

#define WRITE_MASK_SIZE 8

class gpgpu_context;

enum exec_unit_type_t {
  NONE = 0,
  SP = 1,
  SFU = 2,
  MEM = 3,
  DP = 4,
  INT = 5,
  TENSOR = 6,
  SPECIALIZED = 7
};

class thread_ctx_t {
 public:
  unsigned m_cta_id;  // hardware CTA this thread belongs

  // per thread stats (ac stands for accumulative).
  unsigned n_insn;
  unsigned n_insn_ac;
  unsigned n_l1_mis_ac;
  unsigned n_l1_mrghit_ac;
  unsigned n_l1_access_ac;

  bool m_active;
};

class shd_warp_t {
 public:
  shd_warp_t(class shader_core_ctx *shader, unsigned warp_size, int ib_size, int ib_size_in, int ib_size_ooo)	
      : m_shader(shader), m_warp_size(warp_size) {	
    m_stores_outstanding = 0;	
    m_inst_in_pipeline = 0;	
    warp_inst_retired = 0;	
    num_inst_exec = 0;	
    m_ibuffer.resize(ib_size);	
    m_ibuffer_streaming.resize(ib_size);	
    ibuffer_OOO.resize(ib_size);	
    replay_buffer.resize(ib_size);	
    //replay_buffer_mem.resize(ib_size);	
    replay_buffer_OOO.resize(ib_size);	
    SCHED_IBUFFER_SIZE = 4;
    ibuffer_OOO_1.resize(SCHED_IBUFFER_SIZE);	// big change Ishita
    IBUFFER_SIZE = ib_size;	
    IBUFFER_SIZE_IN_ORDER = ib_size_in;	
    IBUFFER_SIZE_OOO_ORDER = ib_size_ooo;	
    reset();	
  }
  void reset() {
    assert(m_stores_outstanding == 0);
    assert(m_inst_in_pipeline == 0);
    m_imiss_pending = false;
    m_warp_id = (unsigned)-1;
    m_dynamic_warp_id = (unsigned)-1;
    num_inst_exec_warp = 0;
    n_completed = m_warp_size;
    m_n_atomic = 0;
    m_membar = false;
    m_done_exit = true;
    m_last_fetch = 0;
    m_next = 0;

    replay_m_next = 0;		
    replay_m_fill_next = 0;		
    ibuffer_m_fill_next_streaming = 0;	
    replay_m_next_mem = 0;		
    ibuffer_m_next_streaming = 0;	
    replay_m_fill_next_mem = 0;	
    replay_buffer_head = 0;	
    replay_buffer_tail = 0;	
    replay_buffer_store_inst = 0;	
    replay_buffer_memory_inst = 0;	
    load_to_TLB = 0;
    ib_replay_buffer_tail_DEB_OOO = 0;	
    ibuffer_tail = 0;	
    ibuffer_tail_full_OOO = 0;	
    ibuffer_store_inst = 0;	
    ibuffer_memory_inst = 0;	
    ibuffer_control_inst = 0;	
    inst_counter = 0;
    predicate_inst = 0;
    stalls_for_inst = 0;
    ibuffer_inst_number = 0;	
    stall_points = 0;	
    stall_point_PC = -1;	
    // Jin: cdp support	
    m_cdp_latency = 0;	
    m_cdp_dummy = false;	
    num_inst_exec = 0;	
    num_warp_inst = 0;	
    warp_wb = 0;
    ib_enter = 0;
    warp_inst_retired = 0;	
    stalls_between_issues = 0;	
    stalled_on_replay = -1;	
    last_pc_decoded_streaming_test = 0;	
    num_inst_OOO = 0;
    stop_fetching_when_inst_0 = 0;
    //SCHED_IBUFFER_SIZE = 64;

    ibuffer_1_flush();
    ibuffer_flush_OOO();
  }
  void init(address_type start_pc, unsigned cta_id, unsigned wid,
            const std::bitset<MAX_WARP_SIZE> &active,
            unsigned dynamic_warp_id) {
    m_cta_id = cta_id;
    m_warp_id = wid;
    m_dynamic_warp_id = dynamic_warp_id;
    m_next_pc = start_pc;
    assert(n_completed >= active.count());
    assert(n_completed <= m_warp_size);
    n_completed -= active.count();  // active threads are not yet completed
    m_active_threads = active;
    m_done_exit = false;

    // Jin: cdp support
    m_cdp_latency = 0;
    m_cdp_dummy = false;
    num_inst_exec = 0;	
    num_warp_inst = 0;	
    warp_wb = 0;
    ib_enter = 0;
    num_inst_exec_warp = 0;	
    replay_m_next = 0;		
    replay_m_fill_next = 0;		
    ibuffer_m_fill_next_streaming = 0;	
    replay_m_next_mem = 0;		
    ibuffer_m_next_streaming = 0;	
    replay_m_fill_next_mem = 0;	
    last_pc_decoded_streaming = start_pc;	
    last_pc_decoded_streaming_test = 0;	
    warp_inst_retired = 0;	
    replay_buffer_head = 0;	
    replay_buffer_tail = 0;	
    replay_buffer_store_inst = 0;	
    replay_buffer_memory_inst = 0;	
    load_to_TLB = 0;
    ib_replay_buffer_tail_DEB_OOO = 0;	
    ibuffer_tail = 0;	
    ibuffer_tail_full_OOO = 0;	
    ibuffer_store_inst = 0;	
    ibuffer_memory_inst = 0;	
    stall_points = 0;	
    stall_point_PC = -1;	
    ibuffer_control_inst = 0;	
    inst_counter = 0;
    predicate_inst = 0;
    stalls_for_inst = 0;
    ibuffer_inst_number = 0;	
    stalls_between_issues = 0;	
    stalled_on_replay = -1;	
    //warp_stuck_replay = 0;	
    stop_fetching_when_inst_0 = 0;
    num_inst_OOO = 0;
    //SCHED_IBUFFER_SIZE = 64;

    ibuffer_1_flush();
    ibuffer_flush_OOO();
  }


  bool functional_done() const;	
  bool functional_done_check() const;	
  bool waiting();  // not const due to membar	
  bool check_waiting();		
  bool waiting_barrier();		
  bool waiting_fence();		
  bool waiting_idle();	
  bool hardware_done() const;	
  bool hardware_done_print() const;

  bool done_exit() const { return m_done_exit; }
  void set_done_exit() { m_done_exit = true; }

  void print(FILE *fout) const;
  void print_ibuffer(FILE *fout) const;

  unsigned get_n_completed() const { return n_completed; }
  void set_completed(unsigned lane) {
    assert(m_active_threads.test(lane));
    m_active_threads.reset(lane);
    n_completed++;
  }

  void set_last_fetch(unsigned long long sim_cycle) {
    m_last_fetch = sim_cycle;
  }

  unsigned get_n_atomic() const { return m_n_atomic; }
  void inc_n_atomic() { m_n_atomic++; }
  void dec_n_atomic(unsigned n) { m_n_atomic -= n; }

  void set_membar() { m_membar = true; }
  void clear_membar() { m_membar = false; }
  bool get_membar() const { return m_membar; }
  virtual address_type get_pc() const { return m_next_pc; }
  
  void set_streaming_pc_hazard(address_type pc, int warp_id, const int &fetch_warp, const bool &valid)	
  {	
    last_pc_decoded_streaming = pc;	
  }	
  void get_last_pc_decoded_streaming()	
  {	
    std::cout <<"CHECK_last_pc_decoded_streaming "<<last_pc_decoded_streaming<<"\n";	
  }
  virtual kernel_info_t* get_kernel_info() const;
  void set_next_pc(address_type pc) { m_next_pc = pc; }

  void store_info_of_last_inst_at_barrier(const warp_inst_t *pI) {
    m_inst_at_barrier = *pI;
  }
  warp_inst_t *restore_info_of_last_inst_at_barrier() {
    return &m_inst_at_barrier;
  }

  void ibuffer_fill(unsigned slot, const warp_inst_t *pI) {
    assert(slot < IBUFFER_SIZE);
    m_ibuffer[slot].m_inst = pI;
    m_ibuffer[slot].m_valid = true;
    m_next = 0;
  }
  bool ibuffer_empty() const {
    for (unsigned i = 0; i < IBUFFER_SIZE; i++)
      if (m_ibuffer[i].m_valid) return false;
    return true;
  }
  void ibuffer_flush() {
    for (unsigned i = 0; i < IBUFFER_SIZE; i++) {
      if (m_ibuffer[i].m_valid) dec_inst_in_pipeline();
      m_ibuffer[i].m_inst = NULL;
      m_ibuffer[i].m_valid = false;
    }
  }
  const warp_inst_t *ibuffer_next_inst() { return m_ibuffer[m_next].m_inst; }
  bool ibuffer_next_valid() { return m_ibuffer[m_next].m_valid; }
  int freed_pc()	
  {	
    return m_ibuffer[m_next].m_inst->pc;	
  }	
  void set_ibuffer_pushed(int cycle)	
  {	
    m_ibuffer[m_next].m_pushed_in = cycle;	
  }	

  int get_ibuffer_pushed()	
  {	
    return m_ibuffer[m_next].m_pushed_in;	
  }
  void ibuffer_free() {
    m_ibuffer[m_next].m_inst = NULL;
    m_ibuffer[m_next].m_valid = false;
  }
  void ibuffer_step() { m_next = (m_next + 1) % IBUFFER_SIZE; }


  bool* ibuffer_inst_addr(int slot) { return &m_ibuffer[slot].m_valid;}

  void increase_exec_inst() {num_inst_exec = num_inst_exec + 1;}
  long long return_num_exec_inst() {return num_inst_exec;}


  // additional ibuffer functions 		
  

  bool replay_buffer_empty() const {	
    for (unsigned i = 0; i < REPLAY_BUFFER_SIZE; i++)	
      if (replay_buffer[i].m_valid) return false;	
    return true;	
  }	

  	
  void replay_buffer_step() { 	
    if(replay_m_next + 1 < REPLAY_BUFFER_SIZE)	
      replay_m_next = replay_m_next + 1;	
    else  	
      replay_m_next = 0;	
  }	

  // end of REPLAY BUFFER FUNCTIONS	

  // complete OOO Instruction buffer

  bool ibuffer_empty_streaming_OOO() const {
    for (unsigned i = 0; i < IBUFFER_SIZE_IN_ORDER; i++)
      if (ibuffer_OOO[i].m_valid) return false;
    return true;
  }

  bool ibuffer_empty_streaming_full_OOO() const {
    int total_size = IBUFFER_SIZE_IN_ORDER + IBUFFER_SIZE_OOO_ORDER;
    for (unsigned i = 0; i < total_size; i++)
      if (ibuffer_OOO[i].m_valid) return false;
    return true;
  }

  bool ibuffer_1_empty_streaming_full_OOO() const {
    //int total_size = IBUFFER_SIZE;
    int total_size = ibuffer_OOO_1.size();
    for (unsigned i = 0; i < total_size; i++)
      if (ibuffer_OOO_1[i].m_valid) return false;
    return true;
  }

  int ibuffer_1_empty_streaming_full_OOO_count() const {
    //int total_size = IBUFFER_SIZE;
    std::cout<<"SIZE_HERE "<<ibuffer_OOO_1.size()<<" ";
    int counter = 0;
    int total_size = ibuffer_OOO_1.size();
    for (unsigned i = 0; i < total_size; i++)
      if (ibuffer_OOO_1[i].m_valid)
      {
        counter++;
      }
    return counter;
  }

  bool ibuffer_has_space_streaming_OOO() const {
    for (unsigned i = 0; i < IBUFFER_SIZE_IN_ORDER; i++)	
      if (!ibuffer_OOO[i].m_valid) 	
        return true;
    return false;	
  }

  bool ibuffer_has_space_streaming_full_OOO() const {
    //int total_size = IBUFFER_SIZE_IN_ORDER + IBUFFER_SIZE_OOO_ORDER;
    int total_size = ibuffer_OOO.size();
    for (unsigned i = 0; i < total_size; i++)	
      if (!ibuffer_OOO[i].m_valid) 	
        return true;
    return false;	
  }

  bool ibuffer_has_space_streaming_empty_on_WB() const {
    //int total_size = IBUFFER_SIZE_IN_ORDER + IBUFFER_SIZE_OOO_ORDER;
    int total_size = ibuffer_OOO.size();
    for (unsigned i = 0; i < total_size; i++)	
      if (!ibuffer_OOO[i].m_index_full_OOO) 	
        return true;
    return false;	
  }

  int val_ibuffer_has_space_streaming_empty_on_WB() const {
    //int total_size = IBUFFER_SIZE_IN_ORDER + IBUFFER_SIZE_OOO_ORDER;
    int total_size = ibuffer_OOO.size();
    for (unsigned i = 0; i < total_size; i++)	
      if (!ibuffer_OOO[i].m_index_full_OOO) 	
        return i;
    return -1;	
  }

  bool ibuffer_1_has_space_streaming_full_OOO() const {
    //int total_size = IBUFFER_SIZE_IN_ORDER + IBUFFER_SIZE_OOO_ORDER;
    int total_size = ibuffer_OOO_1.size();
    for (unsigned i = 0; i < total_size; i++)	
      if (!ibuffer_OOO_1[i].m_valid) 	
        return true;
    return false;	
  }

  void ibuffer_1_fill(unsigned slot, const warp_inst_t *pI, int PC) {	
    ibuffer_OOO_1[slot].m_inst = pI;	
    ibuffer_OOO_1[slot].m_valid = true;	
    ibuffer_OOO_1[slot].m_pc = PC;
    ibuffer_OOO_1[slot].m_pushed_in = 2;

  }	


  void ibuffer_1_flush() {	
    int total_size = ibuffer_OOO_1.size();
    for (unsigned i = 0; i < total_size; i++) {	
      {
        ibuffer_OOO_1[i].m_inst = NULL;	
        ibuffer_OOO_1[i].m_valid = false;	
        ibuffer_OOO_1[i].m_pc = NULL;	
        ibuffer_OOO_1[i].m_pushed_in = -1;
      }
    }	
    stop_fetching_when_inst_0 = 0;
  }

  void ibuffer_fill_OOO(unsigned slot, const warp_inst_t *pI, int PC,int tail,int tail_full_OOO, int num_stores,int warp, int mem_count, int m_control_count, int cycle, int predicate_inst, int inst_counter, bool isNonIdempotent, int loadInsns) {	
    //assert(slot < IBUFFER_SIZE_IN_ORDER);	
    ibuffer_OOO[slot].m_inst = pI;	
    ibuffer_OOO[slot].m_valid = true;	
    ibuffer_OOO[slot].m_pc = PC;
    ibuffer_OOO[slot].m_index = tail;
    ibuffer_OOO[slot].m_index_full_OOO = 1; // use this to keep track of valid instructions which might be issued but cannot be removed from the IB till the oldest inst has a writeback
    ibuffer_OOO[slot].m_stores = num_stores;	
    ibuffer_OOO[slot].m_meminst = mem_count;
    ibuffer_OOO[slot].control_check = 0;
    ibuffer_OOO[slot].control_check_in_order = m_control_count;
    ibuffer_OOO[slot].m_control_inst_OOO = false; // mark instruction as true once is completes execution so we can remove instructions from IB correctly
    ibuffer_OOO[slot].m_pushed_in_OOO = cycle;
    ibuffer_OOO[slot].m_DEB_bit = predicate_inst;
    ibuffer_OOO[slot].m_inst_number = -1;
    ibuffer_OOO[slot].m_stall_cycles = -1;
    ibuffer_OOO[slot].mem_dep = -1;
    ibuffer_OOO[slot].m_pushed_out = 0;
    ibuffer_OOO[slot].m_inst_counter = inst_counter;
    ibuffer_OOO[slot].isIdempotent = isNonIdempotent;
    ibuffer_OOO[slot].priorLoads = loadInsns;
  }

//issue_warp_push_in_DEB_IB_OOO
  void ibuffer_fill_OOO_replay(unsigned slot, int tail, int index) {	
    assert(slot < (IBUFFER_SIZE_OOO_ORDER + IBUFFER_SIZE_IN_ORDER));	
    ibuffer_OOO[slot].m_inst = ibuffer_OOO[index].m_inst;	
    ibuffer_OOO[slot].m_valid = true;	
    ibuffer_OOO[slot].m_pc = ibuffer_OOO[index].m_pc;
    ibuffer_OOO[slot].m_index = tail;
    ibuffer_OOO[slot].m_stores = ibuffer_OOO[index].m_stores;	
    ibuffer_OOO[slot].m_meminst = ibuffer_OOO[index].m_meminst;
    ibuffer_OOO[slot].control_check = ibuffer_OOO[index].control_check;
    ibuffer_OOO[slot].m_control_inst_OOO = false;
    ibuffer_OOO[slot].m_pushed_out = ibuffer_OOO[index].m_pushed_out;
    ibuffer_OOO[slot].m_pushed_in_OOO = ibuffer_OOO[index].m_pushed_in_OOO;
    ibuffer_OOO[slot].m_DEB_bit = ibuffer_OOO[index].m_DEB_bit;
    ibuffer_OOO[slot].m_inst_number = ibuffer_OOO[index].m_inst_number;
    ibuffer_OOO[slot].m_stall_cycles = ibuffer_OOO[index].m_stall_cycles;
    ibuffer_OOO[slot].mem_dep = ibuffer_OOO[index].mem_dep;
    ibuffer_OOO[slot].control_check_in_order = ibuffer_OOO[index].control_check_in_order;
    ibuffer_OOO[slot].m_inst_counter = ibuffer_OOO[index].m_inst_counter;
    ibuffer_OOO[slot].isIdempotent = ibuffer_OOO[index].isIdempotent;
    ibuffer_OOO[slot].priorLoads = ibuffer_OOO[index].priorLoads;
  }	

  bool getIdempotent(int idx)
  {
    return ibuffer_OOO[idx].isIdempotent;
  }

  int getPrioirLoads(int idx)
  {
    return ibuffer_OOO[idx].priorLoads;
  }

  void reducePriorLoads()
  {
    int tot_size = ibuffer_OOO.size();
    for (unsigned i = 0; i < tot_size ; i++)	
    {
      ibuffer_OOO[i].priorLoads--;
    }
  }

  int get_ibuffer_1_pc(int inst_loc)
  {
    return ibuffer_OOO_1[inst_loc].m_pc;
  }

  int get_ibuffer_1_set_inst_invalid(int inst_loc)
  {
    ibuffer_OOO_1[inst_loc].m_valid = false;
  }

  int get_mem_inst_OOO(unsigned slot)
  {
    return ibuffer_OOO[slot].m_meminst;
  }

  int ibuffer_DEB_bit(int loc)
  {
    return ibuffer_OOO[loc].m_DEB_bit;
  }

  int get_ibuffer_inst_number(int loc)
  {
    return ibuffer_OOO[loc].m_inst_number;
  }

  void set_ibuffer_inst_number(int loc)
  {
    ibuffer_OOO[loc].m_inst_number = ibuffer_inst_number;
  }

  void set_stall_cycle(int cycle, int loc)
  {
    ibuffer_OOO[loc].m_stall_cycles = cycle;
  }

  int get_stall_cycle(int loc)
  {
    return ibuffer_OOO[loc].m_stall_cycles;
  }

  int get_ibuffer_stall_cycles(int loc)
  {
    return ibuffer_OOO[loc].m_stall_cycles;
  }

  void increase_ibuffer_stall_cycle(int loc)
  {
    ibuffer_OOO[loc].m_stall_cycles++;
  }

  void increase_ibuffer_stall_cycle_for_all_inst_IB()
  {
     int total_size = IBUFFER_SIZE_IN_ORDER + IBUFFER_SIZE_OOO_ORDER;
    for (unsigned i = 0; i < total_size; i++)	
    {
      if (ibuffer_OOO[i].m_valid && get_control_bit(i))
        ibuffer_OOO[i].m_stall_cycles++;
    }
  }

  void set_mem_dep(bool dep_found, int loc)
  {
    ibuffer_OOO[loc].mem_dep = dep_found;
  }

  int get_mem_dep(int loc)
  {
    return ibuffer_OOO[loc].mem_dep;
  }

  void increase_ibuffer_stall_cycle_for_inst_zero()
  {
     int total_size = IBUFFER_SIZE_IN_ORDER;
    for (unsigned i = 0; i < total_size; i++)	
    {
      if (ibuffer_OOO[i].m_valid)
      {
        ibuffer_OOO[i].m_stall_cycles++;
        break;
      }
    }
  }

  void increase_ibuffer_stall_cycle_for_deb_OOO()
  {
     int total_size = IBUFFER_SIZE_IN_ORDER + IBUFFER_SIZE_OOO_ORDER;
    for (unsigned i = IBUFFER_SIZE_IN_ORDER; i < total_size; i++)	
    {
      if (ibuffer_OOO[i].m_valid)
      {
        ibuffer_OOO[i].m_stall_cycles++;
        break;
      }
    }
  }

  void set_inst_complete(int idx)
  {
    ibuffer_OOO[idx].m_control_inst_OOO = true;
  }

  void set_inst_incomplete(int idx)
  {
    ibuffer_OOO[idx].m_control_inst_OOO = false;
  }

  int get_inst_complete(int idx)
  {
    return ibuffer_OOO[idx].m_control_inst_OOO;
  }

  void set_inst_as_free(int idx)
  {
    ibuffer_OOO[idx].m_index_full_OOO = false;
  }

  void set_inst_in_or_OOO(int idx, int val)
  {
    ibuffer_OOO[idx].m_index_full_OOO = val;
  }

  int get_inst_as_free(int idx)
  {
    return ibuffer_OOO[idx].m_index_full_OOO;
  }

  int get_inst_valid(int idx)
  {
    return ibuffer_OOO[idx].m_valid;
  }

  bool no_older_inst_for_issue(int inst_index)
  {
    int tot_size = ibuffer_OOO.size();
    for (unsigned i = 0; i < tot_size ; i++)	
    {
      int index = ibuffer_OOO[i].m_index;
      if (ibuffer_OOO[i].m_index_full_OOO && (index < inst_index))	
        return false;	
    }
    return true;	
  }

  int get_index_of_loc(int loc)
  {
    return ibuffer_OOO[loc].m_index;
  }

  int get_ibuffer_pushed_OOO_all(unsigned slot)
  {
    return ibuffer_OOO[slot].m_pushed_in_OOO;
  }

  int ibuffer_empty_idx_OOO() {	
    for (unsigned i = 0; i < IBUFFER_SIZE_IN_ORDER ; i++)	
    {
      if (!ibuffer_OOO[i].m_valid)	
        return i;
    }
    return -1;	
  }

  int ibuffer_empty_idx_FULL_OOO() {	
    int tot_size = ibuffer_OOO.size();
    for (unsigned i = 0; i < tot_size ; i++)	
      if (!ibuffer_OOO[i].m_valid)	
        return i;	
    return -1;	
  }

  int ibuffer_empty_idx_keep_in_IB() {	
    int tot_size = ibuffer_OOO.size();
    for (unsigned i = 0; i < tot_size ; i++)	
      if (!ibuffer_OOO[i].m_index_full_OOO)	
        return i;	
    return -1;	
  }

  int ibuffer_1_empty_idx_FULL_OOO() {	
    int tot_size = ibuffer_OOO_1.size();
    for (unsigned i = 0; i < tot_size ; i++)	
      if (!ibuffer_OOO_1[i].m_valid)	
        return i;	
    return -1;	
  }

  int ibuffer_empty_idx_DEB_OOO() {	
    int tot_size = IBUFFER_SIZE_IN_ORDER + IBUFFER_SIZE_OOO_ORDER;
    for (unsigned i = IBUFFER_SIZE_IN_ORDER; i < tot_size ; i++)	
      if (!ibuffer_OOO[i].m_valid)	
        return i;	
    return -1;	
  }

  void place_inst_in_IB(int inst_loc, warp_inst_t * pI)
  {
    ibuffer_OOO[inst_loc].m_inst = pI;
  }

  int ibuffer_has_inst_DEB_OOO() {	
    int tot_size = IBUFFER_SIZE_IN_ORDER + IBUFFER_SIZE_OOO_ORDER;
    for (unsigned i = IBUFFER_SIZE_IN_ORDER; i < tot_size ; i++)	
      if (ibuffer_OOO[i].m_valid)	
        return true;	
    return false;	
  }

  int ibuffer_count_OOO() const {	
    int counter = 0;	
    for (unsigned i = 0; i < IBUFFER_SIZE_IN_ORDER; i++)	
      if (ibuffer_OOO[i].m_valid) 	
        counter = counter + 1;	
    return counter;	
  }	

  int ibuffer_tot_size_OOO() const {
    return IBUFFER_SIZE_IN_ORDER;
  }

  int ibuffer_tot_size_full_OOO() const {
    int total_size = IBUFFER_SIZE_IN_ORDER + IBUFFER_SIZE_OOO_ORDER;
    return total_size;
  }

  int replay_tot_size_OOO() const {
    return REPLAY_BUFFER_SIZE;
  }

  int ibuffer_space_OOO() const {
    int counter = 0;	
    for (unsigned i = 0; i < IBUFFER_SIZE_IN_ORDER; i++)	
      if (!ibuffer_OOO[i].m_valid) 	
        counter = counter + 1;	
    return counter;	
  }

  std::vector<int> get_index_with_store_issue_DEB_IN(int inst_index)
  {
    std::vector<int> collision_index;
    for (unsigned i = 0; i < IBUFFER_SIZE_IN_ORDER; i++)
    {
      if (ibuffer_OOO[i].m_valid && ibuffer_OOO[i].m_index < inst_index && ibuffer_OOO[i].m_inst->is_store())
      {
        collision_index.push_back(i);
      }
    }
    for (unsigned i = 0; i < IBUFFER_SIZE_OOO_ORDER; i++)
    {
      if (ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_valid && ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_inst->is_store())
      {
        collision_index.push_back(IBUFFER_SIZE_IN_ORDER+i);
      }
    }
    return collision_index;
  }

  std::vector<int> get_index_with_store_issue_DEB_OOO(int inst_index)
  {
    std::vector<int> collision_index;
    for (unsigned i = 0; i < IBUFFER_SIZE_OOO_ORDER; i++)
    {
      if (ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_valid && ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_inst->is_store())
      {
        collision_index.push_back(IBUFFER_SIZE_IN_ORDER+i);
      }
    }
    return collision_index;
  }

  std::vector<int> get_index_with_store_issue_IB(int inst_index)
  {
    std::vector<int> collision_index;
    for (unsigned i = 0; i < IBUFFER_SIZE; i++)
    {
      if (ibuffer_OOO[i].m_valid && ibuffer_OOO[i].m_index < inst_index && ibuffer_OOO[i].m_inst->is_store())
      {
        collision_index.push_back(i);
      }
    }

    return collision_index;
  }

  std::vector<int> get_index_with_mem_issue_DEB_IN(int inst_index)
  {
    std::vector<int> collision_index;
    for (unsigned i = 0; i < IBUFFER_SIZE_IN_ORDER; i++)
    {
      if (ibuffer_OOO[i].m_valid && ibuffer_OOO[i].m_index < inst_index && (ibuffer_OOO[i].m_inst->is_store() || ibuffer_OOO[i].m_inst->is_load()))
      {
        collision_index.push_back(i);
      }
    }
    for (unsigned i = 0; i < IBUFFER_SIZE_OOO_ORDER; i++)
    {
      if (ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_valid && (ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_inst->is_store() || ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_inst->is_load()))
      {
        collision_index.push_back(IBUFFER_SIZE_IN_ORDER+i);
      }
    }
    return collision_index;
  }

  std::vector<int> get_index_with_mem_issue_DEB_OOO(int inst_index)
  {
    std::vector<int> collision_index;
    for (unsigned i = 0; i < IBUFFER_SIZE_OOO_ORDER; i++)
    {
      if (ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_valid && (ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_inst->is_store() || ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_inst->is_load()))
      {
        collision_index.push_back(IBUFFER_SIZE_IN_ORDER+i);
      }
    }
    return collision_index;
  }

  std::vector<int> get_index_with_mem_issue_IB(int inst_index)
  {
    std::vector<int> collision_index;
    for (unsigned i = 0; i < IBUFFER_SIZE; i++)
    {
      if (ibuffer_OOO[i].m_valid && ibuffer_OOO[i].m_index < inst_index && (ibuffer_OOO[i].m_inst->is_store() || ibuffer_OOO[i].m_inst->is_load()))
      {
        collision_index.push_back(i);
      }
    }
    return collision_index;
  }


  bool ibuffer_empty_OOO() const {	
    //for (unsigned i = 0; i < IBUFFER_SIZE_IN_ORDER; i++)	
    for (unsigned i = 0; i < ibuffer_OOO.size(); i++)	
      if (ibuffer_OOO[i].m_valid) return false;	
    return true;	
  }	

  bool ibuffer_empty_OOO_full() const {	
    //int total_size = IBUFFER_SIZE_IN_ORDER + IBUFFER_SIZE_OOO_ORDER;
    int total_size = ibuffer_OOO.size();
    for (unsigned i = 0; i < total_size; i++)	
      if (ibuffer_OOO[i].m_valid) return false;	
    return true;	
  }	

  void ibuffer_flush_OOO() {	
    //int total_size = IBUFFER_SIZE_IN_ORDER + IBUFFER_SIZE_OOO_ORDER;
    int total_size = ibuffer_OOO.size();
    for (unsigned i = 0; i < total_size; i++) {	
      //if(ibuffer_OOO[i].m_index >= index)
      {
        if (ibuffer_OOO[i].m_valid) dec_inst_in_pipeline();	
        ibuffer_OOO[i].m_inst = NULL;	
        ibuffer_OOO[i].m_valid = false;	
        ibuffer_OOO[i].m_pc = NULL;	
        ibuffer_OOO[i].m_index = -1;
        ibuffer_OOO[i].m_index_full_OOO = false;
        ibuffer_OOO[i].m_stores = 0;
        ibuffer_OOO[i].m_meminst = 0;
        ibuffer_OOO[i].control_check = 0;
        ibuffer_OOO[i].m_control_inst_OOO = false;
        ibuffer_OOO[i].m_pushed_out = 0;
        ibuffer_OOO[i].m_DEB_bit = 0;
        ibuffer_OOO[i].mem_dep = -1;
        ibuffer_OOO[i].control_check_in_order = 0;
        ibuffer_OOO[i].m_pushed_in_OOO = 0;
        ibuffer_OOO[i].m_inst_number = 0;
        ibuffer_OOO[i].m_stall_cycles = -1;
        ibuffer_OOO[i].isIdempotent = 0;
        ibuffer_OOO[i].priorLoads = 0;
      }
    }	
    ibuffer_store_inst = 0;
    ibuffer_memory_inst = 0;
    ibuffer_tail = 0;
    ibuffer_tail_full_OOO = 0;
    ibuffer_control_inst = 0;
    inst_counter = 0;
    predicate_inst = 0;
    stalls_for_inst = 0;
    replay_buffer_memory_inst = 0;
    load_to_TLB = 0;
  }	

  void ibuffer_flush_OOO_in_OOO() {	
    int total_size = IBUFFER_SIZE_IN_ORDER;
    for (unsigned i = 0; i < total_size; i++) {	
      //if(ibuffer_OOO[i].m_index >= index)
      {
        if (ibuffer_OOO[i].m_valid) dec_inst_in_pipeline();	
        ibuffer_OOO[i].m_inst = NULL;	
        ibuffer_OOO[i].m_valid = false;	
        ibuffer_OOO[i].m_pc = NULL;	
        ibuffer_OOO[i].m_index = -1;
        ibuffer_OOO[i].m_index_full_OOO = false;
        ibuffer_OOO[i].m_stores = 0;
        ibuffer_OOO[i].m_meminst = 0;
        ibuffer_OOO[i].control_check = 0;
        ibuffer_OOO[i].m_control_inst_OOO = false;
        ibuffer_OOO[i].m_pushed_out = 0;
        ibuffer_OOO[i].control_check_in_order = 0;
        ibuffer_OOO[i].m_DEB_bit = 0;
        ibuffer_OOO[i].mem_dep = -1;
        ibuffer_OOO[i].m_pushed_in_OOO = 0;
        ibuffer_OOO[i].m_inst_number = 0;
        ibuffer_OOO[i].m_stall_cycles = -1;
        ibuffer_OOO[i].isIdempotent = 0;
        ibuffer_OOO[i].priorLoads = 0;
      }
    }	
    ibuffer_store_inst = 0;
    ibuffer_memory_inst = 0;
    ibuffer_tail = 0;
    ibuffer_tail_full_OOO = 0;
    ibuffer_control_inst = 0;
    inst_counter = 0;
    predicate_inst = 0;
    stalls_for_inst = 0;
    replay_buffer_memory_inst = 0;
    load_to_TLB = 0;
  }	

  int get_store_inst_OOO(int slot)
  {
    return ibuffer_OOO[slot].m_stores;
  }

  int ibuffer_index_OOO(int index, int warp)
  {
    for (unsigned i = 0; i < IBUFFER_SIZE_IN_ORDER; i++)	
    {
      if (ibuffer_OOO[i].m_index == index)
        return i;
    }  
    return -1;
  }

  int ibuffer_index_OOO_free_on_oldest(int index, int warp)
  {
    for (unsigned i = 0; i < IBUFFER_SIZE_IN_ORDER; i++)	
    {
      if (ibuffer_OOO[i].m_index == index && ibuffer_OOO[i].m_index_full_OOO)
        return i;
    }  
    return -1;
  }

  std::vector<int>reorder_inst_for_memory()
  {
    std::vector<int> mem_inst_idx;
    std::vector<int> comp_inst_idx;
    std::vector<int> invalid_inst_idx;
    std::vector<int> reordered_inst_idx;
    for (unsigned i = 0; i < IBUFFER_SIZE; i++)	
    {
      if (ibuffer_OOO[i].m_valid == true)
      {
        if(ibuffer_OOO[i].m_inst->is_load() || ibuffer_OOO[i].m_inst->is_store())
          mem_inst_idx.push_back(ibuffer_OOO[i].m_index);
        else
          comp_inst_idx.push_back(ibuffer_OOO[i].m_index);
      }
      else
      {
        invalid_inst_idx.push_back(ibuffer_OOO[i].m_index);
      }
    }
    for(int i = 0; i<mem_inst_idx.size(); i++)
      reordered_inst_idx.push_back(mem_inst_idx[i]);
    for(int i = 0; i<comp_inst_idx.size(); i++)
      reordered_inst_idx.push_back(comp_inst_idx[i]);
    for(int i = 0; i<invalid_inst_idx.size(); i++)
      reordered_inst_idx.push_back(invalid_inst_idx[i]);
    
    return reordered_inst_idx;
  }

  int ibuffer_index_full_OOO(int index, int warp)
  {
    int total_size = IBUFFER_SIZE;
    for (unsigned i = 0; i < total_size; i++)	
    {
      if (ibuffer_OOO[i].m_index == index)
        return i;
    } 
    return -1;
  }


  const warp_inst_t *ibuffer_next_inst_OOO(int index) 
  { 
    return ibuffer_OOO[index].m_inst;
  }	

  const warp_inst_t *ibuffer_1_next_inst_OOO(int index) 
  { 
    return ibuffer_OOO_1[index].m_inst;
  }	

  void reduce_SIB_time(int index) 
  { 
    ibuffer_OOO_1[index].m_pushed_in--;
  }	

  int get_SIB_time(int index) 
  { 
    return ibuffer_OOO_1[index].m_pushed_in;
  }	

  bool ibuffer_next_valid_OOO(int index) { return ibuffer_OOO[index].m_valid; }	
  bool ibuffer_1_next_valid_OOO(int index) { return ibuffer_OOO_1[index].m_valid; }	
  
  void ibuffer_free_OOO(int index) {	
    ibuffer_OOO[index].m_inst = NULL;	
    ibuffer_OOO[index].m_valid = false;	
    ibuffer_OOO[index].m_pc = NULL;
    ibuffer_OOO[index].m_index = -1; // done for free_on_oldest
    //ibuffer_OOO[index].m_index_full_OOO =false;
    ibuffer_OOO[index].m_stores = 0;
    ibuffer_OOO[index].m_meminst = 0;
    ibuffer_OOO[index].control_check = 0;
    //ibuffer_OOO[index].m_control_inst_OOO = false;
    ibuffer_OOO[index].m_pushed_out = 0;
    ibuffer_OOO[index].control_check_in_order = 0;
    ibuffer_OOO[index].m_DEB_bit = 0;
    ibuffer_OOO[index].mem_dep = -1;
    ibuffer_OOO[index].m_pushed_in_OOO = 0;
    ibuffer_OOO[index].m_inst_number = 0;
    ibuffer_OOO[index].m_stall_cycles = -1;
    ibuffer_OOO[index].m_inst_counter = -1;
    ibuffer_OOO[index].isIdempotent = 0;
    ibuffer_OOO[index].priorLoads = 0;
  }	


  void ibuffer_free_OOO_free_on_oldest(int index) {	
    ibuffer_OOO[index].m_inst = NULL;	
    ibuffer_OOO[index].m_valid = false;	
    ibuffer_OOO[index].m_pc = NULL;
    //ibuffer_OOO[index].m_index = -1; // done for free_on_oldest
    //ibuffer_OOO[index].m_index_full_OOO =false;
    ibuffer_OOO[index].m_stores = 0;
    ibuffer_OOO[index].m_meminst = 0;
    ibuffer_OOO[index].control_check = 0;
    //ibuffer_OOO[index].m_control_inst_OOO = false;
    ibuffer_OOO[index].m_pushed_out = 0;
    ibuffer_OOO[index].control_check_in_order = 0;
    ibuffer_OOO[index].m_DEB_bit = 0;
    ibuffer_OOO[index].mem_dep = -1;
    ibuffer_OOO[index].m_pushed_in_OOO = 0;
    ibuffer_OOO[index].m_inst_number = 0;
    ibuffer_OOO[index].m_stall_cycles = -1;
    ibuffer_OOO[index].isIdempotent = 0;
    ibuffer_OOO[index].priorLoads = 0;
    //ibuffer_OOO[index].m_inst_counter = -1;
  }	

  int get_pc_DEB_IB_OOO(int index)
  {
    return ibuffer_OOO[index].m_pc;
  }

  bool get_control_bit(int index) {
    return ibuffer_OOO[index].control_check;
  }

  bool get_control_bit_in_order(int index) {
    return ibuffer_OOO[index].control_check_in_order;
  }

  bool get_predicate(int index) {
    return ibuffer_OOO[index].m_DEB_bit;
  }

  void set_control_bit(int index)
  {
    ibuffer_OOO[index].control_check = 1;
  }

  // reduce control hold up of instructions older than branch instruction once branch is issued
  void reduce_control_val(int index)
  {
    for (unsigned i = 0; i < IBUFFER_SIZE; i++) {	
      if (ibuffer_OOO[i].m_valid && ibuffer_OOO[i].m_index > index)	
      {	
        ibuffer_OOO[i].control_check_in_order--;
      }	
    }	
  }

  void reduce_predicate_val(int index)
  {
    for (unsigned i = 0; i < IBUFFER_SIZE; i++) {	
      if (ibuffer_OOO[i].m_valid && ibuffer_OOO[i].m_index > index)	
      {	
        ibuffer_OOO[i].m_DEB_bit--;
      }	
    }	
  }

  void set_active_mask(active_mask_t active_mask, int index)
  {
    ibuffer_OOO[index].m_active_mask = active_mask;
  }

  active_mask_t get_active_mask(int index)
  {
    return ibuffer_OOO[index].m_active_mask;
  }

  bool get_control_bit_prev(int index)
  {
    {
      for (unsigned i = 0; i < IBUFFER_SIZE_IN_ORDER; i++) {	
        if(ibuffer_OOO[i].m_index == index)
        {
          return  ibuffer_OOO[i].control_check;
        }
      }
    }
  }

  bool get_control_bit_prev_full_OOO(int index)
  {
    {
      int total_size = IBUFFER_SIZE_IN_ORDER + IBUFFER_SIZE_OOO_ORDER;
      for (unsigned i = 0; i < total_size; i++) {	
        if(ibuffer_OOO[i].m_index == index)
        {
          return  ibuffer_OOO[i].control_check;
        }
      }
    }
  }


  //void replay_buffer_step() { replay_m_next = (replay_m_next + 1) % REPLAY_BUFFER_SIZE; }	
  void ibuffer_step_OOO() { 	
    if(replay_m_next + 1 < IBUFFER_SIZE_IN_ORDER)	
      replay_m_next = replay_m_next + 1;	
    else  	
      replay_m_next = 0;	
  }	


  std::vector<const warp_inst_t *> get_ibuffer_Inst_OOO(int index)  {	
    std::vector<const warp_inst_t *> replayInst;	
    for (unsigned i = 0; i < IBUFFER_SIZE; i++) {	
      if (ibuffer_OOO[i].m_valid && ibuffer_OOO[i].m_index < index)	
      {	
        replayInst.push_back(ibuffer_OOO[i].m_inst);	
      }	
    }	
    return replayInst;	
  }	

  std::vector<int> get_ibuffer_index_OOO(int index)  {	
    std::vector<int>replayidx;	
    for (unsigned i = 0; i < IBUFFER_SIZE; i++) {	
      if (ibuffer_OOO[i].m_valid && ibuffer_OOO[i].m_index < index)	
      {	
        replayidx.push_back(i);	
      }	
    }	
    return replayidx;	
  }	

  // OOO IB REPLAY BUFFER
   // use same ibuffer fill function with new location and empty the old ibuffer entery using inst loc

   void ibuffer_step_streaming_IB_DEB_OOO() 
  { 
    if(ibuffer_m_next_streaming + 1 < IBUFFER_SIZE_IN_ORDER)	
      ibuffer_m_next_streaming = ibuffer_m_next_streaming + 1;	
    else  	
      ibuffer_m_next_streaming = 0; 
  }

  bool ibuffer_empty_OOO_DEB_IB() const {	
    int offset = IBUFFER_SIZE_IN_ORDER;
    for (unsigned i = 0; i < IBUFFER_SIZE_OOO_ORDER; i++)	
      if (ibuffer_OOO[i+offset].m_valid) return false;	
    return true;	
  }	

  int ibuffer_index_DEB_IB_OOO(int index, int warp)
  {
    int offset = IBUFFER_SIZE_IN_ORDER;
    for (unsigned i = 0; i < IBUFFER_SIZE_OOO_ORDER; i++)	
    {
      if (ibuffer_OOO[i+offset].m_index == index)
        return (i+offset);
    }  
  }

  std::vector<const warp_inst_t *> get_Replay_buffer_Inst_DEB_IB_OOO(int index)  {	
    std::vector<const warp_inst_t *> replayInst;	
    for (unsigned i = 0; i < IBUFFER_SIZE_OOO_ORDER; i++) {	
      if (ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_valid && (ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_index < index))	
      {
        replayInst.push_back(ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_inst);	
      }	
    }	
    return replayInst;	
  }	

  std::vector<const warp_inst_t *> get_Replay_buffer_Inst_IB_OOO(int index)  {	
    std::vector<const warp_inst_t *> replayInst;	
    for (unsigned i = 0; i < IBUFFER_SIZE; i++) {	
      if (ibuffer_OOO[i].m_valid && (ibuffer_OOO[i].m_index < index))	
      {
        replayInst.push_back(ibuffer_OOO[i].m_inst);	
      }	
    }	
    return replayInst;	
  }

  std::vector<const warp_inst_t *> get_Replay_buffer_Inst_IB_OOOMem(int index)  {	
    std::vector<const warp_inst_t *> replayInst;	
    for (unsigned i = 0; i < IBUFFER_SIZE; i++) {	
      if (ibuffer_OOO[i].m_valid && (ibuffer_OOO[i].m_index < index) && (ibuffer_OOO[i].m_inst->is_load()||ibuffer_OOO[i].m_inst->is_store()))	
      {
        replayInst.push_back(ibuffer_OOO[i].m_inst);	
      }	
    }	
    return replayInst;	
  }

  std::vector<const warp_inst_t *> get_Replay_buffer_Inst_IB_OOOComp(int index)  {	
    std::vector<const warp_inst_t *> replayInst;	
    for (unsigned i = 0; i < IBUFFER_SIZE; i++) {	
      if (ibuffer_OOO[i].m_valid && (ibuffer_OOO[i].m_index < index) && !(ibuffer_OOO[i].m_inst->is_load()||ibuffer_OOO[i].m_inst->is_store()))	
      {
        replayInst.push_back(ibuffer_OOO[i].m_inst);	
      }	
    }	
    return replayInst;	
  }

  std::vector<const warp_inst_t *> get_Replay_buffer_Inst_DEB_OOO(int index)  {	
    std::vector<const warp_inst_t *> replayInst;	
    for (unsigned i = 0; i < IBUFFER_SIZE_OOO_ORDER; i++) {	
      if (ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_valid && (ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_index < index))	
      {
        replayInst.push_back(ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_inst);	
      }	
    }	
    return replayInst;	
  }

  std::vector<int> get_Replay_buffer_index_DEB_OOO(int index)  {	
    std::vector<int> replayInst;	
    for (unsigned i = 0; i < IBUFFER_SIZE_OOO_ORDER; i++) {	
      if (ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_valid && (ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_index < index))	
      {
        int val = IBUFFER_SIZE_IN_ORDER+i;
        replayInst.push_back(val);	
      }	
    }	
    return replayInst;	
  }	

  // Ishita bug found
  std::vector<const warp_inst_t *> get_Replay_buffer_Inst_DEB_in_order(int index)  {	
    std::vector<const warp_inst_t *> replayInst;	
    for (unsigned i = 0; i < IBUFFER_SIZE_OOO_ORDER; i++) {	
      if (ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_valid)	
      {
        replayInst.push_back(ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_inst);	
      }	
    }	
    return replayInst;	
  }	

  std::vector<const warp_inst_t *> get_Replay_buffer_Inst_older(int index)  {	
    std::vector<const warp_inst_t *> replayInst;	
    for (unsigned i = 0; i < IBUFFER_SIZE; i++) {	
      if (ibuffer_OOO[i].m_valid && ibuffer_OOO[i].m_index> index)	
      {
        replayInst.push_back(ibuffer_OOO[i].m_inst);	

        // dont look for independent instruction after a serializing instruction
        if((ibuffer_OOO[i].m_inst->op == BRANCH_OP || ibuffer_OOO[i].m_inst->op == BARRIER_OP || ibuffer_OOO[i].m_inst->op == RET_OPS || ibuffer_OOO[i].m_inst->op == EXIT_OPS || ibuffer_OOO[i].m_inst->isatomic() || ibuffer_OOO[i].m_inst->op == SPECIALIZED_UNIT_1_OP ))
        {
          return replayInst;
        }
      }	
    }	
    return replayInst;	
  }	

  std::vector<int> get_Replay_buffer_loc_DEB_in_order(int index)  {	
    std::vector<int> replay_idx;	
    for (unsigned i = 0; i < IBUFFER_SIZE_OOO_ORDER; i++) {	
      if (ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_valid)	
      {
        int val = IBUFFER_SIZE_IN_ORDER+i;
        replay_idx.push_back(val);	
      }	
    }	
    return replay_idx;	
  }	

  std::vector<const warp_inst_t *> get_Replay_buffer_Inst_DEB_in_orderMem(int index)  {	
    std::vector<const warp_inst_t *> replayInst;	
    for (unsigned i = 0; i < IBUFFER_SIZE_OOO_ORDER; i++) {	
      if (ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_valid && (ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_inst->is_load()||ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_inst->is_store()))	
      {
        replayInst.push_back(ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_inst);	
      }	
    }	
    return replayInst;	
  }	

  std::vector<const warp_inst_t *> get_Replay_buffer_Inst_DEB_in_orderComp(int index)  {	
    std::vector<const warp_inst_t *> replayInst;	
    for (unsigned i = 0; i < IBUFFER_SIZE_OOO_ORDER; i++) {	
      if (ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_valid && !(ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_inst->is_load()||ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_inst->is_store()))	
      {
        replayInst.push_back(ibuffer_OOO[IBUFFER_SIZE_IN_ORDER+i].m_inst);	
      }	
    }	
    return replayInst;	
  }	

  void replay_buffer_increment_tail_IB_DEB_OOO()
  {
    ib_replay_buffer_tail_DEB_OOO++;
  }

  void replay_buffer_decrement_tail_IB_DEB_OOO()
  {
    ib_replay_buffer_tail_DEB_OOO--;
  }

  int replay_buffer_get_tail_IB_DEB_OOO()
  {
    return ib_replay_buffer_tail_DEB_OOO;
  }


  void ibuffer_increment_tail_OOO()
  {
    ibuffer_tail++;
  }

  void ibuffer_decrement_tail_OOO()
  {
    ibuffer_tail--;
  }

  int ibuffer_get_tail_OOO()
  {
    return ibuffer_tail;
  }

  void ibuffer_increment_tail_full_OOO()
  {
    ibuffer_tail_full_OOO++;
  }

  void ibuffer_increment_inst_number_OOO()
  {
    ibuffer_inst_number++;
  }


  void ibuffer_decrement_tail_full_OOO()
  {
    ibuffer_tail_full_OOO--;
  }

  int ibuffer_get_tail_full_OOO()
  {
    return ibuffer_tail_full_OOO;
  }

  long long get_inst_counter()
  {
    return inst_counter;
  }

  void increase_inst_counter()
  {
    inst_counter++;
  }

  long long get_instruction_counter(int idx)
  {
    return ibuffer_OOO[idx].m_inst_counter;
  }

  void set_instruction_counter_issued(int idx)
  {
    ibuffer_OOO[idx].m_inst_counter = -1;
  }

  void ibuffer_increment_control_inst()
  {
    ibuffer_control_inst++;
  }

  void ibuffer_increment_predicate_inst()
  {
    predicate_inst++;
  }

  void increaseLoadToTLB()
  {
    load_to_TLB++;
  }

  int getLoadToTLB()
  {
    return load_to_TLB;
  }

  void reduceInstToTLB()
  {
    load_to_TLB--;
  }

  int get_stalls_for_inst()
  {
    return stalls_for_inst;
  }

  void increase_stalls_for_inst()
  {
    stalls_for_inst++;
  }

  void set_stalls_for_inst_zero()
  {
    stalls_for_inst = 0;
  }

  void ibuffer_decrement_control_inst_OOO()
  {
    ibuffer_control_inst--;
  }

  void ibuffer_decrement_predicate_inst_OOO()
  {
    predicate_inst--;
  }

  int ibuffer_get_control_inst_OOO()
  {
    return ibuffer_control_inst;
  }

  int ibuffer_get_predicate_inst_OOO()
  {
    return predicate_inst;
  }


  void ibuffer_increase_store()
  {
    ibuffer_store_inst++;
  }

  void ibuffer_decrease_store()
  {
    ibuffer_store_inst--;
  }

  int ibuffer_get_store()
  {
    return ibuffer_store_inst;
  }

  void ibuffer_increase_memory_count()
  {
    ibuffer_memory_inst++;
  }

  void ibuffer_decrease_memory_count()
  {
    ibuffer_memory_inst--;
  }

  int ibuffer_get_memory_count()
  {
    return ibuffer_memory_inst;
  }

  int ibuffer_distance(int index)
  {
    return ibuffer_OOO[index].m_pushed_out;
  }

  // increase reordering_distance of instructions when something is issued before them
  int increase_inst_reordering_distance(int index, int warp_id)
  {
    for (unsigned i = 0; i < IBUFFER_SIZE; i++)
    {
      if(ibuffer_OOO[i].m_valid && (ibuffer_OOO[i].m_index < index))
      {
        ibuffer_OOO[i].m_pushed_out++;
      }
    }
  }


  void ibuffer_decrease_stores(int index)
  {
    for (unsigned i = 0; i < IBUFFER_SIZE; i++)
    {
      if(ibuffer_OOO[i].m_valid && (ibuffer_OOO[i].m_index > index))
        ibuffer_OOO[i].m_stores--;
    }
  }

  void ibuffer_decrease_stores_IB_IN(int index)
  {
    for (unsigned i = 0; i < IBUFFER_SIZE_IN_ORDER; i++)
    {
      if(ibuffer_OOO[i].m_valid && (ibuffer_OOO[i].m_index > index))
        ibuffer_OOO[i].m_stores--;
    }
  }

  void ibuffer_decrease_stores_IB_OOO(int index)
  {
    int offset = IBUFFER_SIZE_IN_ORDER;
    for (unsigned i = 0; i < IBUFFER_SIZE_OOO_ORDER; i++)
    {
      if(ibuffer_OOO[i+offset].m_valid && (ibuffer_OOO[i+offset].m_index > index))
        ibuffer_OOO[i+offset].m_stores--;
    }
  }

  void ibuffer_decrease_stores_IB_IN_FROM_OOO(int index)
  {
    for (unsigned i = 0; i < IBUFFER_SIZE_IN_ORDER; i++)
    {
      if(ibuffer_OOO[i].m_valid)
        ibuffer_OOO[i].m_stores--;
    }
  }

  void ibuffer_decrease_meminst(int index, int warp)
  {

    for (unsigned i = 0; i < IBUFFER_SIZE; i++)
    {
      if(ibuffer_OOO[i].m_valid && (ibuffer_OOO[i].m_index > index))
        ibuffer_OOO[i].m_meminst--;
    }
  }

  void ibuffer_decrease_meminst_IB_IN(int index, int warp)
  {

    for (unsigned i = 0; i < IBUFFER_SIZE_IN_ORDER; i++)
    {
      if(ibuffer_OOO[i].m_valid && (ibuffer_OOO[i].m_index > index))
        ibuffer_OOO[i].m_meminst--;
    }
  }

  void ibuffer_decrease_meminst_IB_IN_FROM_OOO(int index, int warp)
  {
    for (unsigned i = 0; i < IBUFFER_SIZE_IN_ORDER; i++)
    {
      if(ibuffer_OOO[i].m_valid)
        ibuffer_OOO[i].m_meminst--;
    }
  }

  void ibuffer_decrease_meminst_IB_OOO(int index, int warp)
  {
    int offset = IBUFFER_SIZE_IN_ORDER;
    for (unsigned i = 0; i < IBUFFER_SIZE_OOO_ORDER; i++)
    {
      if(ibuffer_OOO[i+offset].m_valid && (ibuffer_OOO[i+offset].m_index > index))
        ibuffer_OOO[i+offset].m_meminst--;
    }
  }

  void ibuffer_decrease_all_index_OOO(int index, int warp)
  {
    for (unsigned i = 0; i < IBUFFER_SIZE; i++)
    {
      if(ibuffer_OOO[i].m_valid && (ibuffer_OOO[i].m_index > index))
        ibuffer_OOO[i].m_index = ibuffer_OOO[i].m_index - 1;
    }
  }

  void ibuffer_decrease_all_index_free_on_oldest(int index, int warp)
  {
    // if(warp == 3)
    //   std::cout<<"red_index "<<index<<"\n";
    for (unsigned i = 0; i < IBUFFER_SIZE; i++)
    {
      // if(warp == 3)
      //   std::cout<<"old_val "<<ibuffer_OOO[i].m_index<<" ";
      if((ibuffer_OOO[i].m_index > index))
      {
        ibuffer_OOO[i].m_index = ibuffer_OOO[i].m_index - 1;
        // if(warp == 3)
        //   std::cout<<"changing_val "<<ibuffer_OOO[i].m_index<<"\n";
      }
      else
      {
        // if(warp == 3)
        //   std::cout<<"old_val "<<ibuffer_OOO[i].m_index<<"\n";
      }
    }
  }

  void ibuffer_decrease_all_index_IB_IN_OOO(int index, int warp)
  {
    for (unsigned i = 0; i < IBUFFER_SIZE_IN_ORDER; i++)
    {
      if(ibuffer_OOO[i].m_valid && (ibuffer_OOO[i].m_index > index))
        ibuffer_OOO[i].m_index = ibuffer_OOO[i].m_index - 1;
    }
  }

  void ibuffer_decrease_all_index_IB_DEB_OOO(int index, int warp)
  {
    int offset = IBUFFER_SIZE_IN_ORDER;
    for (unsigned i = 0; i < IBUFFER_SIZE_OOO_ORDER; i++)
    {
      if(ibuffer_OOO[i+offset].m_valid && (ibuffer_OOO[i+offset].m_index > index))
        ibuffer_OOO[i+offset].m_index = ibuffer_OOO[i+offset].m_index - 1;
    }
  }

  int get_ibuffer_size()
  {
    return IBUFFER_SIZE;
  }

  int get_Sched_ibuffer_size()
  {
    return SCHED_IBUFFER_SIZE;
  }

  // end of OOO Instruction BUFFER FUNCTIONS	

  //ibuffer changed CORRECT

  void ibuffer_fill_streaming(unsigned slot, const warp_inst_t *pI) {
    assert(slot < IBUFFER_SIZE);
    m_ibuffer[slot].m_inst = pI;
    m_ibuffer[slot].m_valid = true;
    if(ibuffer_m_fill_next_streaming + 1 < IBUFFER_SIZE)	
      ibuffer_m_fill_next_streaming = ibuffer_m_fill_next_streaming + 1;	
    else	
      ibuffer_m_fill_next_streaming = 0;
  }

  void set_ibuffer_pushed_streaming(int cycle,unsigned slot)
  {
    m_ibuffer[slot].m_pushed_in = cycle;
  }

  int ibuffer_empty_idx_streaming() {	
    // for (unsigned i = 0; i < REPLAY_BUFFER_SIZE; i++)	
    //   if (!replay_buffer[i].m_valid)	
    //     return i;	
    if(!m_ibuffer[ibuffer_m_fill_next_streaming].m_valid)	
      return ibuffer_m_fill_next_streaming;	
    return -1;	
  }	

  bool ibuffer_empty_streaming() const {
    for (unsigned i = 0; i < IBUFFER_SIZE; i++)
      if (m_ibuffer[i].m_valid) return false;
    return true;
  }


  bool ibuffer_has_space_stareaming() const {
    for (unsigned i = 0; i < IBUFFER_SIZE; i++)	
      if (!m_ibuffer[i].m_valid) 	
        return true;
    return false;	
  }

  void ibuffer_flush_streaming() {	
    for (unsigned i = 0; i < IBUFFER_SIZE; i++) {	
      if (m_ibuffer[i].m_valid) dec_inst_in_pipeline();	
      m_ibuffer[i].m_inst = NULL;	
      m_ibuffer[i].m_valid = false;		
    }
    ibuffer_m_fill_next_streaming = 0;
    ibuffer_m_next_streaming = 0;
  }	


  // end of streaming IBUFFER functions correct

  void set_DEB_dep_bit(bool val) { DEB_dep = val;}
  bool get_DEB_dep_bit() { return DEB_dep; }


  bool imiss_pending() const { return m_imiss_pending; }
  void set_imiss_pending() { m_imiss_pending = true; }
  void clear_imiss_pending() { m_imiss_pending = false; }

  void inc_inst_count() { num_inst_exec_warp++;  }
  int inst_count_val() {return num_inst_exec_warp;}

  bool stores_done() const { return m_stores_outstanding == 0; }
  void inc_store_req() { m_stores_outstanding++; }
  void dec_store_req() {
    assert(m_stores_outstanding > 0);
    m_stores_outstanding--;
  }

  unsigned num_inst_in_buffer() const {
    unsigned count = 0;
    for (unsigned i = 0; i < IBUFFER_SIZE; i++) {
      if (m_ibuffer[i].m_valid) count++;
    }
    return count;
  }
  unsigned num_inst_in_pipeline() const { return m_inst_in_pipeline; }
  unsigned num_issued_inst_in_pipeline() const {
    return (num_inst_in_pipeline() - num_inst_in_buffer());
  }
  bool inst_in_pipeline() const { return m_inst_in_pipeline > 0; }
  void inc_inst_in_pipeline() { m_inst_in_pipeline++; }
  int inst_in_pipeline_count() const { return m_inst_in_pipeline; }
  void dec_inst_in_pipeline() {
    assert(m_inst_in_pipeline > 0);
    m_inst_in_pipeline--;
  }

  unsigned get_cta_id() const { return m_cta_id; }

  unsigned get_dynamic_warp_id() const { return m_dynamic_warp_id; }
  unsigned get_warp_id() const { return m_warp_id; }

  class shader_core_ctx *get_shader() {
    return m_shader;
  }

  struct ibuffer_entry {
    ibuffer_entry() {
      m_valid = false;
      m_inst = NULL;
      m_pushed_in = 0;
      m_pushed_out = 0;
    }
    const warp_inst_t *m_inst;
    bool m_valid;
    int m_pushed_in;
    int m_pushed_out;
  };

  struct replay_buffer_entry {	
    replay_buffer_entry() {	
      m_valid = false;	
      m_inst = NULL;	
      m_pc = NULL;	
      m_pushed_in = 0;
      m_pushed_out = 0;
    }	
    const warp_inst_t *m_inst;	
    bool m_valid;	
    int m_pc;	
    int m_pushed_in;
    int m_pushed_out;
  };	

  struct replay_buffer_entry_OOO {	
    replay_buffer_entry_OOO() {	
      m_valid = false;	
      m_inst = NULL;	
      m_pc = NULL;
      m_index = -1;	
      m_stores = 0;
      m_meminst = 0;
      m_active_mask;
      m_pushed_in = 0;
      m_pushed_out = 0;
    }	
    const warp_inst_t *m_inst;	
    bool m_valid;	
    int m_pc;	
    int m_index;
    int m_stores;
    int m_meminst;
    active_mask_t m_active_mask;
    int m_pushed_in;
    int m_pushed_out;
  };	

  struct ibuffer_entry_OOO {	
    ibuffer_entry_OOO() {	
      m_valid = false;	
      m_inst = NULL;	
      m_pc = NULL;
      m_index = -1;	
      m_index_full_OOO = false;	
      m_stores = 0;
      m_meminst = 0;
      control_check = 0;
      m_control_inst_OOO = false;
      m_active_mask;
      m_pushed_in_OOO = 0;
      m_pushed_out = 0;
      m_inst_number = 0;
      m_DEB_bit = 0;
      m_inst_counter = -1;
      control_check_in_order = 0;
      m_stall_cycles = -1;
      mem_dep = -1;
      isIdempotent = 0;
      priorLoads = 0;
    }	
    const warp_inst_t *m_inst;	
    bool m_valid;	
    int m_pc;	
    int m_index;
    int m_index_full_OOO;
    int m_stores;
    int m_meminst;
    int control_check;
    int control_check_in_order;
    int m_control_inst_OOO;
    int mem_dep;
    active_mask_t m_active_mask;
    int m_pushed_in_OOO;
    int m_pushed_out;
    int m_DEB_bit;
    long long m_inst_counter;
    int m_inst_number;
    int m_stall_cycles;
    bool isIdempotent;
    int priorLoads;
  };	

  struct replay_buffer_entry_mem {	
    replay_buffer_entry_mem() {	
      m_valid = false;	
      m_inst = NULL;	
      m_pc = NULL;	
      m_pushed_in = 0;
      m_pushed_out = 0;
    }	
    const warp_inst_t *m_inst;	
    bool m_valid;	
    int m_pc;	
    int m_pushed_in;
    int m_pushed_out;
  };

  unsigned last_pc_decoded_streaming;
  unsigned last_pc_decoded_streaming_test;
  unsigned stalls_between_issues;
  unsigned num_inst_OOO;
  int stalled_on_replay;
  //int warp_stuck_replay;

  unsigned IBUFFER_SIZE;
  unsigned SCHED_IBUFFER_SIZE;
  unsigned IBUFFER_SIZE_IN_ORDER;
  unsigned IBUFFER_SIZE_OOO_ORDER;

  vector<replay_buffer_entry>replay_buffer;
  //vector<replay_buffer_entry_mem>replay_buffer_mem;
  vector<replay_buffer_entry_OOO>replay_buffer_OOO;
  vector<replay_buffer_entry>ibuffer_OOO_1;


  vector<ibuffer_entry>m_ibuffer;
  vector<ibuffer_entry>m_ibuffer_streaming;
  vector<ibuffer_entry_OOO>ibuffer_OOO;

 private:
  static const unsigned REPLAY_BUFFER_SIZE = 2  ;	// also change DEB_BUFFER_SIZE size in main.cc
  static const unsigned REPLAY_BUFFER_SIZE_MEM = 1;
  class shader_core_ctx *m_shader;
  unsigned m_cta_id;
  
  unsigned m_warp_id;
  unsigned m_warp_size;
  unsigned m_dynamic_warp_id;

  address_type m_next_pc;
  unsigned n_completed;  // number of threads in warp completed
  std::bitset<MAX_WARP_SIZE> m_active_threads;

  bool m_imiss_pending;
  bool DEB_dep;
  int num_inst_exec_warp;

  warp_inst_t m_inst_at_barrier;
  unsigned m_next;	
  unsigned replay_m_next;	
  unsigned replay_m_fill_next;
  unsigned ibuffer_m_fill_next_streaming;	
  unsigned replay_m_next_mem;	
  unsigned ibuffer_m_next_streaming;
  unsigned replay_m_fill_next_mem;
  unsigned m_n_atomic;  // number of outstanding atomic operations
  bool m_membar;        // if true, warp is waiting at memory barrier
  unsigned replay_buffer_head;
  unsigned replay_buffer_tail;
  unsigned ib_replay_buffer_tail_DEB_OOO;
  unsigned replay_buffer_store_inst;
  unsigned replay_buffer_memory_inst;
  int load_to_TLB;
  unsigned ibuffer_tail;
  unsigned ibuffer_tail_full_OOO;
  unsigned ibuffer_store_inst;
  unsigned ibuffer_memory_inst;
  unsigned ibuffer_control_inst;
  long long inst_counter;
  unsigned predicate_inst;
  unsigned ibuffer_inst_number;

  int stalls_for_inst;

  bool m_done_exit;  // true once thread exit has been registered for threads in
                     // this warp

  unsigned long long m_last_fetch;

  unsigned m_stores_outstanding;  // number of store requests sent but not yet
                                  // acknowledged
  unsigned m_inst_in_pipeline;
  public:
  int stall_points;
  int stall_point_PC;
  unsigned int m_cdp_latency;
  bool m_cdp_dummy;
  long long num_inst_exec;
  unsigned num_warp_inst;
  bool warp_inst_retired;
  bool stop_fetching_when_inst_0;
  bool warp_wb;
  bool ib_enter = 0;
};

inline unsigned hw_tid_from_wid(unsigned wid, unsigned warp_size, unsigned i) {
  return wid * warp_size + i;
};
inline unsigned wid_from_hw_tid(unsigned tid, unsigned warp_size) {
  return tid / warp_size;
};

const unsigned WARP_PER_CTA_MAX = 128;
typedef std::bitset<WARP_PER_CTA_MAX> warp_set_t;

int register_bank(int regnum, int wid, unsigned num_banks,
                  unsigned bank_warp_shift, bool sub_core_model,
                  unsigned banks_per_sched, unsigned sched_id);

class shader_core_ctx;
class shader_core_config;
class shader_core_stats;

enum scheduler_prioritization_type {
  SCHEDULER_PRIORITIZATION_LRR = 0,   // Loose Round Robin
  SCHEDULER_PRIORITIZATION_SRR,       // Strict Round Robin
  SCHEDULER_PRIORITIZATION_GTO,       // Greedy Then Oldest
  SCHEDULER_PRIORITIZATION_GTLRR,     // Greedy Then Loose Round Robin
  SCHEDULER_PRIORITIZATION_GTY,       // Greedy Then Youngest
  SCHEDULER_PRIORITIZATION_OLDEST,    // Oldest First
  SCHEDULER_PRIORITIZATION_YOUNGEST,  // Youngest First
};

// Each of these corresponds to a string value in the gpgpsim.config file
// For example - to specify the LRR scheudler the config must contain lrr
enum concrete_scheduler {
  CONCRETE_SCHEDULER_LRR = 0,
  CONCRETE_SCHEDULER_GTO,
  CONCRETE_SCHEDULER_TWO_LEVEL_ACTIVE,
  CONCRETE_SCHEDULER_RRR,
  CONCRETE_SCHEDULER_WARP_LIMITING,
  CONCRETE_SCHEDULER_OLDEST_FIRST,
  NUM_CONCRETE_SCHEDULERS
};

class scheduler_unit {  // this can be copied freely, so can be used in std
                        // containers.
 public:
  scheduler_unit(shader_core_stats *stats, shader_core_ctx *shader,
                 Scoreboard *scoreboard, simt_stack **simt,
                 std::vector<shd_warp_t *> *warp, register_set *sp_out,
                 register_set *dp_out, register_set *sfu_out,
                 register_set *int_out, register_set *tensor_core_out,
                 std::vector<register_set *> &spec_cores_out,
                 register_set *mem_out, int id)
      : m_supervised_warps(),
        m_stats(stats),
        m_shader(shader),
        m_scoreboard(scoreboard),
        m_simt_stack(simt),
        /*m_pipeline_reg(pipe_regs),*/ m_warp(warp),
        m_sp_out(sp_out),
        m_dp_out(dp_out),
        m_sfu_out(sfu_out),
        m_int_out(int_out),
        m_tensor_core_out(tensor_core_out),
        m_spec_cores_out(spec_cores_out),
        m_mem_out(mem_out),
        m_id(id) {}
  virtual ~scheduler_unit() {}
  virtual void add_supervised_warp_id(int i) {
    m_supervised_warps.push_back(&warp(i));
  }
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.end();
  }

  // The core scheduler cycle method is meant to be common between
  // all the derived schedulers.  The scheduler's behaviour can be
  // modified by changing the contents of the m_next_cycle_prioritized_warps
  // list.
  void cycle();
  void cycle_ibuffer_OOO(int m_cluster_id, int sched_num);	
  int replay_ib_inst(int m_cluster_id, int sched_num, int &warp_id_issued, std::vector<int>&warp_problem, std::vector<int>&loc_issue, std::vector<vector<int>>&loc_dep_on);	
  bool canPushInst(const warp_inst_t *pI);

  // These are some common ordering fucntions that the
  // higher order schedulers can take advantage of
  template <typename T>
  void order_lrr(
      typename std::vector<T> &result_list,
      const typename std::vector<T> &input_list,
      const typename std::vector<T>::const_iterator &last_issued_from_input,
      unsigned num_warps_to_add);
  template <typename T>
  void order_rrr(
      typename std::vector<T> &result_list,
      const typename std::vector<T> &input_list,
      const typename std::vector<T>::const_iterator &last_issued_from_input,
      unsigned num_warps_to_add);

  enum OrderingType {
    // The item that issued last is prioritized first then the sorted result
    // of the priority_function
    ORDERING_GREEDY_THEN_PRIORITY_FUNC = 0,
    // No greedy scheduling based on last to issue. Only the priority function
    // determines priority
    ORDERED_PRIORITY_FUNC_ONLY,
    NUM_ORDERING,
  };
  template <typename U>
  void order_by_priority(
      std::vector<U> &result_list, const typename std::vector<U> &input_list,
      const typename std::vector<U>::const_iterator &last_issued_from_input,
      unsigned num_warps_to_add, OrderingType age_ordering,
      bool (*priority_func)(U lhs, U rhs));
  static bool sort_warps_by_oldest_dynamic_id(shd_warp_t *lhs, shd_warp_t *rhs);

  // Derived classes can override this function to populate
  // m_supervised_warps with their scheduling policies
  virtual void order_warps() = 0;

  int get_schd_id() const { return m_id; }

 protected:
  virtual void do_on_warp_issued(
      unsigned warp_id, unsigned num_issued,
      const std::vector<shd_warp_t *>::const_iterator &prioritized_iter);
  
  virtual void replay_do_on_warp_issued(	
      unsigned warp_id, unsigned num_issued,	
      const std::vector<shd_warp_t *>::const_iterator &prioritized_iter, int MEM_ON);	
  inline int get_sid() const;

 protected:
  shd_warp_t &warp(int i);

  // This is the prioritized warp list that is looped over each cycle to
  // determine which warp gets to issue.
  std::vector<shd_warp_t *> m_next_cycle_prioritized_warps;
  // The m_supervised_warps list is all the warps this scheduler is supposed to
  // arbitrate between.  This is useful in systems where there is more than
  // one warp scheduler. In a single scheduler system, this is simply all
  // the warps assigned to this core.
  std::vector<shd_warp_t *> m_supervised_warps;
  // This is the iterator pointer to the last supervised warp you issued
  std::vector<shd_warp_t *>::const_iterator m_last_supervised_issued;
  shader_core_stats *m_stats;
  shader_core_ctx *m_shader;
  // these things should become accessors: but would need a bigger rearchitect
  // of how shader_core_ctx interacts with its parts.
  Scoreboard *m_scoreboard;
  simt_stack **m_simt_stack;
  // warp_inst_t** m_pipeline_reg;
  std::vector<shd_warp_t *> *m_warp;
  register_set *m_sp_out;
  register_set *m_dp_out;
  register_set *m_sfu_out;
  register_set *m_int_out;
  register_set *m_tensor_core_out;
  register_set *m_mem_out;
  std::vector<register_set *> &m_spec_cores_out;
  unsigned m_num_issued_last_cycle;
  unsigned m_current_turn_warp;

  int m_id;
};

class lrr_scheduler : public scheduler_unit {
 public:
  lrr_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
                Scoreboard *scoreboard, simt_stack **simt,
                std::vector<shd_warp_t *> *warp, register_set *sp_out,
                register_set *dp_out, register_set *sfu_out,
                register_set *int_out, register_set *tensor_core_out,
                std::vector<register_set *> &spec_cores_out,
                register_set *mem_out, int id)
      : scheduler_unit(stats, shader, scoreboard, simt, warp, sp_out, dp_out,
                       sfu_out, int_out, tensor_core_out, spec_cores_out,
                       mem_out, id) {}
  virtual ~lrr_scheduler() {}
  virtual void order_warps();
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.end();
  }
};

class rrr_scheduler : public scheduler_unit {
 public:
  rrr_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
                Scoreboard *scoreboard, simt_stack **simt,
                std::vector<shd_warp_t *> *warp, register_set *sp_out,
                register_set *dp_out, register_set *sfu_out,
                register_set *int_out, register_set *tensor_core_out,
                std::vector<register_set *> &spec_cores_out,
                register_set *mem_out, int id)
      : scheduler_unit(stats, shader, scoreboard, simt, warp, sp_out, dp_out,
                       sfu_out, int_out, tensor_core_out, spec_cores_out,
                       mem_out, id) {}
  virtual ~rrr_scheduler() {}
  virtual void order_warps();
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.end();
  }
};

class gto_scheduler : public scheduler_unit {
 public:
  gto_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
                Scoreboard *scoreboard, simt_stack **simt,
                std::vector<shd_warp_t *> *warp, register_set *sp_out,
                register_set *dp_out, register_set *sfu_out,
                register_set *int_out, register_set *tensor_core_out,
                std::vector<register_set *> &spec_cores_out,
                register_set *mem_out, int id)
      : scheduler_unit(stats, shader, scoreboard, simt, warp, sp_out, dp_out,
                       sfu_out, int_out, tensor_core_out, spec_cores_out,
                       mem_out, id) {}
  virtual ~gto_scheduler() {}
  virtual void order_warps();
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.begin();
  }
};

class oldest_scheduler : public scheduler_unit {
 public:
  oldest_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
                   Scoreboard *scoreboard, simt_stack **simt,
                   std::vector<shd_warp_t *> *warp, register_set *sp_out,
                   register_set *dp_out, register_set *sfu_out,
                   register_set *int_out, register_set *tensor_core_out,
                   std::vector<register_set *> &spec_cores_out,
                   register_set *mem_out, int id)
      : scheduler_unit(stats, shader, scoreboard, simt, warp, sp_out, dp_out,
                       sfu_out, int_out, tensor_core_out, spec_cores_out,
                       mem_out, id) {}
  virtual ~oldest_scheduler() {}
  virtual void order_warps();
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.begin();
  }
};

class two_level_active_scheduler : public scheduler_unit {
 public:
  two_level_active_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
                             Scoreboard *scoreboard, simt_stack **simt,
                             std::vector<shd_warp_t *> *warp,
                             register_set *sp_out, register_set *dp_out,
                             register_set *sfu_out, register_set *int_out,
                             register_set *tensor_core_out,
                             std::vector<register_set *> &spec_cores_out,
                             register_set *mem_out, int id, char *config_str)
      : scheduler_unit(stats, shader, scoreboard, simt, warp, sp_out, dp_out,
                       sfu_out, int_out, tensor_core_out, spec_cores_out,
                       mem_out, id),
        m_pending_warps() {
    unsigned inner_level_readin;
    unsigned outer_level_readin;
    int ret =
        sscanf(config_str, "two_level_active:%d:%d:%d", &m_max_active_warps,
               &inner_level_readin, &outer_level_readin);
    assert(3 == ret);
    m_inner_level_prioritization =
        (scheduler_prioritization_type)inner_level_readin;
    m_outer_level_prioritization =
        (scheduler_prioritization_type)outer_level_readin;
  }
  virtual ~two_level_active_scheduler() {}
  virtual void order_warps();
  void add_supervised_warp_id(int i) {
    if (m_next_cycle_prioritized_warps.size() < m_max_active_warps) {
      m_next_cycle_prioritized_warps.push_back(&warp(i));
    } else {
      m_pending_warps.push_back(&warp(i));
    }
  }
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.begin();
  }

 protected:
  virtual void do_on_warp_issued(
      unsigned warp_id, unsigned num_issued,
      const std::vector<shd_warp_t *>::const_iterator &prioritized_iter);

 private:
  std::deque<shd_warp_t *> m_pending_warps;
  scheduler_prioritization_type m_inner_level_prioritization;
  scheduler_prioritization_type m_outer_level_prioritization;
  unsigned m_max_active_warps;
};

// Static Warp Limiting Scheduler
class swl_scheduler : public scheduler_unit {
 public:
  swl_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
                Scoreboard *scoreboard, simt_stack **simt,
                std::vector<shd_warp_t *> *warp, register_set *sp_out,
                register_set *dp_out, register_set *sfu_out,
                register_set *int_out, register_set *tensor_core_out,
                std::vector<register_set *> &spec_cores_out,
                register_set *mem_out, int id, char *config_string);
  virtual ~swl_scheduler() {}
  virtual void order_warps();
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.begin();
  }

 protected:
  scheduler_prioritization_type m_prioritization;
  unsigned m_num_warps_to_limit;
};

class opndcoll_rfu_t {  // operand collector based register file unit
 public:
  // constructors
  opndcoll_rfu_t() {
    m_num_banks = 0;
    m_shader = NULL;
    m_initialized = false;
  }
  void add_cu_set(unsigned cu_set, unsigned num_cu, unsigned num_dispatch);
  typedef std::vector<register_set *> port_vector_t;
  typedef std::vector<unsigned int> uint_vector_t;
  void add_port(port_vector_t &input, port_vector_t &ouput,
                uint_vector_t cu_sets);
  void init(unsigned num_banks, shader_core_ctx *shader);

  // modifiers
  bool writeback(warp_inst_t &warp);

  void step() {
    dispatch_ready_cu();
    allocate_reads();
    for (unsigned p = 0; p < m_in_ports.size(); p++) allocate_cu(p);
    process_banks();
    //dump(stdout);
  }

  void dump(FILE *fp) const {
    fprintf(fp, "\n");
    fprintf(fp, "Operand Collector State %d:\n",cycles_passed);
    for (unsigned n = 0; n < m_cu.size(); n++) {
      fprintf(fp, "   CU-%2u: ", n);
      m_cu[n]->dump(fp, m_shader);
    }
    //m_arbiter.dump(fp);
  }

  shader_core_ctx *shader_core() { return m_shader; }

 private:
  void process_banks() { m_arbiter.reset_alloction(); }

  void dispatch_ready_cu();
  void allocate_cu(unsigned port);
  void allocate_reads();

  // types

  class collector_unit_t;

  class op_t {
   public:
    op_t() { m_valid = false; }
    op_t(collector_unit_t *cu, unsigned op, unsigned reg, unsigned num_banks,
         unsigned bank_warp_shift, bool sub_core_model,
         unsigned banks_per_sched, unsigned sched_id) {
      m_valid = true;
      m_warp = NULL;
      m_cu = cu;
      m_operand = op;
      m_register = reg;
      m_shced_id = sched_id;
      m_bank = register_bank(reg, cu->get_warp_id(), num_banks, bank_warp_shift,
                             sub_core_model, banks_per_sched, sched_id);
    }
    op_t(const warp_inst_t *warp, unsigned reg, unsigned num_banks,
         unsigned bank_warp_shift, bool sub_core_model,
         unsigned banks_per_sched, unsigned sched_id) {
      m_valid = true;
      m_warp = warp;
      m_register = reg;
      m_cu = NULL;
      m_operand = -1;
      m_shced_id = sched_id;
      m_bank = register_bank(reg, warp->warp_id(), num_banks, bank_warp_shift,
                             sub_core_model, banks_per_sched, sched_id);
    }

    // accessors
    bool valid() const { return m_valid; }
    unsigned get_reg() const {
      assert(m_valid);
      return m_register;
    }
    unsigned get_wid() const {
      if (m_warp)
        return m_warp->warp_id();
      else if (m_cu)
        return m_cu->get_warp_id();
      else
        abort();
    }
    unsigned get_sid() const { return m_shced_id; }
    unsigned get_active_count() const {
      if (m_warp)
        return m_warp->active_count();
      else if (m_cu)
        return m_cu->get_active_count();
      else
        abort();
    }
    const active_mask_t &get_active_mask() {
      if (m_warp)
        return m_warp->get_active_mask();
      else if (m_cu)
        return m_cu->get_active_mask();
      else
        abort();
    }
    unsigned get_sp_op() const {
      if (m_warp)
        return m_warp->sp_op;
      else if (m_cu)
        return m_cu->get_sp_op();
      else
        abort();
    }
    unsigned get_oc_id() const { return m_cu->get_id(); }
    unsigned get_bank() const { return m_bank; }
    unsigned get_operand() const { return m_operand; }
    void dump(FILE *fp) const {
      if (m_cu)
        fprintf(fp, " <R%u, CU:%u, w:%02u, pc:%d> ", m_register, m_cu->get_id(),
                m_cu->get_warp_id(), m_cu->get_pc());
      else if (!m_warp->empty())
        fprintf(fp, " <R%u, wid:%02u, pc:%d> ", m_register, m_warp->warp_id(), m_warp->pc);
    }
    std::string get_reg_string() const {
      char buffer[64];
      snprintf(buffer, 64, "R%u", m_register);
      return std::string(buffer);
    }

    // modifiers
    void reset() { m_valid = false; }

   private:
    bool m_valid;
    collector_unit_t *m_cu;
    const warp_inst_t *m_warp;
    unsigned m_operand;  // operand offset in instruction. e.g., add r1,r2,r3;
                         // r2 is oprd 0, r3 is 1 (r1 is dst)
    unsigned m_register;
    unsigned m_bank;
    unsigned m_shced_id;  // scheduler id that has issued this inst
  };

  enum alloc_t {
    NO_ALLOC,
    READ_ALLOC,
    WRITE_ALLOC,
  };

  class allocation_t {
   public:
    allocation_t() { m_allocation = NO_ALLOC; }
    bool is_read() const { return m_allocation == READ_ALLOC; }
    bool is_write() const { return m_allocation == WRITE_ALLOC; }
    bool is_free() const { return m_allocation == NO_ALLOC; }
    void dump(FILE *fp) const {
      if (m_allocation == NO_ALLOC) {
        fprintf(fp, "<free>");
      } else if (m_allocation == READ_ALLOC) {
        fprintf(fp, "rd: ");
        m_op.dump(fp);
      } else if (m_allocation == WRITE_ALLOC) {
        fprintf(fp, "wr: ");
        m_op.dump(fp);
      }
      fprintf(fp, "\n");
    }
    void alloc_read(const op_t &op) {
      assert(is_free());
      m_allocation = READ_ALLOC;
      m_op = op;
    }
    void alloc_write(const op_t &op) {
      assert(is_free());
      m_allocation = WRITE_ALLOC;
      m_op = op;
    }
    void reset() { m_allocation = NO_ALLOC; }

   private:
    enum alloc_t m_allocation;
    op_t m_op;
  };

  class arbiter_t {
   public:
    // constructors
    arbiter_t() {
      m_queue = NULL;
      m_allocated_bank = NULL;
      m_allocator_rr_head = NULL;
      _inmatch = NULL;
      _outmatch = NULL;
      _request = NULL;
      m_last_cu = 0;
    }
    void init(unsigned num_cu, unsigned num_banks) {
      assert(num_cu > 0);
      assert(num_banks > 0);
      m_num_collectors = num_cu;
      m_num_banks = num_banks;
      _inmatch = new int[m_num_banks];
      _outmatch = new int[m_num_collectors];
      _request = new int *[m_num_banks];
      for (unsigned i = 0; i < m_num_banks; i++)
        _request[i] = new int[m_num_collectors];
      m_queue = new std::list<op_t>[num_banks];
      m_allocated_bank = new allocation_t[num_banks];
      m_allocator_rr_head = new unsigned[num_cu];
      for (unsigned n = 0; n < num_cu; n++)
        m_allocator_rr_head[n] = n % num_banks;
      reset_alloction();
    }

    // accessors
    void dump(FILE *fp) const {
      fprintf(fp, "\n");
      fprintf(fp, "  Arbiter State:\n");
      fprintf(fp, "  requests:\n");
      for (unsigned b = 0; b < m_num_banks; b++) {
        fprintf(fp, "    bank %u : ", b);
        std::list<op_t>::const_iterator o = m_queue[b].begin();
        for (; o != m_queue[b].end(); o++) {
          o->dump(fp);
        }
        fprintf(fp, "\n");
      }
      fprintf(fp, "  grants:\n");
      for (unsigned b = 0; b < m_num_banks; b++) {
        fprintf(fp, "    bank %u : ", b);
        m_allocated_bank[b].dump(fp);
      }
      fprintf(fp, "\n");
    }

    // modifiers
    std::list<op_t> allocate_reads();

    void add_read_requests(collector_unit_t *cu) {
      const op_t *src = cu->get_operands();
      for (unsigned i = 0; i < MAX_REG_OPERANDS * 2; i++) {
        const op_t &op = src[i];
        if (op.valid()) {
          unsigned bank = op.get_bank();
          m_queue[bank].push_back(op);
        }
      }
    }
    bool bank_idle(unsigned bank) const {
      return m_allocated_bank[bank].is_free();
    }
    void allocate_bank_for_write(unsigned bank, const op_t &op) {
      assert(bank < m_num_banks);
      m_allocated_bank[bank].alloc_write(op);
    }
    void allocate_for_read(unsigned bank, const op_t &op) {
      assert(bank < m_num_banks);
      m_allocated_bank[bank].alloc_read(op);
    }
    void reset_alloction() {
      for (unsigned b = 0; b < m_num_banks; b++) m_allocated_bank[b].reset();
    }

   private:
    unsigned m_num_banks;
    unsigned m_num_collectors;

    allocation_t *m_allocated_bank;  // bank # -> register that wins
    std::list<op_t> *m_queue;

    unsigned *
        m_allocator_rr_head;  // cu # -> next bank to check for request (rr-arb)
    unsigned m_last_cu;       // first cu to check while arb-ing banks (rr)

    int *_inmatch;
    int *_outmatch;
    int **_request;
  };

  class input_port_t {
   public:
    input_port_t(port_vector_t &input, port_vector_t &output,
                 uint_vector_t cu_sets)
        : m_in(input), m_out(output), m_cu_sets(cu_sets) {
      assert(input.size() == output.size());
      assert(not m_cu_sets.empty());
    }
    // private:
    port_vector_t m_in, m_out;
    uint_vector_t m_cu_sets;
  };

  class collector_unit_t {
   public:
    // constructors
    collector_unit_t() {
      m_free = true;
      m_warp = NULL;
      m_output_register = NULL;
      m_src_op = new op_t[MAX_REG_OPERANDS * 2];
      m_not_ready.reset();
      m_warp_id = -1;
      m_num_banks = 0;
      m_bank_warp_shift = 0;
    }
    // accessors
    bool ready() const;
    const op_t *get_operands() const { return m_src_op; }
    void dump(FILE *fp, const shader_core_ctx *shader) const;

    unsigned get_warp_id() const { return m_warp_id; }
    unsigned get_active_count() const { return m_warp->active_count(); }
    const active_mask_t &get_active_mask() const {
      return m_warp->get_active_mask();
    }
    unsigned get_sp_op() const { return m_warp->sp_op; }
    unsigned get_id() const { return m_cuid; }  // returns CU hw id
    unsigned get_reg_id() const { return m_reg_id; }
    int get_pc() const { return m_warp->pc; }

    // modifiers
    void init(unsigned n, unsigned num_banks, unsigned log2_warp_size,
              const core_config *config, opndcoll_rfu_t *rfu,
              bool m_sub_core_model, unsigned reg_id,
              unsigned num_banks_per_sched);
    bool allocate(register_set *pipeline_reg, register_set *output_reg);

    void collect_operand(unsigned op) { m_not_ready.reset(op); }
    unsigned get_num_operands() const { return m_warp->get_num_operands(); }
    unsigned get_num_regs() const { return m_warp->get_num_regs(); }
    void dispatch();
    bool is_free() { return m_free; }

   private:
    bool m_free;
    unsigned m_cuid;  // collector unit hw id
    unsigned m_warp_id;
    warp_inst_t *m_warp;
    register_set
        *m_output_register;  // pipeline register to issue to when ready
    op_t *m_src_op;
    std::bitset<MAX_REG_OPERANDS * 2> m_not_ready;
    unsigned m_num_banks;
    unsigned m_bank_warp_shift;
    opndcoll_rfu_t *m_rfu;

    unsigned m_num_banks_per_sched;
    bool m_sub_core_model;
    unsigned m_reg_id;  // if sub_core_model enabled, limit regs this cu can r/w
  };

  class dispatch_unit_t {
   public:
    dispatch_unit_t(std::vector<collector_unit_t> *cus) {
      m_last_cu = 0;
      m_collector_units = cus;
      m_num_collectors = (*cus).size();
      m_next_cu = 0;
    }
    void init(bool sub_core_model, unsigned num_warp_scheds) {
      m_sub_core_model = sub_core_model;
      m_num_warp_scheds = num_warp_scheds;
    }

    collector_unit_t *find_ready() {
      // With sub-core enabled round robin starts with the next cu assigned to a
      // different sub-core than the one that dispatched last
      unsigned cusPerSched = m_num_collectors / m_num_warp_scheds;
      unsigned rr_increment = m_sub_core_model ?
                              cusPerSched - (m_last_cu % cusPerSched) : 1;
      for (unsigned n = 0; n < m_num_collectors; n++) {
        unsigned c = (m_last_cu + n + rr_increment) % m_num_collectors;
        if ((*m_collector_units)[c].ready()) {
          m_last_cu = c;
          return &((*m_collector_units)[c]);
        }
      }
      return NULL;
    }

   private:
    unsigned m_num_collectors;
    std::vector<collector_unit_t> *m_collector_units;
    unsigned m_last_cu;  // dispatch ready cu's rr
    unsigned m_next_cu;  // for initialization
    bool m_sub_core_model;
    unsigned m_num_warp_scheds;
  };

  // opndcoll_rfu_t data members
  bool m_initialized;

  unsigned m_num_collector_sets;
  // unsigned m_num_collectors;
  unsigned m_num_banks;
  unsigned m_bank_warp_shift;
  unsigned m_warp_size;
  std::vector<collector_unit_t *> m_cu;
  arbiter_t m_arbiter;

  unsigned m_num_banks_per_sched;
  unsigned m_num_warp_scheds;
  bool sub_core_model;

  // unsigned m_num_ports;
  // std::vector<warp_inst_t**> m_input;
  // std::vector<warp_inst_t**> m_output;
  // std::vector<unsigned> m_num_collector_units;
  // warp_inst_t **m_alu_port;

  std::vector<input_port_t> m_in_ports;
  typedef std::map<unsigned /* collector set */,
                   std::vector<collector_unit_t> /*collector sets*/>
      cu_sets_t;
  cu_sets_t m_cus;
  std::vector<dispatch_unit_t> m_dispatch_units;

  // typedef std::map<warp_inst_t**/*port*/,dispatch_unit_t> port_to_du_t;
  // port_to_du_t                     m_dispatch_units;
  // std::map<warp_inst_t**,std::list<collector_unit_t*> > m_free_cu;
  shader_core_ctx *m_shader;
};

class barrier_set_t {
 public:
  barrier_set_t(shader_core_ctx *shader, unsigned max_warps_per_core,
                unsigned max_cta_per_core, unsigned max_barriers_per_cta,
                unsigned warp_size);

  // during cta allocation
  void allocate_barrier(unsigned cta_id, warp_set_t warps);

  // during cta deallocation
  void deallocate_barrier(unsigned cta_id);

  typedef std::map<unsigned, warp_set_t> cta_to_warp_t;
  typedef std::map<unsigned, warp_set_t>
      bar_id_to_warp_t; /*set of warps reached a specific barrier id*/

  // individual warp hits barrier
  void warp_reaches_barrier(unsigned cta_id, unsigned warp_id,
                            warp_inst_t *inst);

  // warp reaches exit
  void warp_exit(unsigned warp_id);

  // assertions
  bool warp_waiting_at_barrier(unsigned warp_id) const;

  // debug
  void dump();

 private:
  unsigned m_max_cta_per_core;
  unsigned m_max_warps_per_core;
  unsigned m_max_barriers_per_cta;
  unsigned m_warp_size;
  cta_to_warp_t m_cta_to_warps;
  bar_id_to_warp_t m_bar_id_to_warps;
  warp_set_t m_warp_active;
  warp_set_t m_warp_at_barrier;
  shader_core_ctx *m_shader;
};

struct insn_latency_info {
  unsigned pc;
  unsigned long latency;
};

struct ifetch_buffer_t {
  ifetch_buffer_t() { m_valid = false; }

  ifetch_buffer_t(address_type pc, unsigned nbytes, unsigned warp_id) {
    m_valid = true;
    m_pc = pc;
    m_nbytes = nbytes;
    m_warp_id = warp_id;
  }

  bool m_valid;
  address_type m_pc;
  unsigned m_nbytes;
  unsigned m_warp_id;
};

class shader_core_config;

class simd_function_unit {
 public:
  simd_function_unit(const shader_core_config *config);
  ~simd_function_unit() { delete m_dispatch_reg; }

  // modifiers
  virtual void issue(register_set &source_reg);
  virtual void cycle() = 0;
  virtual void active_lanes_in_pipeline() = 0;

  // accessors
  virtual unsigned clock_multiplier() const { return 1; }
  virtual bool can_issue(const warp_inst_t &inst) const {
    return m_dispatch_reg->empty() && !occupied.test(inst.latency);
  }
  virtual bool is_issue_partitioned() = 0;
  virtual unsigned get_issue_reg_id() = 0;
  virtual bool stallable() const = 0;
  virtual void print(FILE *fp) const {
    fprintf(fp, "%s dispatch= ", m_name.c_str());
    m_dispatch_reg->print(fp);
  }
  const char *get_name() { return m_name.c_str(); }

 protected:
  std::string m_name;
  const shader_core_config *m_config;
  warp_inst_t *m_dispatch_reg;
  static const unsigned MAX_ALU_LATENCY = 512;
  std::bitset<MAX_ALU_LATENCY> occupied;
};

class pipelined_simd_unit : public simd_function_unit {
 public:
  pipelined_simd_unit(register_set *result_port,
                      const shader_core_config *config, unsigned max_latency,
                      shader_core_ctx *core, unsigned issue_reg_id);

  // modifiers
  virtual void cycle();
  virtual void issue(register_set &source_reg);
  virtual unsigned get_active_lanes_in_pipeline();

  virtual void active_lanes_in_pipeline() = 0;
  /*
      virtual void issue( register_set& source_reg )
      {
          //move_warp(m_dispatch_reg,source_reg);
          //source_reg.move_out_to(m_dispatch_reg);
          simd_function_unit::issue(source_reg);
      }
  */
  // accessors
  virtual bool stallable() const { return false; }
  virtual bool can_issue(const warp_inst_t &inst) const {
    return simd_function_unit::can_issue(inst);
  }
  virtual bool is_issue_partitioned() = 0;
  unsigned get_issue_reg_id() { return m_issue_reg_id; }
  virtual void print(FILE *fp) const {
    simd_function_unit::print(fp);
    for (int s = m_pipeline_depth - 1; s >= 0; s--) {
      if (!m_pipeline_reg[s]->empty()) {
        fprintf(fp, "      %s[%2d] ", m_name.c_str(), s);
        m_pipeline_reg[s]->print(fp);
      }
    }
  }

 protected:
  unsigned m_pipeline_depth;
  warp_inst_t **m_pipeline_reg;
  register_set *m_result_port;
  class shader_core_ctx *m_core;
  unsigned m_issue_reg_id;  // if sub_core_model is enabled we can only issue
                            // from a subset of operand collectors

  unsigned active_insts_in_pipeline;
};

class sfu : public pipelined_simd_unit {
 public:
  sfu(register_set *result_port, const shader_core_config *config,
      shader_core_ctx *core, unsigned issue_reg_id);
  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
      case SFU_OP:
        break;
      case ALU_SFU_OP:
        break;
      case DP_OP:
        break;  // for compute <= 29 (i..e Fermi and GT200)
      default:
        return false;
    }
    return pipelined_simd_unit::can_issue(inst);
  }
  virtual void active_lanes_in_pipeline();
  virtual void issue(register_set &source_reg);
  bool is_issue_partitioned() { return true; }
};

class dp_unit : public pipelined_simd_unit {
 public:
  dp_unit(register_set *result_port, const shader_core_config *config,
          shader_core_ctx *core, unsigned issue_reg_id);
  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
      case DP_OP:
        break;
      default:
        return false;
    }
    return pipelined_simd_unit::can_issue(inst);
  }
  virtual void active_lanes_in_pipeline();
  virtual void issue(register_set &source_reg);
  bool is_issue_partitioned() { return true; }
};

class tensor_core : public pipelined_simd_unit {
 public:
  tensor_core(register_set *result_port, const shader_core_config *config,
              shader_core_ctx *core, unsigned issue_reg_id);
  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
      case TENSOR_CORE_OP:
        break;
      default:
        return false;
    }
    return pipelined_simd_unit::can_issue(inst);
  }
  virtual void active_lanes_in_pipeline();
  virtual void issue(register_set &source_reg);
  bool is_issue_partitioned() { return true; }
};

class int_unit : public pipelined_simd_unit {
 public:
  int_unit(register_set *result_port, const shader_core_config *config,
           shader_core_ctx *core, unsigned issue_reg_id);
  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
      case SFU_OP:
        return false;
      case LOAD_OP:
        return false;
      case TENSOR_CORE_LOAD_OP:
        return false;
      case STORE_OP:
        return false;
      case TENSOR_CORE_STORE_OP:
        return false;
      case MEMORY_BARRIER_OP:
        return false;
      case SP_OP:
        return false;
      case DP_OP:
        return false;
      default:
        break;
    }
    return pipelined_simd_unit::can_issue(inst);
  }
  virtual void active_lanes_in_pipeline();
  virtual void issue(register_set &source_reg);
  bool is_issue_partitioned() { return true; }
};

class sp_unit : public pipelined_simd_unit {
 public:
  sp_unit(register_set *result_port, const shader_core_config *config,
          shader_core_ctx *core, unsigned issue_reg_id);
  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
      case SFU_OP:
        return false;
      case LOAD_OP:
        return false;
      case TENSOR_CORE_LOAD_OP:
        return false;
      case STORE_OP:
        return false;
      case TENSOR_CORE_STORE_OP:
        return false;
      case MEMORY_BARRIER_OP:
        return false;
      case DP_OP:
        return false;
      default:
        break;
    }
    return pipelined_simd_unit::can_issue(inst);
  }
  virtual void active_lanes_in_pipeline();
  virtual void issue(register_set &source_reg);
  bool is_issue_partitioned() { return true; }
};

class specialized_unit : public pipelined_simd_unit {
 public:
  specialized_unit(register_set *result_port, const shader_core_config *config,
                   shader_core_ctx *core, unsigned supported_op,
                   char *unit_name, unsigned latency, unsigned issue_reg_id);
  virtual bool can_issue(const warp_inst_t &inst) const {
    if (inst.op != m_supported_op) {
      return false;
    }
    return pipelined_simd_unit::can_issue(inst);
  }
  virtual void active_lanes_in_pipeline();
  virtual void issue(register_set &source_reg);
  bool is_issue_partitioned() { return true; }

 private:
  unsigned m_supported_op;
};

class simt_core_cluster;
class shader_memory_interface;
class shader_core_mem_fetch_allocator;
class cache_t;

class ldst_unit : public pipelined_simd_unit {
 public:
  ldst_unit(mem_fetch_interface *icnt,
            shader_core_mem_fetch_allocator *mf_allocator,
            shader_core_ctx *core, opndcoll_rfu_t *operand_collector,
            Scoreboard *scoreboard, const shader_core_config *config,
            const memory_config *mem_config, class shader_core_stats *stats,
            unsigned sid, unsigned tpc);

  // modifiers
  virtual void issue(register_set &inst);
  bool is_issue_partitioned() { return false; }
  virtual void cycle();

  void fill(mem_fetch *mf);
  void flush();
  void invalidate();
  void writeback();

  // accessors
  virtual unsigned clock_multiplier() const;

  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
      case LOAD_OP:
        break;
      case TENSOR_CORE_LOAD_OP:
        break;
      case STORE_OP:
        break;
      case TENSOR_CORE_STORE_OP:
        break;
      case MEMORY_BARRIER_OP:
        break;
      default:
        return false;
    }
    return m_dispatch_reg->empty();
  }

  virtual void active_lanes_in_pipeline();
  virtual bool stallable() const { return true; }
  bool response_buffer_full() const;
  void print(FILE *fout) const;
  void print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                         unsigned &dl1_misses);
  void get_cache_stats(unsigned &read_accesses, unsigned &write_accesses,
                       unsigned &read_misses, unsigned &write_misses,
                       unsigned cache_type);
  void get_cache_stats(cache_stats &cs);

  void get_L1D_sub_stats(struct cache_sub_stats &css) const;
  void get_L1C_sub_stats(struct cache_sub_stats &css) const;
  void get_L1T_sub_stats(struct cache_sub_stats &css) const;
  unsigned m_sid;


 protected:
  ldst_unit(mem_fetch_interface *icnt,
            shader_core_mem_fetch_allocator *mf_allocator,
            shader_core_ctx *core, opndcoll_rfu_t *operand_collector,
            Scoreboard *scoreboard, const shader_core_config *config,
            const memory_config *mem_config, shader_core_stats *stats,
            unsigned sid, unsigned tpc, l1_cache *new_l1d_cache);
  void init(mem_fetch_interface *icnt,
            shader_core_mem_fetch_allocator *mf_allocator,
            shader_core_ctx *core, opndcoll_rfu_t *operand_collector,
            Scoreboard *scoreboard, const shader_core_config *config,
            const memory_config *mem_config, shader_core_stats *stats,
            unsigned sid, unsigned tpc);

 protected:
  bool shared_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
                    mem_stage_access_type &fail_type);
  bool constant_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
                      mem_stage_access_type &fail_type);
  bool texture_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
                     mem_stage_access_type &fail_type);
  bool memory_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
                    mem_stage_access_type &fail_type);

  virtual mem_stage_stall_type process_cache_access(
      cache_t *cache, new_addr_type address, warp_inst_t &inst,
      std::list<cache_event> &events, mem_fetch *mf,
      enum cache_request_status status);
  mem_stage_stall_type process_memory_access_queue(cache_t *cache,
                                                   warp_inst_t &inst);
  mem_stage_stall_type process_memory_access_queue_l1cache(l1_cache *cache,
                                                           warp_inst_t &inst);

  const memory_config *m_memory_config;
  class mem_fetch_interface *m_icnt;
  shader_core_mem_fetch_allocator *m_mf_allocator;
  class shader_core_ctx *m_core;
  unsigned m_tpc;

  tex_cache *m_L1T;        // texture cache
  read_only_cache *m_L1C;  // constant cache
  l1_cache *m_L1D;         // data cache
  std::map<unsigned /*warp_id*/,
           std::map<unsigned /*regnum*/, unsigned /*count*/>>
      m_pending_writes;
  std::list<mem_fetch *> m_response_fifo;
  opndcoll_rfu_t *m_operand_collector;
  Scoreboard *m_scoreboard;

  mem_fetch *m_next_global;
  warp_inst_t m_next_wb;
  unsigned m_writeback_arb;  // round-robin arbiter for writeback contention
                             // between L1T, L1C, shared
  unsigned m_num_writeback_clients;

  enum mem_stage_stall_type m_mem_rc;

  shader_core_stats *m_stats;

  // for debugging
  unsigned long long m_last_inst_gpu_sim_cycle;
  unsigned long long m_last_inst_gpu_tot_sim_cycle;

  std::vector<std::deque<mem_fetch *>> l1_latency_queue;
  void L1_latency_queue_cycle();
};

enum pipeline_stage_name_t {
  ID_OC_SP = 0,
  ID_OC_DP,
  ID_OC_INT,
  ID_OC_SFU,
  ID_OC_MEM,
  OC_EX_SP,
  OC_EX_DP,
  OC_EX_INT,
  OC_EX_SFU,
  OC_EX_MEM,
  EX_WB,
  ID_OC_TENSOR_CORE,
  OC_EX_TENSOR_CORE,
  N_PIPELINE_STAGES
};

const char *const pipeline_stage_name_decode[] = {
    "ID_OC_SP",          "ID_OC_DP",         "ID_OC_INT", "ID_OC_SFU",
    "ID_OC_MEM",         "OC_EX_SP",         "OC_EX_DP",  "OC_EX_INT",
    "OC_EX_SFU",         "OC_EX_MEM",        "EX_WB",     "ID_OC_TENSOR_CORE",
    "OC_EX_TENSOR_CORE", "N_PIPELINE_STAGES"};

struct specialized_unit_params {
  unsigned latency;
  unsigned num_units;
  unsigned id_oc_spec_reg_width;
  unsigned oc_ex_spec_reg_width;
  char name[20];
  unsigned ID_OC_SPEC_ID;
  unsigned OC_EX_SPEC_ID;
};

class shader_core_config : public core_config {
 public:
  shader_core_config(gpgpu_context *ctx) : core_config(ctx) {
    pipeline_widths_string = NULL;
    gpgpu_ctx = ctx;
  }

  void init() {
    int ntok = sscanf(gpgpu_shader_core_pipeline_opt, "%d:%d",
                      &n_thread_per_shader, &warp_size);
    if (ntok != 2) {
      printf(
          "GPGPU-Sim uArch: error while parsing configuration string "
          "gpgpu_shader_core_pipeline_opt\n");
      abort();
    }

    char *toks = new char[100];
    char *tokd = toks;
    strcpy(toks, pipeline_widths_string);

    toks = strtok(toks, ",");

    /*	Removing the tensorcore pipeline while reading the config files if the
       tensor core is not available. If we won't remove it, old regression will
       be broken. So to support the legacy config files it's best to handle in
       this way.
     */
    int num_config_to_read = N_PIPELINE_STAGES - 2 * (!gpgpu_tensor_core_avail);

    for (int i = 0; i < num_config_to_read; i++) {
      assert(toks);
      ntok = sscanf(toks, "%d", &pipe_widths[i]);
      assert(ntok == 1);
      toks = strtok(NULL, ",");
    }

    delete[] tokd;

    if (n_thread_per_shader > MAX_THREAD_PER_SM) {
      printf(
          "GPGPU-Sim uArch: Error ** increase MAX_THREAD_PER_SM in "
          "abstract_hardware_model.h from %u to %u\n",
          MAX_THREAD_PER_SM, n_thread_per_shader);
      abort();
    }
    max_warps_per_shader = n_thread_per_shader / warp_size;
    assert(!(n_thread_per_shader % warp_size));

    set_pipeline_latency();

    m_L1I_config.init(m_L1I_config.m_config_string, FuncCachePreferNone);
    m_L1T_config.init(m_L1T_config.m_config_string, FuncCachePreferNone);
    m_L1C_config.init(m_L1C_config.m_config_string, FuncCachePreferNone);
    m_L1D_config.init(m_L1D_config.m_config_string, FuncCachePreferNone);
    gpgpu_cache_texl1_linesize = m_L1T_config.get_line_sz();
    gpgpu_cache_constl1_linesize = m_L1C_config.get_line_sz();
    m_valid = true;

    m_specialized_unit_num = 0;
    // parse the specialized units
    for (unsigned i = 0; i < SPECIALIZED_UNIT_NUM; ++i) {
      unsigned enabled;
      specialized_unit_params sparam;
      sscanf(specialized_unit_string[i], "%u,%u,%u,%u,%u,%s", &enabled,
             &sparam.num_units, &sparam.latency, &sparam.id_oc_spec_reg_width,
             &sparam.oc_ex_spec_reg_width, sparam.name);

      if (enabled) {
        m_specialized_unit.push_back(sparam);
        strncpy(m_specialized_unit.back().name, sparam.name,
                sizeof(m_specialized_unit.back().name));
        m_specialized_unit_num += sparam.num_units;
      } else
        break;  // we only accept continuous specialized_units, i.e., 1,2,3,4
    }

    // parse gpgpu_shmem_option for adpative cache config
    if (adaptive_cache_config) {
      std::stringstream ss(gpgpu_shmem_option);
      while (ss.good()) {
        std::string option;
        std::getline(ss, option, ',');
        shmem_opt_list.push_back((unsigned)std::stoi(option) * 1024);
      }
      std::sort(shmem_opt_list.begin(), shmem_opt_list.end());
    }
  }
  void reg_options(class OptionParser *opp);
  unsigned max_cta(const kernel_info_t &k) const;
  unsigned num_shader() const {
    return n_simt_clusters * n_simt_cores_per_cluster;
  }
  unsigned sid_to_cluster(unsigned sid) const {
    return sid / n_simt_cores_per_cluster;
  }
  unsigned sid_to_cid(unsigned sid) const {
    return sid % n_simt_cores_per_cluster;
  }
  unsigned cid_to_sid(unsigned cid, unsigned cluster_id) const {
    return cluster_id * n_simt_cores_per_cluster + cid;
  }
  void set_pipeline_latency();

  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  // data
  char *gpgpu_shader_core_pipeline_opt;
  bool gpgpu_perfect_mem;
  bool gpgpu_perfect_mem_data;		
  bool gpgpu_write_sched_order; // write sched order to file		
  bool gpgpu_follow_defined_sched_order;  // follow sched order from file		
  bool gpgpu_print_stall_data;		
  bool gpgpu_print_cout_statements;		
  bool gpgpu_sched_all_warps;		
  bool gpgpu_reply_buffer;		
  //int gpgpu_limit_OoO;	
  bool gpgpu_reply_buffer_seperate_mem_comp;
  bool gpgpu_clock_gated_reg_file;
  bool gpgpu_clock_gated_lanes;
  enum divergence_support_t model;
  unsigned n_thread_per_shader;
  unsigned n_regfile_gating_group;
  unsigned max_warps_per_shader;
  unsigned
      max_cta_per_core;  // Limit on number of concurrent CTAs in shader core
  unsigned max_barriers_per_cta;
  char *gpgpu_scheduler_string;
  unsigned gpgpu_shmem_per_block;
  unsigned gpgpu_registers_per_block;
  char *pipeline_widths_string;
  int pipe_widths[N_PIPELINE_STAGES];

  mutable cache_config m_L1I_config;
  mutable cache_config m_L1T_config;
  mutable cache_config m_L1C_config;
  mutable l1d_cache_config m_L1D_config;

  bool gpgpu_dwf_reg_bankconflict;

  unsigned gpgpu_num_sched_per_core;
  int gpgpu_max_insn_issue_per_warp;
  bool gpgpu_dual_issue_diff_exec_units;

  // op collector
  bool enable_specialized_operand_collector;
  int gpgpu_operand_collector_num_units_sp;
  int gpgpu_operand_collector_num_units_dp;
  int gpgpu_operand_collector_num_units_sfu;
  int gpgpu_operand_collector_num_units_tensor_core;
  int gpgpu_operand_collector_num_units_mem;
  int gpgpu_operand_collector_num_units_gen;
  int gpgpu_operand_collector_num_units_int;

  unsigned int gpgpu_operand_collector_num_in_ports_sp;
  unsigned int gpgpu_operand_collector_num_in_ports_dp;
  unsigned int gpgpu_operand_collector_num_in_ports_sfu;
  unsigned int gpgpu_operand_collector_num_in_ports_tensor_core;
  unsigned int gpgpu_operand_collector_num_in_ports_mem;
  unsigned int gpgpu_operand_collector_num_in_ports_gen;
  unsigned int gpgpu_operand_collector_num_in_ports_int;

  unsigned int gpgpu_operand_collector_num_out_ports_sp;
  unsigned int gpgpu_operand_collector_num_out_ports_dp;
  unsigned int gpgpu_operand_collector_num_out_ports_sfu;
  unsigned int gpgpu_operand_collector_num_out_ports_tensor_core;
  unsigned int gpgpu_operand_collector_num_out_ports_mem;
  unsigned int gpgpu_operand_collector_num_out_ports_gen;
  unsigned int gpgpu_operand_collector_num_out_ports_int;

  int gpgpu_num_sp_units;
  int gpgpu_tensor_core_avail;
  int gpgpu_num_dp_units;
  int gpgpu_num_sfu_units;
  int gpgpu_num_tensor_core_units;
  int gpgpu_num_mem_units;
  int gpgpu_num_int_units;

  int ib_size_config;	
  int ib_size_ooo;	
  int ib_size_in;	
  int streaming_ib;	
  int perfect_memory_aliasing;

  // Shader core resources
  unsigned gpgpu_shader_registers;
  int gpgpu_warpdistro_shader;
  int gpgpu_warp_issue_shader;
  unsigned gpgpu_num_reg_banks;
  bool gpgpu_reg_bank_use_warp_id;
  bool gpgpu_local_mem_map;
  bool gpgpu_ignore_resources_limitation;
  bool sub_core_model;

  bool perfect_control;		
  bool ignore_synchronization;		
  bool pending_write_ignore;	


  unsigned max_sp_latency;
  unsigned max_int_latency;
  unsigned max_sfu_latency;
  unsigned max_dp_latency;
  unsigned max_tensor_core_latency;

  unsigned n_simt_cores_per_cluster;
  unsigned n_simt_clusters;
  unsigned n_simt_ejection_buffer_size;
  unsigned ldst_unit_response_queue_size;

  int simt_core_sim_order;

  unsigned smem_latency;

  unsigned mem2device(unsigned memid) const { return memid + n_simt_clusters; }

  // Jin: concurrent kernel on sm
  bool gpgpu_concurrent_kernel_sm;

  bool perfect_inst_const_cache;
  unsigned inst_fetch_throughput;
  unsigned reg_file_port_throughput;

  // specialized unit config strings
  char *specialized_unit_string[SPECIALIZED_UNIT_NUM];
  mutable std::vector<specialized_unit_params> m_specialized_unit;
  unsigned m_specialized_unit_num;
};

struct shader_core_stats_pod {
  void *
      shader_core_stats_pod_start[0];  // DO NOT MOVE FROM THE TOP - spaceless
                                       // pointer to the start of this structure
  unsigned long long *shader_cycles;
  unsigned *m_num_sim_insn;   // number of scalar thread instructions committed
                              // by this shader core
  unsigned *m_num_sim_winsn;  // number of warp instructions committed by this
                              // shader core
  unsigned *m_last_num_sim_insn;
  unsigned *m_last_num_sim_winsn;
  unsigned *
      m_num_decoded_insn;  // number of instructions decoded by this shader core
  unsigned *	
      m_num_tried_to_issue_insn;  // number of cycles inst was tried to be issued	
  unsigned *m_num_tried_to_issue_DEB;	
  unsigned *m_num_put_inst_in_DEB;
  float *m_pipeline_duty_cycle;
  unsigned *m_num_FPdecoded_insn;
  unsigned *m_num_INTdecoded_insn;
  unsigned *m_num_storequeued_insn;
  unsigned *m_num_loadqueued_insn;
  unsigned *m_num_tex_inst;
  double *m_num_ialu_acesses;
  double *m_num_fp_acesses;
  double *m_num_imul_acesses;
  double *m_num_fpmul_acesses;
  double *m_num_idiv_acesses;
  double *m_num_fpdiv_acesses;
  double *m_num_sp_acesses;
  double *m_num_sfu_acesses;
  double *m_num_tensor_core_acesses;
  double *m_num_tex_acesses;
  double *m_num_const_acesses;
  double *m_num_dp_acesses;
  double *m_num_dpmul_acesses;
  double *m_num_dpdiv_acesses;
  double *m_num_sqrt_acesses;
  double *m_num_log_acesses;
  double *m_num_sin_acesses;
  double *m_num_exp_acesses;
  double *m_num_mem_acesses;
  unsigned *m_num_sp_committed;
  unsigned *m_num_tlb_hits;
  unsigned *m_num_tlb_accesses;
  unsigned *m_num_sfu_committed;
  unsigned *m_num_tensor_core_committed;
  unsigned *m_num_mem_committed;
  unsigned *m_read_regfile_acesses;
  unsigned *m_write_regfile_acesses;
  unsigned *m_non_rf_operands;
  double *m_num_imul24_acesses;
  double *m_num_imul32_acesses;
  unsigned *m_active_sp_lanes;
  unsigned *m_active_sfu_lanes;
  unsigned *m_active_tensor_core_lanes;
  unsigned *m_active_fu_lanes;
  unsigned *m_active_fu_mem_lanes;
  double *m_active_exu_threads; //For power model
  double *m_active_exu_warps; //For power model
  unsigned *m_n_diverge;  // number of divergence occurring in this shader
  unsigned gpgpu_n_load_insn;
  unsigned gpgpu_n_store_insn;
  unsigned gpgpu_n_shmem_insn;
  unsigned gpgpu_n_sstarr_insn;
  unsigned gpgpu_n_tex_insn;
  unsigned gpgpu_n_const_insn;
  unsigned gpgpu_n_param_insn;
  unsigned gpgpu_n_shmem_bkconflict;
  unsigned gpgpu_n_cache_bkconflict;
  int gpgpu_n_intrawarp_mshr_merge;
  unsigned gpgpu_n_cmem_portconflict;
  unsigned gpu_stall_shd_mem_breakdown[N_MEM_STAGE_ACCESS_TYPE]
                                      [N_MEM_STAGE_STALL_TYPE];
  unsigned gpu_reg_bank_conflict_stalls;
  unsigned *shader_cycle_distro;
  unsigned *last_shader_cycle_distro;
  unsigned *num_warps_issuable;
  unsigned gpgpu_n_stall_shd_mem;
  unsigned *single_issue_nums;
  unsigned *dual_issue_nums;

  unsigned ctas_completed;
  // memory access classification
  int gpgpu_n_mem_read_local;
  int gpgpu_n_mem_write_local;
  int gpgpu_n_mem_texture;
  int gpgpu_n_mem_const;
  int gpgpu_n_mem_read_global;
  int gpgpu_n_mem_write_global;
  int gpgpu_n_mem_read_inst;

  int gpgpu_n_mem_l2_writeback;
  int gpgpu_n_mem_l1_write_allocate;
  int gpgpu_n_mem_l2_write_allocate;

  unsigned made_write_mfs;
  unsigned made_read_mfs;

  unsigned *gpgpu_n_shmem_bank_access;
  long *n_simt_to_mem;  // Interconnect power stats
  long *n_mem_to_simt;
};

class shader_core_stats : public shader_core_stats_pod {
 public:
  shader_core_stats(const shader_core_config *config) {
    m_config = config;
    shader_core_stats_pod *pod = reinterpret_cast<shader_core_stats_pod *>(
        this->shader_core_stats_pod_start);
    memset(pod, 0, sizeof(shader_core_stats_pod));
    shader_cycles = (unsigned long long *)calloc(config->num_shader(),
                                                 sizeof(unsigned long long));
    m_num_sim_insn = (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_sim_winsn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_last_num_sim_winsn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_last_num_sim_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_pipeline_duty_cycle =
        (float *)calloc(config->num_shader(), sizeof(float));
     m_num_tried_to_issue_insn =	
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));	
    m_num_tried_to_issue_DEB =	
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));	
    m_num_put_inst_in_DEB =	
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_decoded_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_FPdecoded_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_storequeued_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_loadqueued_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_tex_inst = 
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_INTdecoded_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_ialu_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_fp_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_imul_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_imul24_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_imul32_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_fpmul_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_idiv_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_fpdiv_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_dp_acesses = 
        (double*) calloc(config->num_shader(),sizeof(double));
    m_num_dpmul_acesses = 
        (double*) calloc(config->num_shader(),sizeof(double));
    m_num_dpdiv_acesses = 
        (double*) calloc(config->num_shader(),sizeof(double));
    m_num_sp_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_sfu_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_tensor_core_acesses = 
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_const_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_tex_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_sqrt_acesses = 
        (double*) calloc(config->num_shader(),sizeof(double));
    m_num_log_acesses = 
        (double*) calloc(config->num_shader(),sizeof(double));
    m_num_sin_acesses = 
        (double*) calloc(config->num_shader(),sizeof(double));
    m_num_exp_acesses = 
        (double*) calloc(config->num_shader(),sizeof(double));
    m_num_mem_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_sp_committed =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_tlb_hits = 
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_tlb_accesses =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_active_sp_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_active_sfu_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_active_tensor_core_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_active_fu_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_active_exu_threads =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_active_exu_warps =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_active_fu_mem_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_sfu_committed =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_tensor_core_committed =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_mem_committed =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_read_regfile_acesses =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_write_regfile_acesses =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_non_rf_operands =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_n_diverge = 
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    shader_cycle_distro =
        (unsigned *)calloc(config->warp_size + 3, sizeof(unsigned));
    last_shader_cycle_distro =
        (unsigned *)calloc(m_config->warp_size + 3, sizeof(unsigned));
    single_issue_nums =
        (unsigned *)calloc(config->gpgpu_num_sched_per_core, sizeof(unsigned));
    dual_issue_nums =
        (unsigned *)calloc(config->gpgpu_num_sched_per_core, sizeof(unsigned));

    ctas_completed = 0;
    n_simt_to_mem = (long *)calloc(config->num_shader(), sizeof(long));
    n_mem_to_simt = (long *)calloc(config->num_shader(), sizeof(long));

    m_outgoing_traffic_stats = new traffic_breakdown("coretomem");
    m_incoming_traffic_stats = new traffic_breakdown("memtocore");

    gpgpu_n_shmem_bank_access =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));

    m_shader_dynamic_warp_issue_distro.resize(config->num_shader());
    m_shader_warp_slot_issue_distro.resize(config->num_shader());
  }

  ~shader_core_stats() {
    delete m_outgoing_traffic_stats;
    delete m_incoming_traffic_stats;
    free(m_num_sim_insn);
    free(m_num_sim_winsn);
    free(m_num_FPdecoded_insn);
    free(m_num_INTdecoded_insn);
    free(m_num_storequeued_insn);
    free(m_num_loadqueued_insn);
    free(m_num_ialu_acesses);
    free(m_num_fp_acesses);
    free(m_num_imul_acesses);
    free(m_num_tex_inst);
    free(m_num_fpmul_acesses);
    free(m_num_idiv_acesses);
    free(m_num_fpdiv_acesses);
    free(m_num_sp_acesses);
    free(m_num_sfu_acesses);
    free(m_num_tensor_core_acesses);
    free(m_num_tex_acesses);
    free(m_num_const_acesses);
    free(m_num_dp_acesses);
    free(m_num_dpmul_acesses);
    free(m_num_dpdiv_acesses);
    free(m_num_sqrt_acesses);
    free(m_num_log_acesses);
    free(m_num_sin_acesses);
    free(m_num_exp_acesses);
    free(m_num_mem_acesses);
    free(m_num_sp_committed);
    free(m_num_tlb_hits);
    free(m_num_tlb_accesses);
    free(m_num_sfu_committed);
    free(m_num_tensor_core_committed);
    free(m_num_mem_committed);
    free(m_read_regfile_acesses);
    free(m_write_regfile_acesses);
    free(m_non_rf_operands);
    free(m_num_imul24_acesses);
    free(m_num_imul32_acesses);
    free(m_active_sp_lanes);
    free(m_active_sfu_lanes);
    free(m_active_tensor_core_lanes);
    free(m_active_fu_lanes);
    free(m_active_exu_threads);
    free(m_active_exu_warps);
    free(m_active_fu_mem_lanes);
    free(m_n_diverge);
    free(shader_cycle_distro);
    free(last_shader_cycle_distro);
  }

  void new_grid() {}

  void event_warp_issued(unsigned s_id, unsigned warp_id, unsigned num_issued,
                         unsigned dynamic_warp_id);

  void visualizer_print(gzFile visualizer_file);

  void print(FILE *fout) const;

  const std::vector<std::vector<unsigned>> &get_dynamic_warp_issue() const {
    return m_shader_dynamic_warp_issue_distro;
  }

  const std::vector<std::vector<unsigned>> &get_warp_slot_issue() const {
    return m_shader_warp_slot_issue_distro;
  }

 private:
  const shader_core_config *m_config;

  traffic_breakdown *m_outgoing_traffic_stats;  // core to memory partitions
  traffic_breakdown *m_incoming_traffic_stats;  // memory partition to core

  // Counts the instructions issued for each dynamic warp.
  std::vector<std::vector<unsigned>> m_shader_dynamic_warp_issue_distro;
  std::vector<unsigned> m_last_shader_dynamic_warp_issue_distro;
  std::vector<std::vector<unsigned>> m_shader_warp_slot_issue_distro;
  std::vector<unsigned> m_last_shader_warp_slot_issue_distro;

  friend class power_stat_t;
  friend class shader_core_ctx;
  friend class ldst_unit;
  friend class simt_core_cluster;
  friend class scheduler_unit;
  friend class TwoLevelScheduler;
  friend class LooseRoundRobbinScheduler;
};

class memory_config;
class shader_core_mem_fetch_allocator : public mem_fetch_allocator {
 public:
  shader_core_mem_fetch_allocator(unsigned core_id, unsigned cluster_id,
                                  const memory_config *config) {
    m_core_id = core_id;
    m_cluster_id = cluster_id;
    m_memory_config = config;
  }
  mem_fetch *alloc(new_addr_type addr, mem_access_type type, unsigned size,
                   bool wr, unsigned long long cycle) const;
  mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                   const active_mask_t &active_mask,
                   const mem_access_byte_mask_t &byte_mask,
                   const mem_access_sector_mask_t &sector_mask, unsigned size,
                   bool wr, unsigned long long cycle, unsigned wid,
                   unsigned sid, unsigned tpc, mem_fetch *original_mf) const;
  mem_fetch *alloc(const warp_inst_t &inst, const mem_access_t &access,
                   unsigned long long cycle) const {
    warp_inst_t inst_copy = inst;
    mem_fetch *mf = new mem_fetch(
        access, &inst_copy,
        access.is_write() ? WRITE_PACKET_SIZE : READ_PACKET_SIZE,
        inst.warp_id(), m_core_id, m_cluster_id, m_memory_config, cycle);
    return mf;
  }

 private:
  unsigned m_core_id;
  unsigned m_cluster_id;
  const memory_config *m_memory_config;
};

class shader_core_ctx : public core_t {
 public:
  // creator:
  shader_core_ctx(class gpgpu_sim *gpu, class simt_core_cluster *cluster,
                  unsigned shader_id, unsigned tpc_id,
                  const shader_core_config *config,
                  const memory_config *mem_config, shader_core_stats *stats);

  // used by simt_core_cluster:
  // modifiers
  void cycle();
  void reinit(unsigned start_thread, unsigned end_thread,
              bool reset_not_completed);
  void issue_block2core(class kernel_info_t &kernel);

  void cache_flush();
  void cache_invalidate();
  void accept_fetch_response(mem_fetch *mf);
  void accept_ldst_unit_response(class mem_fetch *mf);
  void broadcast_barrier_reduction(unsigned cta_id, unsigned bar_id,
                                   warp_set_t warps);
  void set_kernel(kernel_info_t *k) {
    assert(k);
    m_kernel = k;
    //        k->inc_running();
    printf("GPGPU-Sim uArch: Shader %d bind to kernel %u \'%s\'\n", m_sid,
           m_kernel->get_uid(), m_kernel->name().c_str());
  }
  PowerscalingCoefficients *scaling_coeffs;
  // accessors
  bool fetch_unit_response_buffer_full() const;
  bool ldst_unit_response_buffer_full() const;
  unsigned get_not_completed() const { return m_not_completed; }
  unsigned get_n_active_cta() const { return m_n_active_cta; }
  unsigned isactive() const {
    if (m_n_active_cta > 0)
      return 1;
    else
      return 0;
  }
  kernel_info_t *get_kernel() { return m_kernel; }
  unsigned get_sid() const { return m_sid; }

  int get_stalled_on_TLB(int wid) {return m_warp[wid]->last_pc_decoded_streaming_test; }

  void set_stalled_on_TLB(int wid, int val) { m_warp[wid]->last_pc_decoded_streaming_test = val; }

  void reduceLoadsToTLB(int wid) { m_warp[wid]->reduceInstToTLB(); }

  int getLoadToTLB(int wid) { return m_warp[wid]->getLoadToTLB(); }

  void reducePriorLoads(int wid) { m_warp[wid]->reducePriorLoads(); }

  // used by functional simulation:
  // modifiers
  virtual void warp_exit(unsigned warp_id);

  // accessors
  virtual bool warp_waiting_at_barrier(unsigned warp_id) const;
  void get_pdom_stack_top_info(unsigned tid, unsigned *pc, unsigned *rpc) const;
  float get_current_occupancy(unsigned long long &active,
                              unsigned long long &total) const;

  // used by pipeline timing model components:
  // modifiers
  void mem_instruction_stats(const warp_inst_t &inst);
  void decrement_atomic_count(unsigned wid, unsigned n);
  void inc_store_req(unsigned warp_id) { m_warp[warp_id]->inc_store_req(); }
  void dec_inst_in_pipeline(unsigned warp_id) {
    m_warp[warp_id]->dec_inst_in_pipeline();
  }  // also used in writeback()
  int set_wb(unsigned warp_id) {
    return m_warp[warp_id]->warp_wb = 1; 
  }

  void store_ack(class mem_fetch *mf);
  bool warp_waiting_at_mem_barrier(unsigned warp_id);
  void set_max_cta(const kernel_info_t &kernel);
  void warp_inst_complete(const warp_inst_t &inst);

  // accessors
  std::list<unsigned> get_regs_written(const inst_t &fvt) const;
  const shader_core_config *get_config() const { return m_config; }
  void print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                         unsigned &dl1_misses);

  void get_cache_stats(cache_stats &cs);
  void get_L1I_sub_stats(struct cache_sub_stats &css) const;
  void get_L1D_sub_stats(struct cache_sub_stats &css) const;
  void get_L1C_sub_stats(struct cache_sub_stats &css) const;
  void get_L1T_sub_stats(struct cache_sub_stats &css) const;

  void get_icnt_power_stats(long &n_simt_to_mem, long &n_mem_to_simt) const;

  // debug:
  void display_simt_state(FILE *fout, int mask) const;
  void display_pipeline(FILE *fout, int print_mem, int mask3bit) const;

  void incload_stat() { m_stats->m_num_loadqueued_insn[m_sid]++; }
  void incstore_stat() { m_stats->m_num_storequeued_insn[m_sid]++; }
  void incialu_stat(unsigned active_count,double latency) {
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_ialu_acesses[m_sid]=m_stats->m_num_ialu_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_nonsfu(active_count, latency);
    }else {
      m_stats->m_num_ialu_acesses[m_sid]=m_stats->m_num_ialu_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }
  void incimul_stat(unsigned active_count,double latency) {
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_imul_acesses[m_sid]=m_stats->m_num_imul_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_nonsfu(active_count, latency);
    }else {
      m_stats->m_num_imul_acesses[m_sid]=m_stats->m_num_imul_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }
  void incimul24_stat(unsigned active_count,double latency) {
  if(m_config->gpgpu_clock_gated_lanes==false){
    m_stats->m_num_imul24_acesses[m_sid]=m_stats->m_num_imul24_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_nonsfu(active_count, latency);
    }else {
      m_stats->m_num_imul24_acesses[m_sid]=m_stats->m_num_imul24_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;    
   }
   void incimul32_stat(unsigned active_count,double latency) {
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_imul32_acesses[m_sid]=m_stats->m_num_imul32_acesses[m_sid]+(double)active_count*latency
         + inactive_lanes_accesses_sfu(active_count, latency);          
    }else{
      m_stats->m_num_imul32_acesses[m_sid]=m_stats->m_num_imul32_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }
   void incidiv_stat(unsigned active_count,double latency) {
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_idiv_acesses[m_sid]=m_stats->m_num_idiv_acesses[m_sid]+(double)active_count*latency
         + inactive_lanes_accesses_sfu(active_count, latency); 
    }else {
      m_stats->m_num_idiv_acesses[m_sid]=m_stats->m_num_idiv_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;    
  }
   void incfpalu_stat(unsigned active_count,double latency) {
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_fp_acesses[m_sid]=m_stats->m_num_fp_acesses[m_sid]+(double)active_count*latency
         + inactive_lanes_accesses_nonsfu(active_count, latency);
    }else {
    m_stats->m_num_fp_acesses[m_sid]=m_stats->m_num_fp_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;     
  }
   void incfpmul_stat(unsigned active_count,double latency) {
              // printf("FP MUL stat increament\n");
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_fpmul_acesses[m_sid]=m_stats->m_num_fpmul_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_nonsfu(active_count, latency);
    }else {
    m_stats->m_num_fpmul_acesses[m_sid]=m_stats->m_num_fpmul_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
   }
   void incfpdiv_stat(unsigned active_count,double latency) {
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_fpdiv_acesses[m_sid]=m_stats->m_num_fpdiv_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_sfu(active_count, latency); 
    }else {
      m_stats->m_num_fpdiv_acesses[m_sid]=m_stats->m_num_fpdiv_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
   }
   void incdpalu_stat(unsigned active_count,double latency) {
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_dp_acesses[m_sid]=m_stats->m_num_dp_acesses[m_sid]+(double)active_count*latency
         + inactive_lanes_accesses_nonsfu(active_count, latency);
    }else {
    m_stats->m_num_dp_acesses[m_sid]=m_stats->m_num_dp_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++; 
   }
   void incdpmul_stat(unsigned active_count,double latency) {
              // printf("FP MUL stat increament\n");
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_dpmul_acesses[m_sid]=m_stats->m_num_dpmul_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_nonsfu(active_count, latency);
    }else {
    m_stats->m_num_dpmul_acesses[m_sid]=m_stats->m_num_dpmul_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
   }
   void incdpdiv_stat(unsigned active_count,double latency) {
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_dpdiv_acesses[m_sid]=m_stats->m_num_dpdiv_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_sfu(active_count, latency); 
    }else {
      m_stats->m_num_dpdiv_acesses[m_sid]=m_stats->m_num_dpdiv_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
   }

   void incsqrt_stat(unsigned active_count,double latency) {
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_sqrt_acesses[m_sid]=m_stats->m_num_sqrt_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_sfu(active_count, latency); 
    }else{
      m_stats->m_num_sqrt_acesses[m_sid]=m_stats->m_num_sqrt_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
   }

   void inclog_stat(unsigned active_count,double latency) {
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_log_acesses[m_sid]=m_stats->m_num_log_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_sfu(active_count, latency); 
    }else{
      m_stats->m_num_log_acesses[m_sid]=m_stats->m_num_log_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
   }

   void incexp_stat(unsigned active_count,double latency) {
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_exp_acesses[m_sid]=m_stats->m_num_exp_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_sfu(active_count, latency); 
    }else{
      m_stats->m_num_exp_acesses[m_sid]=m_stats->m_num_exp_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }

   void incsin_stat(unsigned active_count,double latency) {
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_sin_acesses[m_sid]=m_stats->m_num_sin_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_sfu(active_count, latency); 
    }else{
      m_stats->m_num_sin_acesses[m_sid]=m_stats->m_num_sin_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }


   void inctensor_stat(unsigned active_count,double latency) {
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_tensor_core_acesses[m_sid]=m_stats->m_num_tensor_core_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_sfu(active_count, latency); 
    }else{
      m_stats->m_num_tensor_core_acesses[m_sid]=m_stats->m_num_tensor_core_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }

  void inctex_stat(unsigned active_count,double latency) {
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_tex_acesses[m_sid]=m_stats->m_num_tex_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_sfu(active_count, latency); 
    }else{
      m_stats->m_num_tex_acesses[m_sid]=m_stats->m_num_tex_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }

  void inc_const_accesses(unsigned active_count) {
    m_stats->m_num_const_acesses[m_sid]=m_stats->m_num_const_acesses[m_sid]+active_count;
  }

  void incsfu_stat(unsigned active_count, double latency) {
    m_stats->m_num_sfu_acesses[m_sid] =
        m_stats->m_num_sfu_acesses[m_sid] + (double)active_count*latency;
  }
  void incsp_stat(unsigned active_count, double latency) {
    m_stats->m_num_sp_acesses[m_sid] =
        m_stats->m_num_sp_acesses[m_sid] + (double)active_count*latency;
  }
  void incmem_stat(unsigned active_count, double latency) {
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_mem_acesses[m_sid] =
          m_stats->m_num_mem_acesses[m_sid] + (double)active_count*latency +
          inactive_lanes_accesses_nonsfu(active_count, latency);
    } else {
      m_stats->m_num_mem_acesses[m_sid] =
          m_stats->m_num_mem_acesses[m_sid] + (double)active_count*latency;
    }
  }
  void incexecstat(warp_inst_t *&inst);

  void incregfile_reads(unsigned active_count) {
    m_stats->m_read_regfile_acesses[m_sid] =
        m_stats->m_read_regfile_acesses[m_sid] + active_count;
  }
  void incregfile_writes(unsigned active_count) {
    m_stats->m_write_regfile_acesses[m_sid] =
        m_stats->m_write_regfile_acesses[m_sid] + active_count;
  }
  void incnon_rf_operands(unsigned active_count) {
    m_stats->m_non_rf_operands[m_sid] =
        m_stats->m_non_rf_operands[m_sid] + active_count;
  }

  void incspactivelanes_stat(unsigned active_count) {
    m_stats->m_active_sp_lanes[m_sid] =
        m_stats->m_active_sp_lanes[m_sid] + active_count;
  }
  void incsfuactivelanes_stat(unsigned active_count) {
    m_stats->m_active_sfu_lanes[m_sid] =
        m_stats->m_active_sfu_lanes[m_sid] + active_count;
  }
  void incfuactivelanes_stat(unsigned active_count) {
    m_stats->m_active_fu_lanes[m_sid] =
        m_stats->m_active_fu_lanes[m_sid] + active_count;
  }
  void incfumemactivelanes_stat(unsigned active_count) {
    m_stats->m_active_fu_mem_lanes[m_sid] =
        m_stats->m_active_fu_mem_lanes[m_sid] + active_count;
  }

  void inc_simt_to_mem(unsigned n_flits) {
    m_stats->n_simt_to_mem[m_sid] += n_flits;
  }
  bool check_if_non_released_reduction_barrier(warp_inst_t &inst);

 protected:
  unsigned inactive_lanes_accesses_sfu(unsigned active_count, double latency) {
    return (((32 - active_count) >> 1) * latency) +
           (((32 - active_count) >> 3) * latency) +
           (((32 - active_count) >> 3) * latency);
  }
  unsigned inactive_lanes_accesses_nonsfu(unsigned active_count,
                                          double latency) {
    return (((32 - active_count) >> 1) * latency);
  }

  int test_res_bus(int latency);
  address_type next_pc(int tid) const;
  void fetch();
  void register_cta_thread_exit(unsigned cta_num, kernel_info_t *kernel);

  void decode();
  void PutInstInIB2(int warp);

  void mark_insts_as_completed(int warp);

  void issue();
  friend class scheduler_unit;  // this is needed to use private issue warp.
  friend class TwoLevelScheduler;
  friend class LooseRoundRobbinScheduler;
  virtual void issue_warp(register_set &warp, const warp_inst_t *pI,		
                  const active_mask_t &active_mask, unsigned warp_id,		
                  unsigned sch_id);	
  virtual void issue_warp_ibuffer_OOO_in_order(register_set &warp, const warp_inst_t *pI,		
                  const active_mask_t &active_mask, unsigned warp_id,		
                  unsigned sch_id, int sid, int m_cluster_id, int index_loc, int index_num, int num_dist, int OOO_dep, 	
                  int stalls_between_issues, int num_inst_OOO);		
  virtual void issue_warp_push_from_replay_DEB_IB_OOO(register_set &warp, const warp_inst_t *pI,	
                  const active_mask_t &active_mask, unsigned warp_id,	
                  unsigned sch_id, int sid, int MEM_ON, int m_cluster_id,int replay_index, int replay_loc, int pc, int total_tail, int cycles_spent_in_ib,
                                 int OOO_dep, int stalls_between_issues, int num_inst_OOO);
  virtual void issue_warp_push_in_DEB_IB_OOO(const warp_inst_t *pI,	
                  const active_mask_t &active_mask, unsigned warp_id,	
                  unsigned sch_id, int id, int pc, int sid, int m_cluster_id, int tail,int inst_loc, int inst_idx);	
  virtual bool isSyncInst(const warp_inst_t *inst, int warp_num);	
  virtual bool isSyncInstMemory(const warp_inst_t *inst, int warp_num);	
  virtual bool isSyncInstNonMemory(const warp_inst_t *inst, int warp_num);	
  virtual bool isSyncInstCore(const warp_inst_t *inst, int warp_num) = 0;		
  virtual bool isSyncInstCoreMemory(const warp_inst_t *inst, int warp_num) = 0;		
  virtual bool isSyncInstCoreNonMemory(const warp_inst_t *inst, int warp_num) = 0;		
  virtual void func_exec_inst_updatePCOnly(warp_inst_t &inst, int warp_num) = 0;		
  virtual void set_up_func_exec_inst_for_addr(warp_inst_t &inst) = 0;	
  // executes the inst, does not update any PC information, executed when pushing instruction out 		
  // of OOO (replay) queue		
  virtual void func_exec_inst_ExecInstOnly(warp_inst_t &inst, int warp_num) = 0;

  void create_front_pipeline();
  void create_schedulers();
  void create_exec_pipeline();

  // pure virtual methods implemented based on the current execution mode
  // (execution-driven vs trace-driven)
  virtual void init_warps(unsigned cta_id, unsigned start_thread,
                          unsigned end_thread, unsigned ctaid, int cta_size,
                          kernel_info_t &kernel);
  virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                             unsigned tid) = 0;
  virtual void checkExecutionStatusAndUpdate_for_pc(warp_inst_t &inst, unsigned t,	
                                             unsigned tid) = 0;
  virtual void func_exec_inst(warp_inst_t &inst) = 0;

  virtual unsigned sim_init_thread(kernel_info_t &kernel,
                                   ptx_thread_info **thread_info, int sid,
                                   unsigned tid, unsigned threads_left,
                                   unsigned num_threads, core_t *core,
                                   unsigned hw_cta_id, unsigned hw_warp_id,
                                   gpgpu_t *gpu) = 0;

  virtual void create_shd_warp() = 0;

  virtual const warp_inst_t *get_next_inst(unsigned warp_id,
                                           address_type pc) = 0;
  virtual void get_pdom_stack_top_info(unsigned warp_id, const warp_inst_t *pI,
                                       unsigned *pc, unsigned *rpc) = 0;
  virtual const active_mask_t &get_active_mask(unsigned warp_id,
                                               const warp_inst_t *pI) = 0;
  virtual const active_mask_t &get_active_mask_test(unsigned warp_id,	
                                               const warp_inst_t *pI) = 0;

  // Returns numbers of addresses in translated_addrs
  unsigned translate_local_memaddr(address_type localaddr, unsigned tid,
                                   unsigned num_shader, unsigned datasize,
                                   new_addr_type *translated_addrs);

  void read_operands();

  void execute();

  void writeback();

  // used in display_pipeline():
  void dump_warp_state(FILE *fout) const;
  void print_stage(unsigned int stage, FILE *fout) const;

  unsigned long long m_last_inst_gpu_sim_cycle;
  unsigned long long m_last_inst_gpu_tot_sim_cycle;

  // general information
  unsigned m_sid;  // shader id
  unsigned m_tpc;  // texture processor cluster id (aka, node id when using
                   // interconnect concentration)
  const shader_core_config *m_config;
  const memory_config *m_memory_config;
  class simt_core_cluster *m_cluster;

  // statistics
  shader_core_stats *m_stats;

  // CTA scheduling / hardware thread allocation
  unsigned m_n_active_cta;  // number of Cooperative Thread Arrays (blocks)
                            // currently running on this shader.
  unsigned m_cta_status[MAX_CTA_PER_SHADER];  // CTAs status
  unsigned m_not_completed;  // number of threads to be completed (==0 when all
                             // thread on this core completed)
  std::bitset<MAX_THREAD_PER_SM> m_active_threads;

  // thread contexts
  thread_ctx_t *m_threadState;

  // interconnect interface
  mem_fetch_interface *m_icnt;
  shader_core_mem_fetch_allocator *m_mem_fetch_allocator;

  // fetch
  read_only_cache *m_L1I;  // instruction cache
  int m_last_warp_fetched;

  // decode/dispatch
  std::vector<shd_warp_t *> m_warp;  // per warp information array
  barrier_set_t m_barriers;
  ifetch_buffer_t m_inst_fetch_buffer;
  std::vector<register_set> m_pipeline_reg;
  Scoreboard *m_scoreboard;
  opndcoll_rfu_t m_operand_collector;
  int m_active_warps;
  std::vector<register_set *> m_specilized_dispatch_reg;

  // schedule
  std::vector<scheduler_unit *> schedulers;

  // issue
  unsigned int Issue_Prio;

  // execute
  unsigned m_num_function_units;
  std::vector<unsigned> m_dispatch_port;
  std::vector<unsigned> m_issue_port;
  std::vector<simd_function_unit *>
      m_fu;  // stallable pipelines should be last in this array
  ldst_unit *m_ldst_unit;
  static const unsigned MAX_ALU_LATENCY = 512;
  unsigned num_result_bus;
  std::vector<std::bitset<MAX_ALU_LATENCY> *> m_result_bus;

  // used for local address mapping with single kernel launch
  unsigned kernel_max_cta_per_shader;
  unsigned kernel_padded_threads_per_cta;
  // Used for handing out dynamic warp_ids to new warps.
  // the differnece between a warp_id and a dynamic_warp_id
  // is that the dynamic_warp_id is a running number unique to every warp
  // run on this shader, where the warp_id is the static warp slot.
  unsigned m_dynamic_warp_id;

  // Jin: concurrent kernels on a sm
 public:
  bool can_issue_1block(kernel_info_t &kernel);
  bool occupy_shader_resource_1block(kernel_info_t &kernel, bool occupy);
  void release_shader_resource_1block(unsigned hw_ctaid, kernel_info_t &kernel);
  int find_available_hwtid(unsigned int cta_size, bool occupy);

 private:
  unsigned int m_occupied_n_threads;
  unsigned int m_occupied_shmem;
  unsigned int m_occupied_regs;
  unsigned int m_occupied_ctas;
  std::bitset<MAX_THREAD_PER_SM> m_occupied_hwtid;
  std::map<unsigned int, unsigned int> m_occupied_cta_to_hwtid;
};

class exec_shader_core_ctx : public shader_core_ctx {
 public:
  exec_shader_core_ctx(class gpgpu_sim *gpu, class simt_core_cluster *cluster,
                       unsigned shader_id, unsigned tpc_id,
                       const shader_core_config *config,
                       const memory_config *mem_config,
                       shader_core_stats *stats)
      : shader_core_ctx(gpu, cluster, shader_id, tpc_id, config, mem_config,
                        stats) {
    create_front_pipeline();
    create_shd_warp();
    create_schedulers();
    create_exec_pipeline();
  }

  virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                             unsigned tid);
  virtual void checkExecutionStatusAndUpdate_for_pc(warp_inst_t &inst, unsigned t,	
                                             unsigned tid);
  virtual void func_exec_inst(warp_inst_t &inst);
  virtual void set_up_func_exec_inst_for_addr(warp_inst_t &inst);	
  virtual bool isSyncInstCore(const warp_inst_t* inst, int warp_num);		
  virtual bool isSyncInstCoreMemory(const warp_inst_t* inst, int warp_num);		
  virtual bool isSyncInstCoreNonMemory(const warp_inst_t* inst, int warp_num);		
  // only updates if PC was last PC to execute for fetch to be correct 		
  // Executed when pushing instruction from normal queue to OOO (replay) queue		
  virtual void func_exec_inst_updatePCOnly(warp_inst_t &inst, int warp_num);		
  // executes the inst, does not update any PC information, executed when pushing instruction out 		
  // of OOO (replay) queue		
  virtual void func_exec_inst_ExecInstOnly(warp_inst_t &inst, int warp_num);
  virtual unsigned sim_init_thread(kernel_info_t &kernel,
                                   ptx_thread_info **thread_info, int sid,
                                   unsigned tid, unsigned threads_left,
                                   unsigned num_threads, core_t *core,
                                   unsigned hw_cta_id, unsigned hw_warp_id,
                                   gpgpu_t *gpu);
  virtual void create_shd_warp();
  virtual const warp_inst_t *get_next_inst(unsigned warp_id, address_type pc);
  virtual void get_pdom_stack_top_info(unsigned warp_id, const warp_inst_t *pI,
                                       unsigned *pc, unsigned *rpc);
  virtual const active_mask_t &get_active_mask(unsigned warp_id,
                                               const warp_inst_t *pI);
  virtual const active_mask_t &get_active_mask_test(unsigned warp_id,	
                                               const warp_inst_t *pI);
};

class simt_core_cluster {
 public:
  simt_core_cluster(class gpgpu_sim *gpu, unsigned cluster_id,
                    const shader_core_config *config,
                    const memory_config *mem_config, shader_core_stats *stats,
                    memory_stats_t *mstats);

  void core_cycle();
  void icnt_cycle();

  void reinit();
  unsigned issue_block2core();
  void cache_flush();
  void cache_invalidate();
  bool icnt_injection_buffer_full(unsigned size, bool write);
  void icnt_inject_request_packet(class mem_fetch *mf);

  // for perfect memory interface
  bool response_queue_full() {
    return (m_response_fifo.size() >= m_config->n_simt_ejection_buffer_size);
  }
  void push_response_fifo(class mem_fetch *mf) {
    m_response_fifo.push_back(mf);
  }

  void get_pdom_stack_top_info(unsigned sid, unsigned tid, unsigned *pc,
                               unsigned *rpc) const;
  unsigned max_cta(const kernel_info_t &kernel);
  unsigned get_not_completed() const;
  void print_not_completed(FILE *fp) const;
  unsigned get_n_active_cta() const;
  unsigned get_n_active_sms() const;
  gpgpu_sim *get_gpu() { return m_gpu; }

  void display_pipeline(unsigned sid, FILE *fout, int print_mem, int mask);
  void print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                         unsigned &dl1_misses) const;

  void get_cache_stats(cache_stats &cs) const;
  void get_L1I_sub_stats(struct cache_sub_stats &css) const;
  void get_L1D_sub_stats(struct cache_sub_stats &css) const;
  void get_L1C_sub_stats(struct cache_sub_stats &css) const;
  void get_L1T_sub_stats(struct cache_sub_stats &css) const;

  void get_icnt_stats(long &n_simt_to_mem, long &n_mem_to_simt) const;
  float get_current_occupancy(unsigned long long &active,
                              unsigned long long &total) const;
  virtual void create_shader_core_ctx() = 0;
  unsigned m_cluster_id;


 protected:
  gpgpu_sim *m_gpu;
  const shader_core_config *m_config;
  shader_core_stats *m_stats;
  memory_stats_t *m_memory_stats;
  shader_core_ctx **m_core;
  const memory_config *m_mem_config;

  unsigned m_cta_issue_next_core;
  std::list<unsigned> m_core_sim_order;
  std::list<mem_fetch *> m_response_fifo;
};

class exec_simt_core_cluster : public simt_core_cluster {
 public:
  exec_simt_core_cluster(class gpgpu_sim *gpu, unsigned cluster_id,
                         const shader_core_config *config,
                         const memory_config *mem_config,
                         class shader_core_stats *stats,
                         class memory_stats_t *mstats)
      : simt_core_cluster(gpu, cluster_id, config, mem_config, stats, mstats) {
    create_shader_core_ctx();
  }

  virtual void create_shader_core_ctx();
};

class shader_memory_interface : public mem_fetch_interface {
 public:
  shader_memory_interface(shader_core_ctx *core, simt_core_cluster *cluster) {
    m_core = core;
    m_cluster = cluster;
  }
  virtual bool full(unsigned size, bool write) const {
    return m_cluster->icnt_injection_buffer_full(size, write);
  }
  virtual void push(mem_fetch *mf) {
    m_core->inc_simt_to_mem(mf->get_num_flits(true));
    m_cluster->icnt_inject_request_packet(mf);
  }

 private:
  shader_core_ctx *m_core;
  simt_core_cluster *m_cluster;
};

class perfect_memory_interface : public mem_fetch_interface {
 public:
  perfect_memory_interface(shader_core_ctx *core, simt_core_cluster *cluster) {
    m_core = core;
    m_cluster = cluster;
  }
  virtual bool full(unsigned size, bool write) const {
    return m_cluster->response_queue_full();
  }
  virtual void push(mem_fetch *mf) {
    if (mf && mf->isatomic())
      mf->do_atomic();  // execute atomic inside the "memory subsystem"
    m_core->inc_simt_to_mem(mf->get_num_flits(true));
    m_cluster->push_response_fifo(mf);
  }

 private:
  shader_core_ctx *m_core;
  simt_core_cluster *m_cluster;
};

inline int scheduler_unit::get_sid() const { return m_shader->get_sid(); }

#endif /* SHADER_H */
