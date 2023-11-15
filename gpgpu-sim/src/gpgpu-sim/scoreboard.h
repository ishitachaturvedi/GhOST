// Copyright (c) 2009-2011, Tor M. Aamodt, Inderpreet Singh
// The University of British Columbia
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

#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <vector>
#include "assert.h"
#include "fast.h"
#include <queue>

#ifndef SCOREBOARD_H_
#define SCOREBOARD_H_

#include "../abstract_hardware_model.h"

class Scoreboard {
 public:
  Scoreboard(unsigned sid, unsigned n_warps, class gpgpu_t *gpu);

  void reserveRegisters(const warp_inst_t *inst, bool gpgpu_perfect_mem_data, int status, int m_cluster_id, int sid, int instNum, int numStalls);
  void releaseRegisters(const warp_inst_t *inst);
  void releaseRegister(unsigned wid, unsigned regnum);
  void releaseRegisterWAR(unsigned wid, unsigned regnum);

  bool checkCollision(unsigned wid, const inst_t *inst, bool print, bool reg_renaming) const;

  void reg_values(unsigned wid) const;
  //bool checkCollisionAddr(unsigned wid, const inst_t *inst, bool print) const;
  void get_closest_dependence(unsigned wid, const inst_t *inst1, bool print, int m_cluster_id, int sid, int OOO_dep, int num_inst_OOO, unsigned stalls_between_issues, int isOOO);
  bool pendingWrites(unsigned wid) const;
  bool pendingWrites(unsigned wid, bool ignore) const;
  void printContents() const;
  const bool islongop(unsigned warp_id, unsigned regnum);
  const bool islongop_hold(unsigned warp_id, const class warp_inst_t* inst);

  bool check_for_WAW_deps(unsigned wid, const class inst_t* inst) const;
  bool check_WAR_or_WAW_replay(unsigned wid, const class inst_t* inst, std::vector<const warp_inst_t *> replayInst) const;

  /* Added Functions */

  bool checkReplayCollision(unsigned wid, const inst_t *inst, std::vector<const warp_inst_t *> replayInst) const;

  int getIndependentInstructionCount(unsigned wid, const inst_t *inst, std::vector<const warp_inst_t *> replayInst) const;

  void reserveRegistersMem(const warp_inst_t *inst);
  void releaseRegistersMem(const warp_inst_t *inst,int val);
  void releaseRegisterMem(unsigned wid, unsigned regnum,int val,int op, int type);

  void reserveRegistersComp(const warp_inst_t *inst);
  void releaseRegistersComp(const warp_inst_t *inst);
  void releaseRegisterComp(unsigned wid, unsigned regnum);

  void appendMemStatus(warp_inst_t &inst, int type);

  bool pendingWritesMem(unsigned wid) const;
  bool pendingWritesComp(unsigned wid) const;

  void releaseAllBBRegs(int wid);
  void addWriteRegs(const class warp_inst_t* inst, int wid);

  bool checkIsIdempotent(unsigned wid, const class inst_t* inst) const;

  void RenameRegs(unsigned wid, class inst_t* inst, bool print, bool reg_renaming,std::vector<const warp_inst_t *> replayInst);

  std::vector<int> checkCollisionMem(unsigned wid, const inst_t *inst) const;
  std::vector<int> checkCollisionComp(unsigned wid, const inst_t *inst) const;

  void collisionInstPC(unsigned wid, const inst_t* inst, bool print, int pc, int m_cluster_id, int sid, std::vector<const warp_inst_t *> replayInst);

  bool checkConsecutiveInstIndep(const inst_t *pI, const inst_t *last_exec_inst) const;

  void set_num_cycles_deocode_issue(int num_cycles, const class warp_inst_t* inst);

  void resetValuesOfScoreboard(int warp_id);

  void get_dep_distance(unsigned wid, const warp_inst_t *inst1, int m_cluster_id, int sid, int stalls_between_issues, int OOO, int instNum, int numStalls);

 private:
  void reserveRegister(unsigned wid, unsigned regnum, bool gpgpu_perfect_mem_data, int pc, int m_cluster_id, int sid, int stalls_between_issues, int inst_num);
  void addWriteReg(unsigned wid, unsigned regnum);
  void reserveRegisterWAR(unsigned wid, unsigned regnum, bool gpgpu_perfect_mem_data, int pc, int m_cluster_id, int sid, int stalls_between_issues, int inst_num);
  int get_sid() const { return m_sid; }

  unsigned m_sid;

  // keeps track of pending writes to registers
  // indexed by warp id, reg_id => pending write count
  std::vector<std::set<unsigned> > reg_table;
  //std::vector<std::set<unsigned> > reg_table_WAR;
  std::vector<std::set<unsigned> > addr_table;
  std::vector<std::set<unsigned> > reg_table_all_regs_used_list;
  std::vector<std::set<unsigned> > write_regs_in_BB;
  // Register that depend on a long operation (global, local or tex memory)
  std::vector<std::set<unsigned> > longopregs;

  void reserveRegisterMem(unsigned wid, unsigned regnum, bool is_load);
  void reserveRegisterComp(unsigned wid, unsigned regnum);

  //keep track of pending writes to memory operations
  std::vector<std::set<unsigned> > reg_table_mem;
  //keep track of pending writes to computation operations
  std::vector<std::set<unsigned> > reg_table_comp;
  std::vector<unsigned> warp_inst_issue_num;

  std::vector<std::vector<int>> arch_reg; // registers which were renamed
  std::vector<std::vector<int>> physical_reg; // What the regs were renamed to 
  std::vector<std::vector<int>> free_regs;

  std::vector<int> last_physical_reg; // last physical reg allocated
  //int last_physical_reg;

  // Data structure to store reserve cycle plus reserving warp
  std::vector<std::map<unsigned, int>> reg_reserved_mem;
  std::vector<std::map<unsigned, int>> reg_released_mem;

  std::vector<std::map<unsigned, int>> reg_reserved_comp;
  std::vector<std::map<unsigned, int>> reg_released_comp;

  std::vector<std::map<unsigned, int>> reg_reserved_type; // type of reserved instruction (mem or comp)
  std::vector<std::map<unsigned, int>> reg_table_all_regs_used;

  std::vector<std::map<unsigned, int>> reg_reserved;

  std::vector<std::map<unsigned, int>> reg_table_WAR;

  std::vector<std::map<unsigned, int>> reg_pc;
  std::vector<std::map<unsigned, int>> reg_released;

  std::vector<std::map<unsigned, int>> num_cycles_decode_issue;

  //std::vector<std::map<unsigned, int>> reg_reserved_inst;

  std::vector<vector<int>> reg_reserved_inst;

  std::vector<std::map<unsigned, int>> reg_type_mem;

  std::vector<std::map<unsigned, int>> reg_load_type;

  class gpgpu_t *m_gpu;
};

#endif /* SCOREBOARD_H_ */
